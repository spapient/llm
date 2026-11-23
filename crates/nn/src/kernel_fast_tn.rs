//! 线性层布局（TN）的稠密 f32 GEMM：`out[rows,n] = x[rows,k] · w[n,k]ᵀ`。
use tensor::View;

use super::pack::{group_panel, pack_panel};
use super::pool::{Bufs, Shape, pool, run_pooled};
use super::{Layout, MR, NR, align_up, threads_override};

/// 走这条入口的最少行数：一个完整微内核行块。再少的批只有残行（标量），打包的转置成本摊不回来，
/// 调用方应当继续用按列驻留的 tile 核。
pub const MIN_ROWS: usize = MR;

/// 多线程 TN 稠密 GEMM：`out[rows,n] = x[rows,k] · w[n,k]ᵀ`。
///
/// 语义同 [`crate::matmul::forward_tn_tiled`]（同一组累加口径，IEEE 容差内一致），吞吐按父模块的
/// 8×8 寄存器分块 + 常驻池给。
///
/// # Panics
///
/// 三条缓冲长度不足 `rows·k` / `n·k` / `rows·n` 时 panic（调用方先校验）。
pub fn gemm_f32_tn(x: View<'_, f32>, w: View<'_, f32>, out: &mut [f32]) {
    let (rows, k) = crate::kernel::matmul::dim2(x.dims(), "gemm_f32_tn").expect("二维输入");
    let (n, k2) = crate::kernel::matmul::dim2(w.dims(), "gemm_f32_tn").expect("二维输入");
    assert_eq!(k, k2, "gemm_f32_tn 收缩轴不符");
    let (x, w) = (x.as_slice(), w.as_slice());
    check_dims_tn(x, w, rows, n, k, out);
    let t = threads_override().min(rows.div_ceil(MR)).max(1);
    if t <= 1 {
        single_tn(x, w, rows, n, k, out);
        return;
    }
    if !pooled_tn(x, w, rows, n, k, out, t) {
        scoped_tn(x, w, out, rows, n, k, t);
    }
}

/// 尺寸校验：三条缓冲不足 `rows·k` / `n·k` / `rows·n` 时 panic。
fn check_dims_tn(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &[f32]) {
    assert!(
        x.len() >= rows * k && w.len() >= n * k && out.len() >= rows * n,
        "gemm_f32_tn 尺寸不符"
    );
}

/// 单线程：整段打包一次，再按行块算完整段 K。
fn single_tn(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
    let npad = align_up(n, NR);
    super::with_panels(npad * k, |bpack| {
        pack_tn_into(bpack, w, n, k);
        compute_tn(x, bpack, w, rows, n, k, out);
    });
}

/// 常驻池路径；池不可用（参与数不足 2）时返回 `false`，由 [`scoped_tn`] 兜底。
fn pooled_tn(
    x: &[f32],
    w: &[f32],
    rows: usize,
    n: usize,
    k: usize,
    out: &mut [f32],
    t: usize,
) -> bool {
    let npad = align_up(n, NR);
    super::with_panels(npad * k, |buf| {
        run_pooled(
            pool(),
            Bufs {
                a: x.as_ptr() as *const u8,
                b: w.as_ptr() as *const u8,
                buf: buf.as_mut_ptr() as *mut u8,
                out: out.as_mut_ptr() as *mut u8,
            },
            Shape {
                m: rows,
                n,
                k,
                npad,
                tn: true,
            },
            MR,
            t,
            super::f32_pack,
            super::f32_calc,
        )
    })
}

/// 兜底（池不可用）：打包一次后 shared 给 scope 线程，行块按 MR 对齐切。
fn scoped_tn(x: &[f32], w: &[f32], out: &mut [f32], rows: usize, n: usize, k: usize, t: usize) {
    let npad = align_up(n, NR);
    let per = rows.div_ceil(MR).div_ceil(t) * MR;
    super::with_panels(npad * k, |bpack| {
        pack_tn_into(bpack, w, n, k);
        let bp: &[f32] = bpack;
        std::thread::scope(|s| {
            for (ti, block) in out[..rows * n].chunks_mut(per * n).enumerate() {
                let r0 = ti * per;
                let got = block.len() / n;
                let a_l = &x[r0 * k..(r0 + got) * k];
                s.spawn(move || compute_tn(a_l, bp, w, got, n, k, block));
            }
        });
    });
}

/// 整段打包（单槽、兜底路径与单线程用）：面板布局与父模块一致，只是源按 `[n][k]` 读。
fn pack_tn_into(bpack: &mut [f32], w: &[f32], n: usize, k: usize) {
    let npad = align_up(n, NR);
    debug_assert_eq!(bpack.len(), npad * k);
    for g in 0..npad / NR {
        pack_panel(w, bpack, n, k, g * NR, g * k * NR, true);
    }
}

/// 本槽负责的列组按 `p` 步长轮转，别槽的列组由别槽打包（与父模块同一套分工）。
pub(super) fn stage(
    w: &[f32],
    buf: &mut [f32],
    n: usize,
    k: usize,
    npad: usize,
    p: usize,
    slot: usize,
) {
    group_panel::<f32, NR, false>(w, buf, n, k, npad, p, slot, true);
}

/// 一段行的计算：A 面板打包 + 按 [`KB`] 分块扫列 + 行残尾。
///
/// 正文与父模块的 Nt 视角**共用一份**（`fast::compute_rows`），这里只钉住 `Layout::Tn`。
pub(super) fn compute_tn(
    a: &[f32],
    bpack: &[f32],
    w: &[f32],
    rows: usize,
    n: usize,
    k: usize,
    out: &mut [f32],
) {
    super::compute_rows(a, bpack, w, super::Extent { rows, n, k }, out, Layout::Tn);
}

#[cfg(test)]
mod tests {
    include!("../tests/kernel_tn_tests.rs");
}
