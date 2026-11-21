//! 线性层布局（TN）的稠密 f32 GEMM：`out[rows,n] = x[rows,k] · w[n,k]ᵀ`。
//!
//! 与父模块是同一套 8×8 微内核与同一条常驻池，差别只在一处：权重按**输出行**存放（`w[j][q]`），
//! 而微内核要的是「每个 `q` 上连续 8 个输出列」。所以打包阶段要做一次转置——`w` 的 8 行、每行
//! 一段 `k`，转成面板 `[q][8]`；这也正是打包本来就要做的重排，只是读的步长从 `n` 变成 `k`。
//!
//! 存在的理由：prefill 的投影走的就是这份布局（权重按输出行常驻），而父模块只认收缩轴在前的
//! 权重。没有这个入口，服务端只能退回逐列 `forward_tn_tiled`——那份内核每个权重元素被读一遍没错，
//! 但没有寄存器分块与多线程，实测差 20 倍以上。
use super::pack::pack_panels;
use super::pool::{Buffers, Shape, pool, run_pooled};
use super::{KB, Layout, MR, NR, Span, Sweep, edge, sweep_columns, threads_override};

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
pub fn gemm_f32_tn(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
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
    let npad = n.div_ceil(NR) * NR;
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
    let npad = n.div_ceil(NR) * NR;
    super::with_panels(npad * k, |buf| {
        run_pooled(
            pool(),
            Buffers {
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
    let npad = n.div_ceil(NR) * NR;
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
///
/// 写进调用方的暂存区（已清零），不在这里分配：`n = k = 4096` 时这块是 67 MB。
fn pack_tn_into(bpack: &mut [f32], w: &[f32], n: usize, k: usize) {
    let npad = n.div_ceil(NR) * NR;
    debug_assert_eq!(bpack.len(), npad * k);
    for g in 0..npad / NR {
        let base = g * k * NR;
        fill_panel(&mut bpack[base..base + k * NR], w, n, k, g * NR);
    }
}

/// 一个列组的转置打包：`w` 的 `first..first+NR` 行、每行 `k` 个元素 → 面板 `[q][NR]`。
///
/// 读是 8 条步长 `k` 的流、写是连续面板；`k` 远大于 cache line 时读端是 8 条并发预取流，
/// 与父模块「读 8 条连续、写跨步」正好互补，实际耗时同一量级。
fn fill_panel(panel: &mut [f32], w: &[f32], n: usize, k: usize, first: usize) {
    let cols = if first < n { NR.min(n - first) } else { 0 };
    for q in 0..k {
        let dst = &mut panel[q * NR..q * NR + cols];
        for (c, slot) in dst.iter_mut().enumerate() {
            *slot = w[(first + c) * k + q];
        }
    }
}

/// 本槽负责的列组按 `p` 步长轮转，别槽的列组由别槽打包（与父模块同一套分工）。
///
/// 读 `w` 的步长是 `k`（按输出行存放），写面板是连续段：这正是与行主序打包唯一的差别。
pub(super) fn stage(
    w: &[f32],
    buf: &mut [f32],
    n: usize,
    k: usize,
    npad: usize,
    p: usize,
    slot: usize,
) {
    let groups = npad / NR;
    let mut g = slot;
    while g < groups {
        let base = g * k * NR;
        fill_panel(&mut buf[base..base + k * NR], w, n, k, g * NR);
        g += p;
    }
}

/// 一段行的计算：A 面板打包 + 按 [`KB`] 分块扫列 + 行残尾。
///
/// 与父模块的 `compute` 逐行对应，只有两处同构偏移：完整列块的微内核吃的是同一份 B 面板，
/// 所以直接复用；列残尾与行残尾要对**原始权重**取值，那边索引从 `w[q*n + j]` 换成 `w[j*k + q]`。
pub(super) fn compute_tn(
    a: &[f32],
    bpack: &[f32],
    w: &[f32],
    rows: usize,
    n: usize,
    k: usize,
    out: &mut [f32],
) {
    let full = rows / MR;
    let apack = pack_panels(a, full, k);
    let mut first = true;
    let mut k0 = 0;
    while k0 < k {
        let kb = KB.min(k - k0);
        sweep_columns(
            Sweep {
                a,
                apack: &apack,
                bpack,
                w,
                full,
                n,
                k,
                layout: Layout::Tn,
            },
            out,
            k0,
            kb,
            first,
        );
        first = false;
        k0 += kb;
    }
    // 残行（≤ MR−1 行）一次走完整段 K，不参与分块累加——与父模块同一条边界。
    let tail = full * MR;
    if tail < rows {
        edge(
            a,
            w,
            out,
            Span {
                i: tail,
                j: 0,
                mr: rows - tail,
                nr: n,
                n,
                k,
                stride: k,
            },
            0,
            true,
            Layout::Tn,
        );
    }
}

#[cfg(test)]
mod tests {
    include!("../../tests/tn_tests.rs");
}
