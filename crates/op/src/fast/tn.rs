//! 线性层布局（TN）的稠密 f32 GEMM：`out[rows,n] = x[rows,k] · w[n,k]ᵀ`。
//!
//! 与父模块是同一套 8×8 微内核与同一条常驻池，差别只在一处：权重按**输出行**存放（`w[j][q]`），
//! 而微内核要的是「每个 `q` 上连续 8 个输出列」。所以打包阶段要做一次转置——`w` 的 8 行、每行
//! 一段 `k`，转成面板 `[q][8]`；这也正是打包本来就要做的重排，只是读的步长从 `n` 变成 `k`。
//!
//! 存在的理由：prefill 的投影走的就是这份布局（权重按输出行常驻），而父模块只认收缩轴在前的
//! 权重。没有这个入口，服务端只能退回逐列 `forward_tn_tiled`——那份内核每个权重元素被读一遍没错，
//! 但没有寄存器分块与多线程，实测差 20 倍以上。

use super::pool::{Job, Shape, pool, run_pooled};
use super::scratch;
use super::{KB, MR, NR, Span, micro_packed, pack_panels, threads_override};

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
    assert!(
        x.len() >= rows * k && w.len() >= n * k && out.len() >= rows * n,
        "gemm_f32_tn 尺寸不符"
    );
    let blocks = rows.div_ceil(MR);
    let t = threads_override().min(blocks).max(1);
    let npad = n.div_ceil(NR) * NR;
    if t <= 1 {
        scratch::with_panels(npad * k, |bpack| {
            pack_tn_into(bpack, w, n, k);
            compute_tn(x, bpack, w, rows, n, k, out);
        });
        return;
    }
    let pooled = scratch::with_panels(npad * k, |buf| {
        run_pooled(
            pool(),
            x.as_ptr() as *const u8,
            w.as_ptr() as *const u8,
            buf.as_mut_ptr() as *mut u8,
            out.as_mut_ptr() as *mut u8,
            Shape {
                m: rows,
                n,
                k,
                npad,
            },
            MR,
            t,
            tn_stage,
            tn_calc,
        )
    });
    if pooled {
        return;
    }
    // 兜底（池不可用）：打包一次后 shared 给 scope 线程，行块按 MR 对齐切。
    let per = blocks.div_ceil(t) * MR;
    scratch::with_panels(npad * k, |bpack| {
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
fn tn_stage(j: &Job, slot: usize) {
    let p = j.p;
    let groups = j.npad / NR;
    #[allow(unsafe_code)]
    unsafe {
        let w = std::slice::from_raw_parts(j.b as *const f32, j.n * j.k);
        let buf = std::slice::from_raw_parts_mut(j.buf as *mut f32, j.npad * j.k);
        let mut g = slot;
        while g < groups {
            let base = g * j.k * NR;
            fill_panel(&mut buf[base..base + j.k * NR], w, j.n, j.k, g * NR);
            g += p;
        }
    }
}

/// 本槽的行段：面板由闸门保证整块就绪后再算。
fn tn_calc(j: &Job, slot: usize) {
    let r0 = slot * j.rows_per;
    if r0 >= j.m {
        return;
    }
    let rows = (j.m - r0).min(j.rows_per);
    #[allow(unsafe_code)]
    unsafe {
        let x = std::slice::from_raw_parts(j.a as *const f32, j.m * j.k);
        let a_l = &x[r0 * j.k..(r0 + rows) * j.k];
        let bp = std::slice::from_raw_parts(j.buf as *const f32, j.npad * j.k);
        let w = std::slice::from_raw_parts(j.b as *const f32, j.n * j.k);
        let out = std::slice::from_raw_parts_mut(j.out as *mut f32, j.m * j.n);
        let out_l = &mut out[r0 * j.n..(r0 + rows) * j.n];
        compute_tn(a_l, bp, w, rows, j.n, j.k, out_l);
    }
}

/// 一段行的计算：A 面板打包 + 按 [`KB`] 分块扫列 + 行残尾。
///
/// 与父模块的 `compute` 逐行对应，只有两处同构偏移：完整列块的微内核吃的是同一份 B 面板，
/// 所以直接复用；列残尾与行残尾要对**原始权重**取值，那边索引从 `w[q*n + j]` 换成 `w[j*k + q]`。
fn compute_tn(
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
        sweep_columns_tn(a, &apack, bpack, w, full, n, k, out, k0, kb, first);
        first = false;
        k0 += kb;
    }
    // 残行（≤ MR−1 行）一次走完整段 K，不参与分块累加——与父模块同一条边界。
    let tail = full * MR;
    if tail < rows {
        edge_tn(
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
        );
    }
}

/// 按列扫：完整 8 列走微内核，列残尾（`n % NR`）走 [`edge_tn`]。
#[allow(clippy::too_many_arguments)]
fn sweep_columns_tn(
    a: &[f32],
    apack: &[f32],
    bpack: &[f32],
    w: &[f32],
    full: usize,
    n: usize,
    k: usize,
    out: &mut [f32],
    k0: usize,
    kb: usize,
    first: bool,
) {
    let mut j = 0;
    while j < n {
        let nr = NR.min(n - j);
        if nr == NR {
            let start = (j / NR) * k * NR + k0 * NR;
            let tile = &bpack[start..start + kb * NR];
            for blk in 0..full {
                micro_packed(
                    &apack[blk * k * MR + k0 * MR..],
                    tile,
                    out,
                    blk * MR,
                    j,
                    n,
                    kb,
                    first,
                );
            }
        } else {
            edge_tn(
                a,
                w,
                out,
                Span {
                    i: 0,
                    j,
                    mr: full * MR,
                    nr,
                    n,
                    k: kb,
                    stride: k,
                },
                k0,
                first,
            );
        }
        j += nr;
    }
}

/// 标量尾块的 TN 口径：与父模块的 `edge` 同一组累加次序，只有权重取值改成 `w[j][q]`。
fn edge_tn(a: &[f32], w: &[f32], out: &mut [f32], span: Span, k0: usize, first: bool) {
    for r in 0..span.mr {
        for c in 0..span.nr {
            let slot = (span.i + r) * span.n + span.j + c;
            let mut acc = if first { 0.0f32 } else { out[slot] };
            for p in 0..span.k {
                acc +=
                    a[(span.i + r) * span.stride + k0 + p] * w[(span.j + c) * span.stride + k0 + p];
            }
            out[slot] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    /// 与非转置入口逐位对拍：同一组 `(rows, n, k)` 下，把 `w` 转置成 NT 布局后两边结果必须相等
    /// （累加次序完全相同，只有权重寻址方式不同）。
    #[test]
    fn tn_matches_transposed_nt_bit_patterns() {
        for (rows, n, k) in [(1, 8, 64), (7, 13, 33), (16, 24, 256), (23, 8, 300)] {
            let x: Vec<f32> = (0..rows * k)
                .map(|i| ((i * 3) % 17) as f32 * 0.25 - 2.0)
                .collect();
            let w: Vec<f32> = (0..n * k)
                .map(|i| ((i * 5) % 11) as f32 * 0.5 - 2.5)
                .collect();
            let mut wt = vec![0.0f32; k * n];
            for j in 0..n {
                for q in 0..k {
                    wt[q * n + j] = w[j * k + q];
                }
            }
            let mut got = vec![0.0f32; rows * n];
            let mut want = vec![0.0f32; rows * n];
            super::gemm_f32_tn(&x, &w, rows, n, k, &mut got);
            super::super::gemm_f32(&x, &wt, rows, n, k, &mut want);
            assert_eq!(got, want, "TN 与 NT（转置后）应当逐位相同：{rows}×{n}×{k}");
        }
    }
}
