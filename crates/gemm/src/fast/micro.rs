//! 职责：微内核整行块扫描的只读侧（Sweep）。
use super::{MR, NR};

/// 一次整行块扫描的只读侧：指针、面板与三个尺寸一次带走（实参不散成一长串）。
#[derive(Clone, Copy)]
pub(super) struct Sweep<'a> {
    pub(super) a: &'a [f32],
    pub(super) apack: &'a [f32],
    pub(super) bpack: &'a [f32],
    pub(super) w: &'a [f32],
    pub(super) full: usize,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) layout: Layout,
}

/// 一个待算的 8×8 微内核块：左上角 `(row0, j)`、矩阵列宽 `n`、内积分块长 `kb`、是否首块。
#[derive(Clone, Copy)]
pub(super) struct Tile {
    pub(super) row0: usize,
    pub(super) j: usize,
    pub(super) n: usize,
    pub(super) kb: usize,
    pub(super) first: bool,
}

/// 打包面板上的 NEON 8×8 微内核（A 面板 `[p][8]`、B 面板 `[p][8]`，一次算 8 行 8 列）。
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(super) fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
    let Tile {
        row0,
        j,
        n,
        kb,
        first,
    } = tile;
    use std::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let pp = panel.as_ptr();
        let bp = b.as_ptr();
        // 后续内积分块接着上一块的累加器算：首块用 0 起手，否则读回 `out` 的 16 个向量。
        let mut acc = if first {
            [zero; 16]
        } else {
            micro_load(out, row0, n, j)
        };
        for p in 0..kb {
            let av0 = vld1q_f32(pp.add(p * MR));
            let av1 = vld1q_f32(pp.add(p * MR + 4));
            let bv0 = vld1q_f32(bp.add(p * NR));
            let bv1 = vld1q_f32(bp.add(p * NR + 4));
            // lane-indexed FMA 把广播并进乘加：每条 `p` 从 24 条指令降到 20 条（省 8 条 dup）。
            acc[0] = vfmaq_laneq_f32::<0>(acc[0], bv0, av0);
            acc[1] = vfmaq_laneq_f32::<0>(acc[1], bv1, av0);
            acc[2] = vfmaq_laneq_f32::<1>(acc[2], bv0, av0);
            acc[3] = vfmaq_laneq_f32::<1>(acc[3], bv1, av0);
            acc[4] = vfmaq_laneq_f32::<2>(acc[4], bv0, av0);
            acc[5] = vfmaq_laneq_f32::<2>(acc[5], bv1, av0);
            acc[6] = vfmaq_laneq_f32::<3>(acc[6], bv0, av0);
            acc[7] = vfmaq_laneq_f32::<3>(acc[7], bv1, av0);
            acc[8] = vfmaq_laneq_f32::<0>(acc[8], bv0, av1);
            acc[9] = vfmaq_laneq_f32::<0>(acc[9], bv1, av1);
            acc[10] = vfmaq_laneq_f32::<1>(acc[10], bv0, av1);
            acc[11] = vfmaq_laneq_f32::<1>(acc[11], bv1, av1);
            acc[12] = vfmaq_laneq_f32::<2>(acc[12], bv0, av1);
            acc[13] = vfmaq_laneq_f32::<2>(acc[13], bv1, av1);
            acc[14] = vfmaq_laneq_f32::<3>(acc[14], bv0, av1);
            acc[15] = vfmaq_laneq_f32::<3>(acc[15], bv1, av1);
        }
        micro_store(out, row0, n, j, acc);
    }
}

/// 读回上一步的 16 个累加向量（非首块续算用）。
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(super) fn micro_load(
    out: &[f32],
    row0: usize,
    n: usize,
    j: usize,
) -> [std::arch::aarch64::float32x4_t; 16] {
    unsafe {
        use std::arch::aarch64::*;
        let mut acc = [vdupq_n_f32(0.0); 16];
        let prev = out.as_ptr();
        let r = row0;
        acc[0] = vld1q_f32(prev.add(r * n + j));
        acc[1] = vld1q_f32(prev.add(r * n + j + 4));
        acc[2] = vld1q_f32(prev.add((r + 1) * n + j));
        acc[3] = vld1q_f32(prev.add((r + 1) * n + j + 4));
        acc[4] = vld1q_f32(prev.add((r + 2) * n + j));
        acc[5] = vld1q_f32(prev.add((r + 2) * n + j + 4));
        acc[6] = vld1q_f32(prev.add((r + 3) * n + j));
        acc[7] = vld1q_f32(prev.add((r + 3) * n + j + 4));
        acc[8] = vld1q_f32(prev.add((r + 4) * n + j));
        acc[9] = vld1q_f32(prev.add((r + 4) * n + j + 4));
        acc[10] = vld1q_f32(prev.add((r + 5) * n + j));
        acc[11] = vld1q_f32(prev.add((r + 5) * n + j + 4));
        acc[12] = vld1q_f32(prev.add((r + 6) * n + j));
        acc[13] = vld1q_f32(prev.add((r + 6) * n + j + 4));
        acc[14] = vld1q_f32(prev.add((r + 7) * n + j));
        acc[15] = vld1q_f32(prev.add((r + 7) * n + j + 4));
        acc
    }
}

/// 把 16 个累加向量写回 `out` 的 8×8 块。
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(super) fn micro_store(
    out: &mut [f32],
    row0: usize,
    n: usize,
    j: usize,
    acc: [std::arch::aarch64::float32x4_t; 16],
) {
    unsafe {
        use std::arch::aarch64::*;
        let op = out.as_mut_ptr();
        let r = row0;
        vst1q_f32(op.add(r * n + j), acc[0]);
        vst1q_f32(op.add(r * n + j + 4), acc[1]);
        vst1q_f32(op.add((r + 1) * n + j), acc[2]);
        vst1q_f32(op.add((r + 1) * n + j + 4), acc[3]);
        vst1q_f32(op.add((r + 2) * n + j), acc[4]);
        vst1q_f32(op.add((r + 2) * n + j + 4), acc[5]);
        vst1q_f32(op.add((r + 3) * n + j), acc[6]);
        vst1q_f32(op.add((r + 3) * n + j + 4), acc[7]);
        vst1q_f32(op.add((r + 4) * n + j), acc[8]);
        vst1q_f32(op.add((r + 4) * n + j + 4), acc[9]);
        vst1q_f32(op.add((r + 5) * n + j), acc[10]);
        vst1q_f32(op.add((r + 5) * n + j + 4), acc[11]);
        vst1q_f32(op.add((r + 6) * n + j), acc[12]);
        vst1q_f32(op.add((r + 6) * n + j + 4), acc[13]);
        vst1q_f32(op.add((r + 7) * n + j), acc[14]);
        vst1q_f32(op.add((r + 7) * n + j + 4), acc[15]);
    }
}

/// 标量微内核：无 SIMD 的架构（以及 x86_64 上未检出 AVX2/FMA 的机器）的兜底。
#[cfg(not(target_arch = "aarch64"))]
pub(super) fn micro_packed_scalar(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
    let Tile {
        row0,
        j,
        n,
        kb,
        first,
    } = tile;
    for r in 0..MR {
        for c in 0..NR {
            let slot = (row0 + r) * n + j + c;
            let mut acc = if first { 0.0f32 } else { out[slot] };
            for p in 0..kb {
                acc += panel[p * MR + r] * b[p * NR + c];
            }
            out[slot] = acc;
        }
    }
}

/// x86_64 的 8×8 微内核：AVX2 + FMA，8 个 `__m256` 累加器（每行一个、8 列一个向量）。
#[cfg(target_arch = "x86_64")]
pub(super) fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        micro_packed_avx2(panel, b, out, tile);
    } else {
        micro_packed_scalar(panel, b, out, tile);
    }
}

/// 8×8 AVX2+FMA 微内核：每行一个 `__m256` 累加器，逐收缩步广播 A 值 × B 向量做 FMA。
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(super) fn micro_packed_avx2(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
    let Tile {
        row0,
        j,
        n,
        kb,
        first,
    } = tile;
    use std::arch::x86_64::*;
    unsafe {
        let zero = _mm256_setzero_ps();
        let pp = panel.as_ptr();
        let bp = b.as_ptr();
        let mut acc = if first {
            [zero; 8]
        } else {
            micro_load(out, row0, n, j)
        };
        for p in 0..kb {
            let bv = _mm256_loadu_ps(bp.add(p * NR));
            // 手写 FMA 微内核：下标形式比迭代器形式更贴近指令序列，便于逐条核对。
            #[allow(clippy::needless_range_loop)]
            for r in 0..MR {
                acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(*pp.add(p * MR + r)), bv, acc[r]);
            }
        }
        micro_store(out, row0, n, j, acc);
    }
}

/// 读回上一步的 8 个累加向量（x86_64 非首块续算用）。
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(super) fn micro_load(
    out: &[f32],
    row0: usize,
    n: usize,
    j: usize,
) -> [std::arch::x86_64::__m256; 8] {
    unsafe {
        use std::arch::x86_64::*;
        let prev = out.as_ptr();
        let r = row0;
        [
            _mm256_loadu_ps(prev.add(r * n + j)),
            _mm256_loadu_ps(prev.add((r + 1) * n + j)),
            _mm256_loadu_ps(prev.add((r + 2) * n + j)),
            _mm256_loadu_ps(prev.add((r + 3) * n + j)),
            _mm256_loadu_ps(prev.add((r + 4) * n + j)),
            _mm256_loadu_ps(prev.add((r + 5) * n + j)),
            _mm256_loadu_ps(prev.add((r + 6) * n + j)),
            _mm256_loadu_ps(prev.add((r + 7) * n + j)),
        ]
    }
}

/// 把 8 个累加向量写回 `out` 的 8×8 块（x86_64）。
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(super) fn micro_store(
    out: &mut [f32],
    row0: usize,
    n: usize,
    j: usize,
    acc: [std::arch::x86_64::__m256; 8],
) {
    unsafe {
        use std::arch::x86_64::*;
        let op = out.as_mut_ptr();
        let r = row0;
        for (i, v) in acc.into_iter().enumerate() {
            _mm256_storeu_ps(op.add((r + i) * n + j), v);
        }
    }
}

/// 无 SIMD 的架构：直接走标量微内核（`micro_packed_scalar` 在这些目标上不会被别处引用）。
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(super) fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
    micro_packed_scalar(panel, b, out, tile);
}

/// 一个待算的标量尾块：左上角 `(i, j)`、覆盖 `mr × nr`，矩阵列宽 `n`、内积长 `k`。
#[derive(Clone, Copy)]
pub(super) struct Span {
    pub(super) i: usize,
    pub(super) j: usize,
    pub(super) mr: usize,
    pub(super) nr: usize,
    pub(super) n: usize,
    pub(super) k: usize,
    /// `a` 的整行宽：内积分块时 `k` 只是当前块，跨块偏移要按整行宽算。
    pub(super) stride: usize,
}

/// 权重布局：`Nt` 是收缩轴在前的 `b[k][n]`，`Tn` 是按输出行常驻的 `w[n][k]`。
#[derive(Clone, Copy)]
pub(super) enum Layout {
    Nt,
    Tn,
}

/// 权重在 `w` 里的寻址三元组：`(基址, 内积步长, 列步长)`——两种布局在这一处收敛。
pub(super) fn weight_stride(span: Span, k0: usize, layout: Layout) -> (usize, usize, usize) {
    match layout {
        Layout::Nt => (k0 * span.n + span.j, span.n, 1),
        Layout::Tn => (span.j * span.stride + k0, 1, span.stride),
    }
}

/// 标量尾块：`out[i..i+mr, j..j+nr]` 逐元素点积（与微内核同一组累加口径，容差内一致）。
pub(super) fn edge(
    a: &[f32],
    w: &[f32],
    out: &mut [f32],
    span: Span,
    k0: usize,
    first: bool,
    layout: Layout,
) {
    let (base, dp, dc) = weight_stride(span, k0, layout);
    for r in 0..span.mr {
        for c in 0..span.nr {
            let slot = (span.i + r) * span.n + span.j + c;
            let mut acc = if first { 0.0f32 } else { out[slot] };
            let wc = base + c * dc;
            for p in 0..span.k {
                acc += a[(span.i + r) * span.stride + k0 + p] * w[wc + p * dp];
            }
            out[slot] = acc;
        }
    }
}
