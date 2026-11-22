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
        fma_step(&mut acc, pp, bp, kb);
        micro_store(out, row0, n, j, acc);
    }
}

/// 一个收缩步的 8×8 瓦片体：4 条 `vld1q` 取数 + 16 条 lane-indexed FMA。
///
/// 生产微内核 [`micro_packed`] 与取数受限端口探针 [`fma_block_rounds`] 共用**这一份**——
/// 探针的全部意义是「与生产逐指令一致」，两份写法必然漂移（N5）。`pp` 是 A 面板每步
/// `MR` 个 f32、`bp` 是 B 面板每步 `NR` 个 f32，两者步距都是 8 即 32 字节。
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[inline]
unsafe fn fma_step(
    acc: &mut [std::arch::aarch64::float32x4_t; 16],
    pp: *const f32,
    bp: *const f32,
    kb: usize,
) {
    use std::arch::aarch64::*;
    unsafe {
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
    }
}

/// 一个收缩步的乘加次数：16 条 FMA × 每向量 4 个 f32 通道。以 `k` 为单位命名，
/// 因为取数受限探针按 `k` 步展开，纯端口探针正好一步。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub const FMA_MAC_PER_K: usize = 16 * 4;

/// f32 FMA 端口峰值：`rounds` 步 × 16 条 lane-indexed FMA，**环内零 load 零 store**。
///
/// 这是 K7 分母里「理想上限」那一半，指令形状与生产瓦片体逐条一致（同 16 个累加器、
/// 同 lane-indexed FMA），只差那 4 条 `vld1q`。`machine_peak` 虽也是 f32 峰值，但它是
/// 自动向量化的标量循环（4 条链 × 16 lane），链数与寄存器分配都不同于手写瓦片，拿它当
/// 「瓦片体的端口上限」是口径混用；本探针才是同一个形状。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub fn port_fma(rounds: usize) -> bool {
    if rounds == 0 {
        return false;
    }
    // 安全性：aarch64 基线含 NEON，函数体内只有寄存器运算，不触碰内存。
    #[allow(unsafe_code)]
    unsafe {
        fma_pure_rounds(rounds);
    }
    true
}

/// 纯 FMA 循环：16 个累加器互不依赖，结果进 `black_box` 防整段消除。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[allow(unsafe_code)]
#[inline(never)]
unsafe fn fma_pure_rounds(rounds: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 16];
        let av = vdupq_n_f32(1.000_000_1);
        for _ in 0..rounds {
            // 16 个累加器互不依赖：每条只读自己，不构成串行链。
            for chain in &mut acc {
                *chain = vfmaq_f32(*chain, av, av);
            }
        }
        std::hint::black_box(acc);
    }
}

/// `fma_step` 的形状校验：`k` 非零，A/B 各留足 `NR * k` 个 f32。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
fn block_shaped(a: &[f32], b: &[f32], k: usize) -> bool {
    k != 0 && a.len() >= NR * k && b.len() >= NR * k
}

/// f32 **取数受限**端口：与生产瓦片体逐指令一致，每步 4 条 `vld1q` + 16 条 FMA。
///
/// **为什么要它**：纯端口探针环内零 load，是理想上限；生产核每步要发 4 条 load。
/// 两者之差就是取数代价，也是「K7 的缺口里多少是结构注定、多少是核外记账」的分界线。
/// K8 的同一对照给出 i8 瓦片体单线程 99.3%、18 线程 84.1%，故那 84% 就是瓦片体的真实天花板。
///
/// `k` 取 1024 使地址在环内单调推进——地址不随循环变化时 LLVM 会把 load 整体提出循环，
/// 那样测的是纯端口而非「取数受限」端口（i8 侧第一版就是这么错的，见 [`super::i8::port_sdot_block`]）。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub fn port_fma_block(rounds: usize, a: &[f32], b: &[f32], k: usize) -> bool {
    if rounds == 0 || !block_shaped(a, b, k) {
        return false;
    }
    // 安全性：A/B 各 `NR * k` 个 f32 已在上面校验。
    #[allow(unsafe_code)]
    unsafe {
        fma_block_rounds(rounds, a.as_ptr(), b.as_ptr(), k);
    }
    true
}

/// 取数受限循环：每轮跑一次完整瓦片体，**零记账**（不读回 `out`、不写回 `out`）。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[allow(unsafe_code)]
#[inline(never)]
unsafe fn fma_block_rounds(rounds: usize, ap: *const f32, bp: *const f32, k: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 16];
        for _ in 0..rounds {
            fma_step(&mut acc, ap, bp, k);
        }
        std::hint::black_box(acc);
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
