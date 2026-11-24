//! f32 8×8 寄存器块微内核：aarch64 走 NEON lane-indexed FMA，x86_64 走 AVX2+FMA，其余目标走标量。
//!
//! 这是「能算一块 8×8 的内核」的最小单位，**不含扫描骨架**：面板怎么切、残列怎么收、权重怎么寻址
//! 都是 `nn` 的 GEMM 骨架职责，本文件只回答「给一块打包好的 8×8，把累加器算出来写回」。
//! 由 `nn` 的 `kernel_fast` 经 `Tile`/`micro_packed` 调用，由测量容器经 `port_fma`/
//! `port_fma_block` 计时（`bench` 特性），三者共用同一份瓦片体 `fma_step`。

use super::f32::{MR, NR};

/// 一个待算的 8×8 微内核块：左上角 `(row0, j)`、矩阵列宽 `n`、内积分块长 `kb`、是否首块。
#[derive(Clone, Copy)]
pub struct Tile {
    /// 块左上角行。
    pub row0: usize,
    /// 块左上角列。
    pub j: usize,
    /// 矩阵列宽（跨行偏移用它）。
    pub n: usize,
    /// 本块的内积分长度。
    pub kb: usize,
    /// 是否首块（首块累加器从 0 起手，否则读回 `out` 续算）。
    pub first: bool,
}

/// 打包面板上的 NEON 8×8 微内核（A 面板 `[p][8]`、B 面板 `[p][8]`，一次算 8 行 8 列）。
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
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
/// 生产微内核 `micro_packed` 与取数受限端口探针 [`fma_block_rounds`] 共用**这一份**——
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
/// 那样测的是纯端口而非「取数受限」端口（i8 侧第一版就是这么错的，见 `nn` 的 `port_sdot_block`）。
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

/// 读回上一步的 16 个累加向量（非首块续算用）；每行两个 4-lane 半块。
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn micro_load(
    out: &[f32],
    row0: usize,
    n: usize,
    j: usize,
) -> [std::arch::aarch64::float32x4_t; 16] {
    unsafe {
        use std::arch::aarch64::*;
        let prev = out.as_ptr();
        std::array::from_fn(|i| {
            let row = prev.add((row0 + i / 2) * n + j);
            vld1q_f32(row.add((i % 2) * 4))
        })
    }
}

/// 把 16 个累加向量按行写回 `out` 的 8×8 块。
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn micro_store(
    out: &mut [f32],
    row0: usize,
    n: usize,
    j: usize,
    acc: [std::arch::aarch64::float32x4_t; 16],
) {
    unsafe {
        use std::arch::aarch64::*;
        let op = out.as_mut_ptr();
        for (r, halves) in acc.as_chunks::<2>().0.iter().enumerate() {
            let row = op.add((row0 + r) * n + j);
            vst1q_f32(row, halves[0]);
            vst1q_f32(row.add(4), halves[1]);
        }
    }
}

/// 标量微内核：无 SIMD 的架构（以及 x86_64 上未检出 AVX2/FMA 的机器）的兜底。
#[cfg(not(target_arch = "aarch64"))]
fn micro_packed_scalar(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
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
pub fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        micro_packed_avx2(panel, b, out, tile);
    } else {
        micro_packed_scalar(panel, b, out, tile);
    }
}

/// 8×8 AVX2+FMA 微内核：每行一个 `__m256` 累加器，逐收缩步广播 A 值 × B 向量做 FMA。
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn micro_packed_avx2(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
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
fn micro_load(out: &[f32], row0: usize, n: usize, j: usize) -> [std::arch::x86_64::__m256; 8] {
    unsafe {
        use std::arch::x86_64::*;
        let prev = out.as_ptr();
        std::array::from_fn(|r| _mm256_loadu_ps(prev.add((row0 + r) * n + j)))
    }
}

/// 把 8 个累加向量写回 `out` 的 8×8 块（x86_64）。
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn micro_store(
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
pub fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
    micro_packed_scalar(panel, b, out, tile);
}
