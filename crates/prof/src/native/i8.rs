//! i8 `sdot` 4×4 寄存器块微内核（aarch64 dotprod）与端口峰值探针。
//!
//! 瓦片几何是 `MR=4 × NR=4`（配置常量的单一事实源在 `super::i8`）：B 转置打包成「每列
//! k 连续」，每个 k16 一次 4 行 A 加载 + 4 次 B 加载 + 16 条 `vdotq_s32`（256 MAC / 24 指令），
//! 一块 B 瓦片被 4 行复用。dotprod 运行时探测，缺失时 `dot_block` 返回 `false`，由 `nn` 骨架
//! 走标量兜底；`unsafe` 仅限本岛的 aarch64 微内核与探针。
//!
//! 生产微内核（`dot_block`）与取数受限探针（`port_sdot_block`）共用同一份瓦片体
//! `sdot_tile`：探针的全部意义是「与生产逐指令一致」，两份写法必然漂移（N5/可替换）。
//!
//! # 不变量
//!
//! 本文件是 `prof::native` 的 i8 sdot 隔离岛：模块级放行 `unsafe_code`（与 storage mmap、
//! device executor 同一形态），所有裸指针与 NEON `dotprod` intrinsic 只出现在这里。安全入口
//! `dot_block` / `port_sdot` / `port_sdot_block` 先做运行时探测与形状守卫，再进 `unsafe`
//! 工人；工人只读调用方给定的打包面板，写回长度由调用方的行段/列段布局保证。

#![allow(unsafe_code)]

use super::i8_geo::{MR, NR};

/// 一个 4 列块的坐标：起点行 `r`、列块起点 `blk`，矩阵列宽 `n`、内积长 `k` 与向量化长 `kvec`。
#[derive(Clone, Copy)]
pub struct Gram {
    /// 块起点行。
    pub r: usize,
    /// 块起点列。
    pub blk: usize,
    /// 矩阵列宽（跨行偏移用它）。
    pub n: usize,
    /// 内积长度。
    pub k: usize,
    /// 向量化段长度（16 的倍数）；标量尾巴由 `nn` 骨架补。
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub kvec: usize,
}

/// aarch64 dotprod：一个 4 列块的 4 行，瓦片累加后水平求和、标量补 `kvec..k` 余数写回。
///
/// 接住（dotprod 可用且 `kvec ≥ 16`）写回 `out` 返回 `true`；非 aarch64 / 无 dotprod / 长度不够
/// 返回 `false`，调用方据此走标量两行兜底。探测与 `unsafe` 全部收在本岛，`nn` 不写 arch/feature 门。
#[cfg(target_arch = "aarch64")]
pub fn dot_block(a: &[i8], bpack: &[i8], out: &mut [i32], g: Gram) -> bool {
    if g.kvec < 16 || !std::arch::is_aarch64_feature_detected!("dotprod") {
        return false;
    }
    // 安全性：dotprod 已运行时探测；A/B 行段与列段长度由调用方的打包布局保证。
    unsafe {
        dot_one(a, bpack, out, g);
    }
    true
}

/// 非 aarch64：没有 dotprod 通道，一律交回标量兜底。
#[cfg(not(target_arch = "aarch64"))]
pub fn dot_block(a: &[i8], bpack: &[i8], out: &mut [i32], g: Gram) -> bool {
    let _ = (a, bpack, out, g);
    false
}

/// aarch64 dotprod：一个 4 列块的 4 行（`unsafe` 岛）。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_one(a: &[i8], bpack: &[i8], out: &mut [i32], g: Gram) {
    unsafe {
        let ap = a.as_ptr().add(g.r * g.k);
        let bp = bpack.as_ptr();
        let accs = sdot_tile(ap, bp.add(g.blk * g.k), g.k, g.kvec);
        harvest(ap, bp, out, (g.r, g.n, g.k, g.kvec), g.blk, accs);
    }
}

/// 一个 4 列块的收口：水平求和 + 标量补 `kvec..k` 的余数，写回 4 行。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn harvest(
    ap: *const i8,
    bp: *const i8,
    out: &mut [i32],
    (r, n, k, kvec): (usize, usize, usize, usize),
    blk: usize,
    accs: [[std::arch::aarch64::int32x4_t; MR]; NR],
) {
    use std::arch::aarch64::*;
    unsafe {
        for (nb, column) in accs.iter().enumerate() {
            let jj = blk + nb;
            if jj < n {
                for (mr, cell) in column.iter().enumerate() {
                    let mut s = vaddvq_s32(*cell);
                    let mut q = kvec;
                    while q < k {
                        s += i32::from(*ap.add(mr * k + q)) * i32::from(*bp.add(jj * k + q));
                        q += 1;
                    }
                    out[(r + mr) * n + jj] = s;
                }
            }
        }
    }
}

/// 一个 4 行 × 4 列瓦片的整数累加：`bp` 指向**本瓦片首列**在 B 面板里的起点，只累加 `p < kvec`。
///
/// 生产微内核（[`dot_one`]）与取数受限探针（[`sdot_block_rounds`]）共用**这一份**：
/// 探针的全部意义是「与生产逐指令一致」，两份写法必然漂移，一份才有保证（N5/可替换）。
/// `kvec` 是向量化段长度，标量尾巴由 `nn` 骨架补，不在这里。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn sdot_tile(
    ap: *const i8,
    bp: *const i8,
    k: usize,
    kvec: usize,
) -> [[std::arch::aarch64::int32x4_t; MR]; NR] {
    use std::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_s32(0);
        // 列优先：一列一个内层迭代，块内推进不必同时索引四行。
        let mut accs = [[zero; MR]; NR];
        let mut p = 0;
        while p < kvec {
            let av: [int8x16_t; MR] = [
                vld1q_s8(ap.add(p)),
                vld1q_s8(ap.add(k + p)),
                vld1q_s8(ap.add(2 * k + p)),
                vld1q_s8(ap.add(3 * k + p)),
            ];
            for (nb, column) in accs.iter_mut().enumerate() {
                let bv = vld1q_s8(bp.add(nb * k + p));
                for (cell, a) in column.iter_mut().zip(av) {
                    *cell = vdotq_s32(*cell, a, bv);
                }
            }
            p += 16;
        }
        accs
    }
}

/// `sdot` 端口峰值探针一轮的乘加次数：12 条互不依赖的 `sdot` × 16 项。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub const SDOT_MAC_PER_ROUND: usize = 12 * 16;

/// i8 `sdot` 端口峰值：`rounds` 轮 × 12 条 `sdot`，操作数与累加器全常驻，**内层零 load 零 store**。
///
/// 这是 K8 的分母：i8 路径走的是 NEON `sdot`（每指令 4 组 × 4 项 = 16 次乘加），不是 SME
/// 的 `fmopa`，所以 f32 FMA 峰值与 SME 峰值**都不能**当分母——那会得到 >100% 的无意义百分比（F2/F18）。
///
/// **12 条链是实测平台起点**（链数扫描，单线程 GMAC/s：4 → 91、8 → 182、12 → 272、
/// 16 → 274、20 → 273、24 → 237、32 → 232；两次重复误差 <1%）。平台是 12–20 条，
/// ≥24 条因累加器溢出寄存器回落到 235 一带。滑下去的 8 条链版本低于平台 1.5 倍，
/// 曾把 GEMM 误判成超峰值——分母偏低会让百分比虚高，所以这里取平台**左侧**的 12 条而不是最高点。
/// 延迟/发射模型解释不了这个数值（274 GMAC/s ÷ 16 MAC = 17 G 条/s），故只报实测值，不报推导值。
///
/// 本机无 `dotprod` 或 `rounds == 0` 时返回 `false` 且不执行任何向量指令。只在 `bench` 特性下编译。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub fn port_sdot(rounds: usize) -> bool {
    if rounds == 0 || !std::arch::is_aarch64_feature_detected!("dotprod") {
        return false;
    }
    // 安全性：dotprod 已运行时探测；函数体内只有寄存器运算，不触碰内存。
    unsafe {
        sdot_rounds(rounds);
    }
    true
}

/// 纯 `sdot` 循环：12 个累加器互不依赖，累加结果进 `black_box` 防整段消除。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[target_feature(enable = "dotprod")]
#[inline(never)]
unsafe fn sdot_rounds(rounds: usize) {
    use std::arch::aarch64::*;
    let a = vdupq_n_s8(3);
    let b = vdupq_n_s8(5);
    let mut acc = [vdupq_n_s32(0); 12];
    for _ in 0..rounds {
        for chain in &mut acc {
            *chain = vdotq_s32(*chain, a, b);
        }
    }
    std::hint::black_box(acc);
}

/// `sdot` **取数受限**峰值一轮的 k16 步数换算系数：每步 16 条 `sdot`，故一轮供 `16 * k` 次乘加。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub const SDOT_BLOCK_MAC_PER_K: usize = 16;

/// `sdot` **取数受限**峰值：一轮 = `k/16` 个 k16 步，每步 4 条 A load + 4 条 B load + 16 条
/// `sdot`，与生产微内核 `sdot_tile` 逐指令一致（同一份代码，不会漂移）。
///
/// **为什么要它**：纯端口探针零 load，是理想上限；生产核每 16 条 `sdot` 要发 8 条 load。
/// 两者之差就是取数代价，也是「K8 的缺口里多少是结构注定、多少是核写得差」的分界线。
///
/// **第一版是错探针，必须记下来**：当时把 A/B 写成固定 64 字节、地址用常量偏移
/// `bp.add(nb * 16)`，而循环变量不进地址，于是 LLVM 把 8 条 load **整体提到循环外**——
/// `otool -tV` 实测循环体只剩 16 条 `sdot` 加 `subs`/`b.ne`，`ldp q16,q17,[x0]` 只出现一次。
/// 那样的「混合峰值 ≈ 纯峰值」是空证据：它根本没在环内取数。改成与生产相同的 `base + p`
/// 单调推进后，取数被迫留在环内，实测比值才是真的（同一台机、同一编译参数，中位 0.875）。
///
/// 但仓内 `criterion` 口径与那支独立程序不同，且**仓内才是权威**：同一会话
/// `i8_peak/1_block/port` 539.37 vs `i8_peak/1/port` 543.19 = **0.993**——单线程取数几乎免费；
/// 18 线程 5667.3 vs 6736.0 = **0.841**，即瓦片体自身能到端口峰值的 84%。
/// 故「4×4 tile 注定只能到 87.5%」被证伪（单线程天花板 99%、18 线程 84%）。
///
/// **它现在与生产共用 `sdot_tile`，因此不再是独立仪器**：N5 要求探针与生产同源、不许漂移，
/// 代价是它的绝对值不能再横向比"换了一版取数写法"的旧数（旧独立版单线程 493.3、18 线程 4032.8）。
/// 它仍然回答一个没有别的口径能回答的问题：**同一个瓦片体放进零记账的合成循环里能跑多快**，
/// 于是"生产离瓦片体上限差多少"= 内存系统 + 瓦片外记账，而不是核写得差。
///
/// A/B 与 `k` 的形状校验：`k` 非零且是 16 的倍数，两侧各留足 `4k`。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
fn block_shaped(a: &[i8], b: &[i8], k: usize) -> bool {
    k != 0 && k.is_multiple_of(16) && a.len() >= 4 * k && b.len() >= 4 * k
}

/// 本机无 `dotprod`、`rounds == 0`、形状不合时返回 `false` 且不执行任何向量指令。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub fn port_sdot_block(rounds: usize, a: &[i8], b: &[i8], k: usize) -> bool {
    if rounds == 0 || !block_shaped(a, b, k) || !std::arch::is_aarch64_feature_detected!("dotprod")
    {
        return false;
    }
    // 安全性：dotprod 已运行时探测；A/B 各 4k 字节已在上面校验。
    unsafe {
        sdot_block_rounds(rounds, a.as_ptr(), b.as_ptr(), k);
    }
    true
}

/// 取数受限循环：4 行 × 4 列累加器，每 k16 步 4 条 A load + 4 条 B load + 16 条 `sdot`。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[target_feature(enable = "dotprod")]
#[inline(never)]
unsafe fn sdot_block_rounds(rounds: usize, ap: *const i8, bp: *const i8, k: usize) {
    unsafe {
        for _ in 0..rounds {
            std::hint::black_box(sdot_tile(ap, bp, k, k));
        }
    }
}
