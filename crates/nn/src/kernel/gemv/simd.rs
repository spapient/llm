//! 真 SIMD 点积，两条通道：
//!
//! * [`forward`]／[`forward_batch`]——打包格式（一字节 `CPB` 个码）。解包、偏移、乘加全在
//!   `LANES` 宽的车道上，[`COLS`] 行权重并行，一份 `x` 块摊到 4 行；`x` 就地转置，不额外分配。
//! * [`elementwise`]——逐元素格式（稠密 `f32`、对称 `int8`）。`LANES` 条车道 × [`CHAINS`] 条
//!   独立累加链，权重按 [`Load`] 取成车道。
//!
//! 两条通道的行尾都只归约一次，于是舍入只发生在"每个元素乘加一次"这一层。

use std::simd::prelude::{SimdFloat, SimdInt, SimdUint};
use std::simd::{Simd, StdFloat};

use super::ScaledRow;

/// 4 列累加器：一次算 4 行，4 条向量链各持一组寄存器，一份 `x` 块摊到 4 行。
pub(crate) const COLS: usize = 4;

/// 宽档车道数：一条 AVX2 `ymm`、四条 NEON 寄存器。
const WIDE: usize = 16;

/// 窄档车道数：一组凑不满 [`WIDE`] 时用，半个 `ymm`、两条 NEON 寄存器。
const NARROW: usize = 8;

/// 逐元素通道的独立累加链数：一条车道链的乘加延迟约 4 拍，4 条刚好把流水填满。
const CHAINS: usize = 4;

/// 反量化表的仿射零：`table[code] == code - zero` 时返回 `zero`，否则 `None`（走标量叶）。
pub(crate) fn affine_zero(dequant: &[f32], codes: usize) -> Option<f32> {
    if dequant.len() != codes {
        return None;
    }
    let zero = -dequant.first()?;
    let affine = dequant
        .iter()
        .enumerate()
        .all(|(code, &value)| value == code as f32 - zero);
    affine.then_some(zero)
}

/// 车道数：一组里每槽位的字节数被 [`WIDE`] 整除取宽档，被 [`NARROW`] 整除取窄档，都不行就不走本通道。
///
/// 车道数整除一组是硬前提：真整除才能让索引算术完全无分支、无掩码、无零头。真实分组
/// （[`GROUP`](super::GROUP) 的倍数配 2 / 4 个码一字节）永远落在宽档或窄档，其余形状走标量叶，
/// 算术一模一样，只是不向量化。
fn lanes_of(per_group: usize) -> Option<usize> {
    if per_group.is_multiple_of(WIDE) {
        Some(WIDE)
    } else if per_group.is_multiple_of(NARROW) {
        Some(NARROW)
    } else {
        None
    }
}

/// 通过校验之后的通道参数：形状（[`ScaledRow`]）、码宽、车道数、反量化表的仿射零点。
#[derive(Clone, Copy, Debug)]
pub(crate) struct Affine {
    ctx: ScaledRow,
    cpb: usize,
    lanes: usize,
    zero: f32,
}

/// 校验并构造 [`Affine`]：`n` 是输出宽度（权重行数），`ctx` 是入口已经量好的行形状。
pub(crate) fn admit(cpb: usize, n: usize, ctx: ScaledRow, dequant: &[f32]) -> Option<Affine> {
    // 码表长度 `1 << width`，`width = 8 / cpb`（与 `quant::width_cpb` 同一公式）。
    let zero = affine_zero(dequant, 1 << crate::quant::width_cpb(cpb))?;
    let lanes = lanes_of(ctx.group / cpb)?;
    (n >= COLS).then_some(Affine {
        ctx,
        cpb,
        lanes,
        zero,
    })
}

/// 唯一入口：一次带上所有输入行，单行只是 `rows` 只有一个元素。
pub(crate) fn forward_batch(
    rows: &[&[f32]],
    weights: &[u8],
    scales: &[f32],
    affine: Affine,
    out: &mut [f32],
) {
    run(
        &Plan {
            weights,
            inputs: rows,
            affine,
            scales,
        },
        out,
    );
}

/// 内核的全部输入：权重、输入行、通道参数、标度。
struct Plan<'a> {
    weights: &'a [u8],
    inputs: &'a [&'a [f32]],
    affine: Affine,
    scales: &'a [f32],
}

/// 两个轴一起单态化：`cpb` 决定槽位循环与位移量，车道数决定向量宽度，编译期全摊平。
fn run(plan: &Plan<'_>, out: &mut [f32]) {
    match (plan.affine.cpb, plan.affine.lanes == WIDE) {
        (2, true) => columns::<2, WIDE>(plan, out),
        (2, false) => columns::<2, NARROW>(plan, out),
        (_, true) => columns::<4, WIDE>(plan, out),
        (_, false) => columns::<4, NARROW>(plan, out),
    }
}

/// 列遍历：整块走 `R = COLS`，不足一块的尾部逐列走 `R = 1`（同一个 [`dot`]、同一副累加器
/// 形态），所以 `n` 不必是 [`COLS`] 的倍数。
fn columns<const CPB: usize, const LANES: usize>(plan: &Plan<'_>, out: &mut [f32]) {
    let n = out.len() / plan.inputs.len();
    let mut j = 0;
    while j + COLS <= n {
        strip::<CPB, LANES, COLS>(plan, out, n, j);
        j += COLS;
    }
    if j < n {
        strip::<CPB, LANES, 1>(plan, out, n, j);
    }
}

/// 一列块：`R` 行权重与同一段 `x` 窗口在 L1 里轮转；权重与标度切片与行无关，只在块首取一次。
fn strip<const CPB: usize, const LANES: usize, const R: usize>(
    plan: &Plan<'_>,
    out: &mut [f32],
    n: usize,
    j: usize,
) {
    let ctx = plan.affine.ctx;
    let (w, s) = (
        &plan.weights[j * ctx.row..][..R * ctx.row],
        &plan.scales[j * ctx.groups..][..R * ctx.groups],
    );
    for (r, input) in plan.inputs.iter().enumerate() {
        let dest = &mut out[r * n + j..][..R];
        dest.copy_from_slice(&dot::<CPB, LANES, R>(w, s, input, ctx, plan.affine.zero));
    }
}

/// `R` 行的点积：一组一条 `LANES` 车道链，`R` 行同时推进，**每行只在行尾归约一次**。
///
/// `ctx.groups` 不是从 `input.len()` 反推出来的，而是入口 [`admit`] 量好的：形状守卫已经保证
/// `input.len() == ctx.groups · ctx.group`，所以这里没有"一组都凑不满"这种需要断言的中间态。
fn dot<const CPB: usize, const LANES: usize, const R: usize>(
    w: &[u8],
    s: &[f32],
    input: &[f32],
    ctx: ScaledRow,
    zero: f32,
) -> [f32; R] {
    let per_group = ctx.group / CPB;
    let shift = Simd::<f32, LANES>::splat(zero);
    let mut running = [Simd::<f32, LANES>::splat(0.0); R];
    for g in 0..ctx.groups {
        let at = g * per_group;
        let acc = group_acc::<CPB, LANES, R>(w, ctx.row, input, at, per_group, shift);
        for (r, (acc, running)) in acc.iter().zip(running.iter_mut()).enumerate() {
            *running = acc.mul_add(Simd::splat(s[r * ctx.groups + g]), *running);
        }
    }
    running.map(SimdFloat::reduce_sum)
}

/// 一组：`CPB` 个槽位 × `per_group / LANES` 段，`R` 条车道链各累加一份。
///
/// `x` 在这里就地转置——每段只取一次连续的 `LANES · CPB` 个元素，摊成 `CPB` 条车道向量，
/// 于是 `x` 每元素只读一遍，入口那一遍整行重排连同它的分配一起消失。
#[inline]
fn group_acc<const CPB: usize, const LANES: usize, const R: usize>(
    w: &[u8],
    row: usize,
    input: &[f32],
    at: usize,
    per_group: usize,
    shift: Simd<f32, LANES>,
) -> [Simd<f32, LANES>; R] {
    let mut acc = [Simd::<f32, LANES>::splat(0.0); R];
    for byte in (at..at + per_group).step_by(LANES) {
        let block = &input[byte * CPB..][..LANES * CPB];
        for slot in 0..CPB {
            let xs = slot_lanes::<CPB, LANES>(block, slot);
            for (r, acc) in acc.iter_mut().enumerate() {
                let bytes = Simd::<u8, LANES>::from_slice(&w[r * row + byte..][..LANES]);
                let value = codes::<CPB, LANES>(bytes, slot).cast::<f32>() - shift;
                *acc = value.mul_add(xs, *acc);
            }
        }
    }
    acc
}

/// 一段输入的某个槽位，摊成 `LANES` 条车道：槽位 `s` 取第 `i · CPB + s` 个元素。
#[inline]
fn slot_lanes<const CPB: usize, const LANES: usize>(
    block: &[f32],
    slot: usize,
) -> Simd<f32, LANES> {
    let mut lane = [0.0f32; LANES];
    for (i, value) in lane.iter_mut().enumerate() {
        *value = block[i * CPB + slot];
    }
    Simd::from_array(lane)
}

/// 第 `slot` 个码的 `LANES` 个无符号码值：低位槽在前，`mask = (1 << width) - 1`。
fn codes<const CPB: usize, const LANES: usize>(
    bytes: Simd<u8, LANES>,
    slot: usize,
) -> Simd<u8, LANES> {
    let width = 8 / CPB as u8;
    (bytes >> Simd::splat(width * slot as u8)) & Simd::splat((1 << width) - 1)
}

/// 逐元素通道的权重取法：稠密 `f32` 直取，对称 `int8` 按二补数重新解释。
///
/// 一条通道只由权重元素类型选，[`elementwise`] 因此对两种格式是同一段代码。
pub(crate) trait Load: Copy {
    /// [`WIDE`] 个元素一次取成 `f32` 车道。
    fn wide(block: &[Self]) -> Simd<f32, WIDE>;

    /// 单个元素：不足一条车道的零头用。
    fn one(elem: Self) -> f32;
}

impl Load for f32 {
    #[inline]
    fn wide(block: &[Self]) -> Simd<f32, WIDE> {
        Simd::from_slice(block)
    }

    #[inline]
    fn one(elem: Self) -> f32 {
        elem
    }
}

impl Load for u8 {
    /// `u8 → i8` 是位模式重解释，`i8 → f32` 是有符号展宽，两条都是单条指令。
    #[inline]
    fn wide(block: &[Self]) -> Simd<f32, WIDE> {
        Simd::<u8, WIDE>::from_slice(block)
            .cast::<i8>()
            .cast::<f32>()
    }

    #[inline]
    fn one(elem: Self) -> f32 {
        (elem as i8) as f32
    }
}

/// 连续点积骨架：`CHAINS` 条独立累加链轮转，权重按 [`Load`] 取成车道，行尾归约一次。
///
/// 这是 `f32` 与对称 `int8` 共用的唯一一份点积——两种格式的差别只剩 [`Load`] 的取法。
///
/// 链按 [`CHAINS`] 分段常量展开，`acc` 的下标因此是编译期常量、累加器整份留在向量寄存器里。
/// 分段之后块 `g·CHAINS + c` 落在链 `c`，与"块下标取模"是同一张映射，故逐位不变——2026-11-22
/// 实测两者同为 `gemv_f32/decode` 39.8–40.2 GiB/s（离散 0.5%），即常量下标只是把意图写明，
/// 并没有换来速度：这条路径只在 `cfg(nightly)` 下编译，那台机器上的实测数字属于 stable 退路
/// （本文件 `gemv::simd` 的 stable 分支），热路径结论必须在那条通道上取。
pub(crate) fn elementwise<T: Load>(w: &[T], x: &[f32]) -> f32 {
    let (w_groups, w_tail) = w.as_chunks::<{ WIDE * CHAINS }>();
    let (x_groups, x_tail) = x.as_chunks::<{ WIDE * CHAINS }>();
    let mut acc = [Simd::<f32, WIDE>::splat(0.0); CHAINS];
    for (wg, xg) in w_groups.iter().zip(x_groups) {
        for c in 0..CHAINS {
            let (wc, xc) = (&wg[c * WIDE..][..WIDE], &xg[c * WIDE..][..WIDE]);
            acc[c] = T::wide(wc).mul_add(Simd::from_slice(xc), acc[c]);
        }
    }
    let mut total = acc.iter().copied().map(SimdFloat::reduce_sum).sum::<f32>();
    for (w, x) in w_tail.iter().zip(x_tail) {
        total += T::one(*w) * *x;
    }
    total
}

/// [`COLS`] 行并行点积：一份 `x` 段摊到 [`COLS`] 行，权重每 64 B 的取数从 8 次降到 5 次。
///
/// 行内按 [`WIDE`] 整除的部分走向量，零头逐行标量收口——零头重读 `x` 的代价可忽略，换来行宽
/// 不必是 [`WIDE`] 的倍数。每行只有一条累加链：链的独立性由"行"提供（[`COLS`] 条独立链），
/// 而不是由链数提供，所以累加器刚好 `COLS` 份，全部留在寄存器里。
pub(crate) fn elementwise_four<T: Load>(w: &[T], x: &[f32], m: usize, out: &mut [f32]) {
    let mut acc = [Simd::<f32, WIDE>::splat(0.0); COLS];
    let full = m - m % WIDE;
    let row = |r: usize| &w[r * m..][..m];
    let chunks: [&[[T; WIDE]]; COLS] = [0, 1, 2, 3].map(|r| row(r).as_chunks::<WIDE>().0);
    let (x_head, x_tail) = x.split_at(full);
    for (k, xv) in x_head.as_chunks::<WIDE>().0.iter().enumerate() {
        let xv = Simd::from_slice(xv);
        for r in 0..COLS {
            acc[r] = T::wide(&chunks[r][k]).mul_add(xv, acc[r]);
        }
    }
    for (r, o) in out.iter_mut().enumerate() {
        let mut total = acc[r].reduce_sum();
        for (wc, xc) in row(r)[full..].iter().zip(x_tail) {
            total += T::one(*wc) * *xc;
        }
        *o = total;
    }
}

#[cfg(test)]
mod tests {
    include!("../../../tests/kernel_simd_tests.rs");
}
