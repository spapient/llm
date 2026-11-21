//! 解包点积的向量通道：`n·m` 点积、4 列累加器、权重按 [`BLOCK`]（即 [`TN_COLS`]）驻留。
//!
//! 标量叶（[`super::packed::dot_row`]）每读一个码要做三件事：查反量化表、乘 `x`、加进**同一条**
//! 串行累加链。后两件是延迟问题——一行的和必须按顺序进同一个 `f32`，后一项要等前一项舍入，
//! 四个码才凑得满一次流水线；第一件是核对问题——查表把四个码的并行度换成了四次访存。本模块
//! 把三件事一起换掉：
//!
//! * **查表换成整数码的直接换算。** 打包格式的反量化表恒为 `table[code] = code - zero`
//!   （[`crate::quant::Row`] 的 `I2` 是 `{-2, -1, 0, 1}`、`I4` 是 `{-8..=7}`；[`affine_zero`]
//!   逐条核对整张表，不是假设），于是
//!   `Σ (code − zero)·x = Σ code·x − zero·Σ x`：左半进内层的乘法，右半每组只做一次。
//! * **串行累加换成一条向量链。** 一组的码宽刚好装满 `LANES = group / cpb` 个车道——`group = 32`
//!   时 4 位码是 16 车道、2 位码是 8 车道；一条 `fmla` 同时算 `LANES` 个码，归约只在组尾做一次。
//! * **4 列共享一次 `x` 载入。** 解码的输出宽度是 `n`（生产形状 4096），四行的权重落在同一批
//!   16 字节窗口上，一份 `x` 供四条链用——这正是 `n·m` 里那个 `n` 的复用点。
//! * **`x` 在入口过一遍。** 元素重排成"槽位优先"（`x[slot·row + byte]`），内层于是只有连续载入，
//!   没有步长，也没有除槽位之外的索引算术。
//!
//! 标度留在原位：每组一条标度、每组一乘。这不是保守——把标度提前乘进 `x` 只有"整行共用一个
//! 标度"时才成立，而量化权重的标度是逐行逐组的，提前乘就意味着每行重排一遍 `m` 个元素，
//! `O(n·m)` 的入口开销会和内核本身一样大。所以入口只做与行无关的重排，标度乘法留在组尾。
//!
//! 同值性：`Σ(code·x)·s` 与 `Σ code·(x·s)` 数学上相等，这里取前者。舍入关联与标量叶不同
//! （融合乘加还少一次舍入），所以本通道的判据是**"与标量叶在舍入量级内一致、且批与逐行逐位
//! 一致"**；`packed_test_support` 里"内核与参考逐位相同"仍然成立，因为 `reference` 走的就是
//! 同一个 `Format::forward`。取值可精确表示时（整数码 × 整数 `x` × 2 的幂标度）两条通道逐位
//! 相同，`exactly_representable_values_are_bit_identical` 钉住的就是这一条。
//!
//! 落地：`std::simd` 是架构无关的源码，aarch64 上落成 NEON（`fmla` / `ucvtf` / `ldr q`），
//! x86_64 上落成 SSE2 基线。两处编译的是同一份算术，于是两条验证链验的是同一段代码；整条通道
//! 不用 `unsafe`（`unsafe_code = "deny"` 不动）。

use std::simd::prelude::{SimdFloat, SimdUint};
use std::simd::{Simd, StdFloat};

use super::{Geometry, ScaledRow};
use crate::matmul::TN_COLS;

/// 一次驻留的权重行数（列块宽度），与 prefill 线性层同一个 [`TN_COLS`]。
///
/// 一个块内 [`TN_COLS`] 行权重按 16 字节窗口轮转：块内权重窗口 `32 × 16 B = 512 B`、一行的 `x`
/// 窗口 `16 码 × 4 B = 64 B` 都在 L1。块宽是 prefill 线性层已经量过的同一个预算——再大就开始
/// 挤掉 `x`。
const BLOCK: usize = TN_COLS;

/// 4 列累加器：一次算 4 行，4 条向量链各持一组寄存器，一份 `x` 载入摊到 4 行。
const COLS: usize = 4;

/// 一个窗口的字节数：16 字节 = 每个槽位 16 个码，是 4 位码（16 车道）与 2 位码（8 车道）的公倍数。
const WINDOW: usize = 16;

/// 反量化表的仿射零：`table[code] == code - zero` 时返回 `zero`，否则 `None`（走标量叶）。
///
/// 逐条核对整张表而不是拿首项猜：`zero = -table[0]` 对任何表都算得出，但只有整张表都落在这条
/// 仿射关系上，"查表"才等价于"码减零点"。表是格式常量（[`crate::quant::Row`]），核对一遍让
/// 通道的选择依据落在表的内容上，而不是"打包格式一定是对称表"这句话上。
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

/// 形状够不够走本通道：一组的码宽必须落成 8 或 16 个车道，行宽必须是整数个 16 字节窗口。
///
/// 车道数是 `group / cpb`（一组里每个槽位多少码）。它同时是窗口的因数——8 与 16 都整除 16——
/// 于是"一组恰好是整数个向量"成立，组尾才只需要一次归约。行宽不是 16 的倍数、或车道数不在这两档，
/// 就回退标量叶：本通道的全部并行度都建立在这两个前提上。
fn fits(cpb: usize, row: usize, lanes: usize) -> bool {
    matches!((cpb, lanes), (2 | 4, 8 | 16)) && row.is_multiple_of(WINDOW)
}

/// 单行入口：`out` 是 `n` 个输出；算得成返回 `true`，算不成（形状/表不符）返回 `false`。
pub(crate) fn forward(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    out: &mut [f32],
    cpb: usize,
    ctx: ScaledRow,
    dequant: &[f32],
) -> bool {
    let Some(zero) = affine_zero(dequant, codes_of(cpb)) else {
        return false;
    };
    if out.len() < COLS || !fits(cpb, ctx.row, ctx.group / cpb) {
        return false;
    }
    let input = [layout(x, ctx, cpb)];
    let sums = [group_sums(x, ctx)];
    run(cpb, ctx.group / cpb, weights, &input, &sums, ctx, zero, scales, out);
    true
}

/// 行批入口：与 [`forward`] 同一段算术、同一份列块遍历，只是一次带上所有输入行。
///
/// 遍历嵌套与标量叶一致：列块在外、输入行在内——同一批权重窗口供每个输入行用一遍，权重只读
/// 一遍；而逐列组的 `dot` 与输入行走在哪一层无关，于是批结果与逐行调用逐位相同。
pub(crate) fn forward_batch(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    geo: Geometry,
    out: &mut [f32],
    cpb: usize,
    ctx: ScaledRow,
    dequant: &[f32],
) -> bool {
    let Some(zero) = affine_zero(dequant, codes_of(cpb)) else {
        return false;
    };
    if geo.n < COLS || !fits(cpb, ctx.row, ctx.group / cpb) {
        return false;
    }
    let input: Vec<Vec<f32>> =
        (0..geo.rows).map(|r| layout(&x[r * geo.m..][..geo.m], ctx, cpb)).collect();
    let sums: Vec<Vec<f32>> =
        (0..geo.rows).map(|r| group_sums(&x[r * geo.m..][..geo.m], ctx)).collect();
    run(cpb, ctx.group / cpb, weights, &input, &sums, ctx, zero, scales, out);
    true
}

/// 码表长度：`1 << width`，`width = 8 / cpb`（与 [`crate::quant::width_cpb`] 同一公式）。
fn codes_of(cpb: usize) -> usize {
    1 << crate::quant::width_cpb(cpb)
}

/// 按 `cpb` 与车道数单态化：槽位循环与组尾归约的形态都由这两个常量决定，编译期摊平。
fn run(
    cpb: usize,
    lanes: usize,
    weights: &[u8],
    inputs: &[Vec<f32>],
    sums: &[Vec<f32>],
    ctx: ScaledRow,
    zero: f32,
    scales: &[f32],
    out: &mut [f32],
) {
    match (cpb, lanes) {
        (2, 8) => columns::<2, 8>(weights, inputs, sums, ctx, zero, scales, out),
        (2, _) => columns::<2, 16>(weights, inputs, sums, ctx, zero, scales, out),
        (_, 8) => columns::<4, 8>(weights, inputs, sums, ctx, zero, scales, out),
        _ => columns::<4, 16>(weights, inputs, sums, ctx, zero, scales, out),
    }
}

/// 列块遍历：每 [`BLOCK`] 列一块，块内 4 列一组、每组 4 行权重与一行的 `x` 窗口在 L1 里轮转。
///
/// 不足 4 列的尾部回到 `R = 1`（同一个 `dot`、同一副累加器形态），所以 `n` 不必是 4 的倍数。
fn columns<const CPB: usize, const LANES: usize>(
    weights: &[u8],
    inputs: &[Vec<f32>],
    sums: &[Vec<f32>],
    ctx: ScaledRow,
    zero: f32,
    scales: &[f32],
    out: &mut [f32],
) {
    let (row, groups) = (ctx.row, ctx.groups);
    let n = out.len() / inputs.len();
    let mut j = 0;
    while j < n {
        let block = (j + BLOCK).min(n);
        while j < block {
            let width = if block - j >= COLS { COLS } else { 1 };
            for (r, input) in inputs.iter().enumerate() {
                let dest = &mut out[r * n + j..][..width];
                if width == COLS {
                    let w: [&[u8]; COLS] =
                        core::array::from_fn(|i| &weights[(j + i) * row..][..row]);
                    let s: [&[f32]; COLS] =
                        core::array::from_fn(|i| &scales[(j + i) * groups..][..groups]);
                    let total = dot::<CPB, LANES, COLS>(&w, &s, input, &sums[r], ctx.group, zero);
                    dest.copy_from_slice(&total);
                } else {
                    let row_w: [&[u8]; 1] = [&weights[j * row..][..row]];
                    let row_s: [&[f32]; 1] = [&scales[j * groups..][..groups]];
                    let total =
                        dot::<CPB, LANES, 1>(&row_w, &row_s, input, &sums[r], ctx.group, zero);
                    dest.copy_from_slice(&total);
                }
            }
            j += width;
        }
    }
}

/// `R` 行的点积：一组一条 `LANES` 车道链，四行同时推进，组尾归约一次并乘上该组的标度。
///
/// 每读一个 16 字节窗口就按槽位拆开（`byte >> (slot · width)`，低位槽在前——与标量叶同一个
/// 码序），每个槽位给 `LANES` 个码配 `LANES` 个 `x`：`cpb = 2` 时两块 16 车道、`cpb = 4` 时四块
/// 8 车道，合起来覆盖一组。
fn dot<const CPB: usize, const LANES: usize, const R: usize>(
    w_rows: &[&[u8]; R],
    s_rows: &[&[f32]; R],
    input: &[f32],
    sums: &[f32],
    group: usize,
    zero: f32,
) -> [f32; R] {
    let (row, per_group) = (input.len() / CPB, group / CPB);
    let mut total = [0.0f32; R];
    for (g, &sum) in sums.iter().enumerate() {
        let at = g * per_group;
        let mut acc = [Simd::<f32, LANES>::splat(0.0); R];
        for slot in 0..CPB {
            for window in (at..at + per_group).step_by(WINDOW) {
                for chunk in (window..window + WINDOW).step_by(LANES) {
                    let xs = Simd::<f32, LANES>::from_slice(&input[slot * row + chunk..][..LANES]);
                    for (r, packed) in w_rows.iter().enumerate() {
                        let bytes = Simd::<u8, LANES>::from_slice(&packed[chunk..][..LANES]);
                        acc[r] =
                            codes::<CPB, LANES>(bytes, slot).cast::<f32>().mul_add(xs, acc[r]);
                    }
                }
            }
        }
        for r in 0..R {
            total[r] += (acc[r].reduce_sum() - zero * sum) * s_rows[r][g];
        }
    }
    total
}

/// 第 `slot` 个码的 `LANES` 个无符号码值：低位槽在前，`mask = (1 << width) - 1`。
fn codes<const CPB: usize, const LANES: usize>(
    bytes: Simd<u8, LANES>,
    slot: usize,
) -> Simd<u8, LANES> {
    let width = 8 / CPB as u8;
    (bytes >> Simd::splat(width * slot as u8)) & Simd::splat((1 << width) - 1)
}

/// 入口那一遍：把元素重排成"槽位优先"——`input[slot·row + byte]`。
///
/// 元素 `e` 落在槽位 `e % cpb`、行内字节 `e / cpb`。重排与行无关（所以只做一遍），标度不进
/// 这一步（每组标度逐行不同，见模块文档）。内层读 `x` 连续、写 `cpb` 个连续流。
fn layout(x: &[f32], ctx: ScaledRow, cpb: usize) -> Vec<f32> {
    let mut input = vec![0.0f32; x.len()];
    for byte in 0..ctx.row {
        for slot in 0..cpb {
            input[slot * ctx.row + byte] = x[byte * cpb + slot];
        }
    }
    input
}

/// 每组的 `x` 之和：`zero` 修正只需要它，与输出行无关，所以一行只算一遍。
fn group_sums(x: &[f32], ctx: ScaledRow) -> Vec<f32> {
    x.chunks_exact(ctx.group).map(|group| group.iter().sum()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{I2, I4};

    const M: usize = 256;
    const N: usize = 13;
    const GROUP: usize = 32;

    /// 反量化表：与 [`crate::quant::Row`] 里两档打包格式同一张。
    fn table(cpb: usize) -> &'static [f32] {
        match cpb {
            2 => I4.table,
            _ => I2.table,
        }
    }

    /// `n` 行、每行 `m / cpb` 字节的确定取值权重。
    fn weights(m: usize, n: usize, cpb: usize) -> Vec<u8> {
        (0..n * (m / cpb)).map(|i| ((i * 37) % 251) as u8).collect()
    }

    /// `n` 行 × `groups` 组的标度，逐行逐组不同；2 的幂，便于逐位比较。
    fn scales(n: usize, groups: usize) -> Vec<f32> {
        (0..n * groups).map(|i| [0.25f32, 0.5, 1.0, 2.0][(i * 5) % 4]).collect()
    }

    fn context(m: usize, cpb: usize) -> ScaledRow {
        ScaledRow { groups: m / GROUP, group: GROUP, row: m / cpb }
    }

    /// 一条输入行：整数，`|x| <= 8`。
    fn whole(m: usize) -> Vec<f32> {
        (0..m).map(|i| ((i * 7 + 3) % 17) as f32 - 8.0).collect()
    }

    /// 一条输入行：带小数的通用取值。
    fn fractional(m: usize) -> Vec<f32> {
        (0..m).map(|i| ((i * 37 % 29) as f32 - 14.0) * 0.037_037).collect()
    }

    /// 标量叶（[`super::packed::dot_row`]）对一行给出的值——独立于本通道的第二实现。
    fn oracle(
        m: usize,
        cpb: usize,
        w_row: &[u8],
        s_row: &[f32],
        x: &[f32],
    ) -> f32 {
        let ctx = context(m, cpb);
        super::super::packed::dot_row(
            w_row,
            x,
            s_row,
            ctx.group / cpb,
            ctx.group,
            cpb,
            table(cpb),
        )
    }

    #[test]
    fn the_affine_zero_is_the_table_zero() {
        assert_eq!(affine_zero(I4.table, 16), Some(8.0));
        assert_eq!(affine_zero(I2.table, 4), Some(2.0));
        // 一张不落在仿射关系上的表必须被挡在通道外。
        let bent = [0.0, 2.0, 4.0, 9.0];
        assert_eq!(affine_zero(&bent, 4), None);
        assert_eq!(affine_zero(&bent, 3), None);
    }

    #[test]
    fn shapes_without_a_whole_window_fall_back() {
        assert!(!fits(2, 8, 16), "行宽不足一个窗口");
        assert!(!fits(4, 24, 8), "行宽不是 16 的倍数");
        assert!(!fits(4, 64, 2), "一组凑不满一个车道向量");
        assert!(fits(2, M / 2, 16));
        assert!(fits(4, M / 4, 8));
        // 走不到通道的形状必须由 `forward` 自己说“算不成”，而不是给出一个数。
        let (w, s, x) = (weights(8, 2, 2), scales(2, 8 / GROUP.max(1)), vec![0.0f32; 8]);
        let mut out = [0.0f32; 2];
        assert!(!forward(&x, &w, &s, &mut out, 2, context(8, 2), table(2)));
    }

    #[test]
    fn exactly_representable_values_are_bit_identical() {
        // 整数码 × 整数 x × 2 的幂标度：每一步都不产生舍入，于是归约顺序无关，两条通道逐位相同。
        for cpb in [2, 4] {
            let ctx = context(M, cpb);
            let (w, s, x) = (weights(M, N, cpb), scales(N, ctx.groups), whole(M));
            let mut out = vec![0.0f32; N];
            assert!(forward(&x, &w, &s, &mut out, cpb, ctx, table(cpb)));
            for (j, &got) in out.iter().enumerate() {
                let want = oracle(M, cpb, &w[j * ctx.row..][..ctx.row], &s[j * ctx.groups..][..ctx.groups], &x);
                assert_eq!(got, want, "cpb = {cpb} 第 {j} 行必须与标量叶逐位相同");
            }
        }
    }

    #[test]
    fn inexact_values_agree_within_rounding() {
        // 非精确取值：只有舍入关联不同，差值必须落在 f32 舍入的量级里。
        for cpb in [2, 4] {
            let ctx = context(M, cpb);
            let (w, s, x) = (weights(M, N, cpb), scales(N, ctx.groups), fractional(M));
            let mut out = vec![0.0f32; N];
            assert!(forward(&x, &w, &s, &mut out, cpb, ctx, table(cpb)));
            for (j, &got) in out.iter().enumerate() {
                let want = oracle(M, cpb, &w[j * ctx.row..][..ctx.row], &s[j * ctx.groups..][..ctx.groups], &x);
                let slack = 1e-5 * want.abs().max(1.0);
                assert!((got - want).abs() <= slack, "cpb = {cpb} 第 {j} 行：{got} vs {want}");
            }
        }
    }

    #[test]
    fn a_row_batch_equals_row_by_row_calls() {
        let cpb = 2;
        let ctx = context(M, cpb);
        let rows = 3;
        let (w, s) = (weights(M, N, cpb), scales(N, ctx.groups));
        let base = fractional(M);
        let mut x = Vec::with_capacity(rows * M);
        for r in 0..rows {
            x.extend(base.iter().map(|v| v + r as f32));
        }
        let geo = Geometry { rows, m: M, n: N };
        let mut batched = vec![0.0f32; rows * N];
        assert!(forward_batch(&x, &w, &s, geo, &mut batched, cpb, ctx, table(cpb)));
        for r in 0..rows {
            let mut single = vec![0.0f32; N];
            assert!(forward(&x[r * M..][..M], &w, &s, &mut single, cpb, ctx, table(cpb)));
            assert_eq!(batched[r * N..][..N], *single, "第 {r} 行：批与逐行调用必须逐位相同");
        }
    }
}
