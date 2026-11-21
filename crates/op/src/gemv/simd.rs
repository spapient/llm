//! 解包点积的 16 车道通道：`n·m` 点积、4 列累加器、权重按 [`TN_COLS`] 驻留。
//!
//! 标量叶（[`super::packed::dot_row`]）每读一个码要做三件事：查反量化表、乘 `x`、加进**同一条**
//! 串行累加链。后两件是延迟问题——一行的和必须按顺序进同一个 `f32`，后一项要等前一项舍入，
//! 每四个码才凑得满一次流水线；第一件是核对问题——查表把四个码的并行度换成了四次访存。本模块
//! 把三件事一起换掉：
//!
//! * **查表换成整数码的直接换算。** 打包格式的反量化表恒为 `table[code] = code - zero`
//!   （[`crate::quant::Row`] 的 `I2` 是 `{-2, -1, 0, 1}`、`I4` 是 `{-8..=7}`，[`affine_zero`]
//!   逐条核对，不是假设），于是
//!   `Σ (code − zero)·x' = Σ code·x' − zero·Σ x'`：左半是内层的乘法，右半整行只做一次。
//! * **串行累加换成 16 车道。** 一条 `fmla` 同时算 16 个码，四个码宽 2 的槽位、四个码宽 4 的
//!   槽位都刚好装满 16 字节向量；归约推到最后一次，十六条独立链把乘加延迟完全遮住。
//! * **4 列共享一次 `x` 载入。** 解码的输出宽度是 `n`（生产形状 4096），四行的权重落在同一批
//!   16 字节窗口上，一份 `x` 供四条链用——这正是 `n·m` 里那个 `n` 的复用点。
//! * **`x` 在入口过一遍。** 按分组标度预乘，并把元素重排成"槽位优先"（`x'[slot·row + byte]`），
//!   内层于是只有连续载入，没有步长，也没有按组归约——分组只剩标度这一步。
//!
//! 同值性：`Σ(code·x)·s` 与 `Σ code·(x·s)` 数学上相等，这里取后者。代价是舍入关联与标量叶不同
//! （融合乘加还少一次舍入），所以本通道对外的判据是**"与标量叶在 1e-6 相对误差内一致、且批与
//! 逐行逐位一致"**，不是逐位等于标量叶：`gemv_test_support` 里的"内核与参考逐位相同"仍然成立，
//! 因为 `reference` 走的就是同一个 `Format::forward`。
//!
//! 落地：`std::simd` 是架构无关的 16 车道源码，aarch64 上落成 NEON（`fmla` / `ucvtf` /
//! `ldr q`），x86_64 上落成 SSE2 基线。两处跑的是同一份算术，于是两条验证链验的是同一段代码。
//! 整条通道只用安全代码（`unsafe_code = "deny"` 不动），每 16 个码一条乘加在标量叶上是
//! 16 条查表 + 16 条串行加。

use std::simd::prelude::{SimdFloat, SimdUint};
use std::simd::{f32x16, u8x16};

use super::{ScaledRow, Geometry};
use crate::matmul::TN_COLS;

/// 一次驻留的权重行数（列块宽度），与 prefill 线性层同一个 [`TN_COLS`]。
///
/// 一个块内 32 行权重按 16 字节窗口轮转：块内权重窗口 `32 × 16 B = 512 B`、一行的 `x` 窗口
/// `16 × 2 槽 × 4 B = 128 B` 都在 L1。块宽再大，驻留窗口就开始挤掉 `x`；这是 prefill 线性层
/// 已经量过的同一个预算。
const BLOCK: usize = TN_COLS;

/// 4 列累加器：一次算 4 行，4 条 16 车道链就是 16 个向量寄存器，正好一条 `x` 载入摊到 4 行。
const COLS: usize = 4;

/// 一个窗口的字节数：`16` 字节 = `16 / cpb` 位精度的 16 个码。
const WINDOW: usize = 16;

/// 反量化表的仿射零：`table[code] == code - zero` 时返回 `zero`，否则 `None`（走标量叶）。
///
/// 逐条核对整张表而不是拿首项猜：`zero = -table[0]` 对任何表都能算出来，但只有整张表都落在这条
/// 仿射关系上，"查表"才等价于"码减零点"。包装格式的表是常量表（[`crate::quant::Row`]），
/// 这里核对一遍让通道的选择依据落在表的内容上，而不是"打包格式一定是对称表"这句话上。
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

/// 形状够不够走本通道：码宽只能是 2 位或 4 位，行宽必须是整数个 16 字节窗口。
///
/// 行宽不是 16 的倍数时回退到标量叶——一行尾部的零头会让"窗口内 16 个码"的前提散掉，
/// 而这条通道的全部并行度都建立在这个前提上。
fn fits(cpb: usize, row: usize) -> bool {
    matches!(cpb, 2 | 4) && row.is_multiple_of(WINDOW)
}

/// 单行入口：`out` 是 `n` 个输出，算得成就返回 `true`，算不成（形状/表不符）返回 `false`。
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
    if out.len() < COLS || !fits(cpb, ctx.row) {
        return false;
    }
    let input = layout(x, scales, ctx, cpb);
    let sums = [input.iter().sum::<f32>()];
    run(cpb, weights, &[input], &sums, ctx, zero, out);
    true
}

/// 行批入口：输出行 `r` 写 `out[r·n..]`，与 [`forward`] 同一段算术、同一份列块遍历。
///
/// 遍历与标量叶同一副嵌套（输出行在外、输入行在里）——一行权重只读一遍，而且逐行调用与行批
/// 走的是同一个 `dot`：哪几行进同一组 4 列只由列号决定，与输入行走在哪一层无关，于是批结果
/// 与逐行调用逐位相同。
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
    if geo.n < COLS || !fits(cpb, ctx.row) {
        return false;
    }
    let inputs: Vec<Vec<f32>> = (0..geo.rows)
        .map(|r| layout(&x[r * geo.m..][..geo.m], scales, ctx, cpb))
        .collect();
    let sums: Vec<f32> = inputs.iter().map(|row| row.iter().sum()).collect();
    run(cpb, weights, &inputs, &sums, ctx, zero, out);
    true
}

/// 码表长度：`1 << width`，`width = 8 / cpb`（与 [`crate::quant::width_cpb`] 同一公式）。
fn codes_of(cpb: usize) -> usize {
    1 << crate::quant::width_cpb(cpb)
}

/// 按 `cpb` 单态化：4 位码与 2 位码的槽位数不同，槽位循环在编译期就摊平。
fn run(
    cpb: usize,
    weights: &[u8],
    inputs: &[Vec<f32>],
    sums: &[f32],
    ctx: ScaledRow,
    zero: f32,
    out: &mut [f32],
) {
    match cpb {
        2 => columns::<2>(weights, inputs, sums, ctx, zero, out),
        _ => columns::<4>(weights, inputs, sums, ctx, zero, out),
    }
}

/// 列块遍历：每 [`BLOCK`] 列一块，块内 4 列一组，4 行权重与一行的 `x` 窗口在 L1 里轮转。
///
/// 不足 4 列的尾部回到 `R = 1`（同一个 `dot`，同一副累加器形态），所以 `n` 不必是 4 的倍数。
fn columns<const CPB: usize>(
    weights: &[u8],
    inputs: &[Vec<f32>],
    sums: &[f32],
    ctx: ScaledRow,
    zero: f32,
    out: &mut [f32],
) {
    let (row, n) = (ctx.row, out.len() / inputs.len());
    let mut j = 0;
    while j < n {
        let block = (j + BLOCK).min(n);
        while j < block {
            if block - j >= COLS {
                let rows: [&[u8]; COLS] =
                    core::array::from_fn(|i| &weights[(j + i) * row..][..row]);
                for (r, input) in inputs.iter().enumerate() {
                    for (i, value) in dot::<CPB, COLS>(&rows, input, row).iter().enumerate() {
                        out[r * n + j + i] = value - zero * sums[r];
                    }
                }
                j += COLS;
            } else {
                let rows: [&[u8]; 1] = [&weights[j * row..][..row]];
                for (r, input) in inputs.iter().enumerate() {
                    out[r * n + j] = dot::<CPB, 1>(&rows, input, row)[0] - zero * sums[r];
                }
                j += 1;
            }
        }
    }
}

/// `R` 行的点积：一行一个 16 车道累加器，窗口与槽位由外到内，`x` 在内层只载一次。
///
/// 每读一个 16 字节窗口就把它按槽位拆开（`byte >> (slot · width)`，低位槽在前，与标量叶
/// 同一个码序），每个槽位给 16 个码配 16 个 `x`——`cpb = 2` 时两块 16 码、`cpb = 4` 时四块，
/// 合起来正好是一整条 16 车道的乘加。
fn dot<const CPB: usize, const R: usize>(rows: &[&[u8]; R], input: &[f32], row: usize) -> [f32; R] {
    let mut acc = [f32x16::splat(0.0); R];
    for window in 0..row / WINDOW {
        let at = window * WINDOW;
        for slot in 0..CPB {
            let xs = f32x16::from_slice(&input[slot * row + at..][..WINDOW]);
            for (r, packed) in rows.iter().enumerate() {
                let bytes = u8x16::from_slice(&packed[at..][..WINDOW]);
                acc[r] = codes::<CPB>(bytes, slot).cast::<f32>().mul_add(xs, acc[r]);
            }
        }
    }
    acc.map(|lane| lane.reduce_sum())
}

/// 第 `slot` 个码的 16 个无符号码值：低位槽在前，`mask = (1 << width) - 1`。
fn codes<const CPB: usize>(bytes: u8x16, slot: usize) -> u8x16 {
    let width = 8 / CPB as u8;
    (bytes >> u8x16::splat(width * slot as u8)) & u8x16::splat((1 << width) - 1)
}

/// 入口那一遍：按分组标度预乘，并把元素重排成"槽位优先"——`input[slot·row + byte]`。
///
/// 元素 `e` 落在第 `e / group` 组、槽位 `e % cpb`、行内字节 `e / cpb`；分组是 `cpb` 的倍数
/// （[`scaled_row_context`](super::scaled_row_context) 的守卫），所以一组的 `group / cpb` 个字节
/// 在每个槽位里都是连续的一段。内层读 `x` 连续、写两个（或四个）连续流。
fn layout(x: &[f32], scales: &[f32], ctx: ScaledRow, cpb: usize) -> Vec<f32> {
    let mut input = vec![0.0f32; x.len()];
    let per_group = ctx.group / cpb;
    for (g, &scale) in scales.iter().enumerate() {
        let first = g * per_group;
        for byte in first..first + per_group {
            for slot in 0..cpb {
                input[slot * ctx.row + byte] = x[byte * cpb + slot] * scale;
            }
        }
    }
    input
}
