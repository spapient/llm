//! 解包点积的向量通道：`n·m` 点积、4 列累加器、4 行权重与一份 `x` 窗口在 L1 里轮转。

use std::simd::prelude::{SimdFloat, SimdUint};
use std::simd::{Simd, StdFloat};

use super::ScaledRow;

/// 4 列累加器：一次算 4 行，4 条向量链各持一组寄存器，一份 `x` 载入摊到 4 行。
const COLS: usize = 4;

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

/// 形状够不够走本通道：车道数落在 8 / 16 两档，且一组的字节数被它整除。
fn fits(cpb: usize, lanes: usize) -> bool {
    matches!((cpb, lanes), (2 | 4, 8 | 16))
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
    let zero = affine_zero(dequant, codes_of(cpb))?;
    let lanes = ctx.group / cpb;
    (n >= COLS && fits(cpb, lanes)).then_some(Affine {
        ctx,
        cpb,
        lanes,
        zero,
    })
}

/// 单行入口：`out` 是 `n` 个输出。
pub(crate) fn forward(x: &[f32], weights: &[u8], scales: &[f32], affine: Affine, out: &mut [f32]) {
    let input = [layout(x, affine)];
    run(
        &Plan {
            weights,
            inputs: &input,
            affine,
            scales,
        },
        out,
    );
}

/// 行批入口：与 [`forward`] 同一段算术、同一份列块遍历，只是一次带上所有输入行。
pub(crate) fn forward_batch(
    rows: &[&[f32]],
    weights: &[u8],
    scales: &[f32],
    affine: Affine,
    out: &mut [f32],
) {
    let input: Vec<Vec<f32>> = rows.iter().map(|row| layout(row, affine)).collect();
    run(
        &Plan {
            weights,
            inputs: &input,
            affine,
            scales,
        },
        out,
    );
}

/// 码表长度：`1 << width`，`width = 8 / cpb`（与 [`crate::quant::width_cpb`] 同一公式）。
fn codes_of(cpb: usize) -> usize {
    1 << crate::quant::width_cpb(cpb)
}

/// 内核的全部输入：权重、排好版的输入行、通道参数、标度。
struct Plan<'a> {
    weights: &'a [u8],
    inputs: &'a [Vec<f32>],
    affine: Affine,
    scales: &'a [f32],
}

/// 按 `cpb` 与车道数单态化：槽位循环与组尾归约的形态都由这两个常量决定，编译期摊平。
fn run(plan: &Plan<'_>, out: &mut [f32]) {
    match (plan.affine.cpb, plan.affine.lanes) {
        (2, 8) => columns::<2, 8>(plan, out),
        (2, _) => columns::<2, 16>(plan, out),
        (_, 8) => columns::<4, 8>(plan, out),
        _ => columns::<4, 16>(plan, out),
    }
}

/// 列遍历：整块走 `R = COLS`，不足一块的尾部逐列走 `R = 1`（同一个 `dot`、同一副累加器
/// 形态），所以 `n` 不必是 `COLS` 的倍数。
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
    let (row, groups, group) = (
        plan.affine.ctx.row,
        plan.affine.ctx.groups,
        plan.affine.ctx.group,
    );
    let (w, s) = (
        &plan.weights[j * row..][..R * row],
        &plan.scales[j * groups..][..R * groups],
    );
    for (r, input) in plan.inputs.iter().enumerate() {
        let dest = &mut out[r * n + j..][..R];
        dest.copy_from_slice(&dot::<CPB, LANES, R>(
            w,
            row,
            s,
            input,
            group,
            plan.affine.zero,
        ));
    }
}

/// `R` 行的点积：一组一条 `LANES` 车道链，四行同时推进，**每行只在行尾归约一次**。
fn dot<const CPB: usize, const LANES: usize, const R: usize>(
    w: &[u8],
    row: usize,
    s: &[f32],
    input: &[f32],
    group: usize,
    zero: f32,
) -> [f32; R] {
    let per_group = group / CPB;
    let groups = input.len() / group;
    assert!(groups > 0, "一组都凑不满的形状在 admit 就该被挡下");
    let shift = Simd::<f32, LANES>::splat(zero);
    let mut running = [Simd::<f32, LANES>::splat(0.0); R];
    for g in 0..groups {
        let at = g * per_group;
        let acc = group_acc::<CPB, LANES, R>(w, row, input, at, per_group, shift);
        for (r, (acc, running)) in acc.iter().zip(running.iter_mut()).enumerate() {
            *running = acc.mul_add(Simd::splat(s[r * groups + g]), *running);
        }
    }
    running.map(SimdFloat::reduce_sum)
}

/// 一组：`CPB` 个槽位 × `per_group / LANES` 段，`R` 条车道链各累加一份。
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
    for slot in 0..CPB {
        for chunk in (at..at + per_group).step_by(LANES) {
            let xs = Simd::<f32, LANES>::from_slice(&input[slot * row + chunk..][..LANES]);
            for (r, acc) in acc.iter_mut().enumerate() {
                let bytes = Simd::<u8, LANES>::from_slice(&w[r * row + chunk..][..LANES]);
                let value = codes::<CPB, LANES>(bytes, slot).cast::<f32>() - shift;
                *acc = value.mul_add(xs, *acc);
            }
        }
    }
    acc
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
fn layout(x: &[f32], affine: Affine) -> Vec<f32> {
    let (row, cpb) = (affine.ctx.row, affine.cpb);
    let mut input = vec![0.0f32; x.len()];
    for byte in 0..row {
        for slot in 0..cpb {
            input[slot * row + byte] = x[byte * cpb + slot];
        }
    }
    input
}

#[cfg(test)]
#[path = "../../tests/simd_tests.rs"]
mod tests;
