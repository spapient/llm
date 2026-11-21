//! 解包点积的向量通道：`n·m` 点积、4 列累加器、权重按 [`BLOCK`]（即 [`TN_COLS`]）驻留。
//!
//! 标量叶（[`super::packed::dot_row`]）每读一个码要做三件事：查反量化表、乘 `x`、加进**同一条**
//! 串行累加链。后两件是延迟问题——一行的和必须按顺序进同一个 `f32`，后一项要等前一项舍入，
//! 四个码才凑得满一次流水线；第一件是核对问题——查表把四个码的并行度换成了四次访存。本模块
//! 把三件事一起换掉：
//!
//! * **查表换成整数码的直接换算。** 打包格式的反量化表恒为 `table[code] = code - zero`
//!   （[`crate::quant::Row`] 的 `I2` 是 `{-2, -1, 0, 1}`、`I4` 是 `{-8..=7}`；[`affine_zero`]
//!   逐条核对整张表，不是假设），于是每个码只剩"移位、掩码、换算成 `f32`、减零点"四步纯算术，
//!   四次访存变成零次。零点按车道减（一条 `fsub` 摊在 `LANES` 个码上）；也可以按组修正
//!   （`Σ (code − zero)·x = Σ code·x − zero·Σ x`，右半每组一次），但那样每组多一条标量链、
//!   还要预先备好每组的 `Σ x`，实测不更快，所以码值就地减零点。
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
//!
//! 实测（`crates/benchmarks/src/kernel.rs` 的 `gemv_i4` 组，生产形状 `1×4096×4096`、权重一遍
//! 9.8 MB）：标量叶 2.73 ms / 3.58 GiB/s，本通道 1.87 ms / 5.21 GiB/s，**1.46×**。剩下的时间
//! 几乎全在换算上——每 16 个码要 4 条 `ucvtf` 加 4 条拓宽，约 0.6 条指令/码，乘加只占
//! 0.06 条；按这个账内核已跑到该形态约 85% 的吞吐上限。列数加密不再有收益：`COLS = 8` 因
//! 16 条累加链溢出到栈帧，实测慢 47%；`COLS = 2` 落在噪声内。真要再上一个量级得靠 `sdot`
//! 的整数点积（8 位码 × 8 位 `x`，一条指令 16 个乘加），那需要 `unsafe` 内建函数，与
//! `unsafe_code = "deny"` 冲突，故本通道止于可移植的这一档。

use std::simd::prelude::{SimdFloat, SimdUint};
use std::simd::{Simd, StdFloat};

use super::ScaledRow;
use crate::matmul::TN_COLS;

/// 一次驻留的权重行数（列块宽度），与 prefill 线性层同一个 [`TN_COLS`]。
///
/// 一个块内 [`TN_COLS`] 行权重按 16 字节窗口轮转：块内权重窗口 `32 × 16 B = 512 B`、一行的 `x`
/// 窗口 `16 码 × 4 B = 64 B` 都在 L1。块宽是 prefill 线性层已经量过的同一个预算——再大就开始
/// 挤掉 `x`。
const BLOCK: usize = TN_COLS;

/// 4 列累加器：一次算 4 行，4 条向量链各持一组寄存器，一份 `x` 载入摊到 4 行。
const COLS: usize = 4;

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

/// 形状够不够走本通道：车道数落在 8 / 16 两档，且一组的字节数被它整除。
///
/// 车道数是 `group / cpb`（一组里每个槽位多少码）；一组的每槽字节数是 `per_group = group / cpb`，
/// 与车道数同一个数——所以"一组恰好是整数个向量"恒成立，组尾才只需要一次归约。恒等之外的形状
/// （一组凑不满一条向量，或凑出来不是 8 / 16 车道）回退标量叶。
fn fits(cpb: usize, lanes: usize) -> bool {
    matches!((cpb, lanes), (2 | 4, 8 | 16))
}

/// 通过校验之后的通道参数：形状（[`ScaledRow`]）、码宽、车道数、反量化表的仿射零点。
///
/// 把"校验过了"做成一个**值**，而不是三个平行的 `usize` 加一个 `bool` 返回值：拿到了 `Affine`
/// 就说明表是仿射的、形状凑得满整条向量、输出宽度够分 [`COLS`] 列，内核于是不必再判一次，
/// 也不可能拿到一组互相矛盾的参数。
#[derive(Clone, Copy, Debug)]
pub(crate) struct Affine {
    ctx: ScaledRow,
    cpb: usize,
    lanes: usize,
    zero: f32,
}

/// 校验并构造 [`Affine`]：`n` 是输出宽度（权重行数），`ctx` 是入口已经量好的行形状。
///
/// 三件事都在这里一次定下来——反量化表逐条核对是否 `table[code] == code - zero`
/// （[`affine_zero`]）、一组的字节数是否落在车道数的整档上（[`fits`]）、输出宽度是否够一整组
/// （[`COLS`]）。任一条不成立就返回 `None`，调用方回退标量叶：本通道是加速，不是唯一实现。
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
///
/// 遍历嵌套与标量叶一致：列块在外、输入行在内——同一批权重窗口供每个输入行用一遍，权重只读
/// 一遍；而逐列组的 `dot` 与输入行走在哪一层无关，于是批结果与逐行调用逐位相同。
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

/// 列块遍历：每 [`BLOCK`] 列一块，块内 4 列一组、每组 4 行权重与一行的 `x` 窗口在 L1 里轮转。
///
/// 不足 4 列的尾部回到 `R = 1`（同一个 `dot`、同一副累加器形态），所以 `n` 不必是 4 的倍数。
fn columns<const CPB: usize, const LANES: usize>(plan: &Plan<'_>, out: &mut [f32]) {
    let (row, groups, group) = (
        plan.affine.ctx.row,
        plan.affine.ctx.groups,
        plan.affine.ctx.group,
    );
    let n = out.len() / plan.inputs.len();
    let mut j = 0;
    while j < n {
        let block = (j + BLOCK).min(n);
        while j < block {
            let width = if block - j >= COLS { COLS } else { 1 };
            for (r, input) in plan.inputs.iter().enumerate() {
                let dest = &mut out[r * n + j..][..width];
                let (w, s) = (
                    &plan.weights[j * row..][..width * row],
                    &plan.scales[j * groups..][..width * groups],
                );
                if width == COLS {
                    dest.copy_from_slice(&dot::<CPB, LANES, COLS>(
                        w,
                        row,
                        s,
                        input,
                        group,
                        plan.affine.zero,
                    ));
                } else {
                    dest.copy_from_slice(&dot::<CPB, LANES, 1>(
                        w,
                        row,
                        s,
                        input,
                        group,
                        plan.affine.zero,
                    ));
                }
            }
            j += width;
        }
    }
}

/// `R` 行的点积：一组一条 `LANES` 车道链，四行同时推进，**每行只在行尾归约一次**。
///
/// 归约是水平求和——串且长（`LANES` 车道要 `log2` 步一路加到底），一组归约一次就意味着每组
/// 一次 ~20 周期的等待，`groups` 个组串起来能把内核按在 L1 里空等。这里用线性关系把它摊开：
/// `Σ_g reduce(acc_g)·s_g = reduce(Σ_g acc_g·s_g)`——每组的 `LANES` 车道先乘上该组标度加进一条
/// **跨组**的累加链（一条 `fma`，`LANES` 车道互不相干），归约推迟到行尾做一次。于是组循环里
/// 只剩一条无等待的向量链，而不是 `groups` 条。
///
/// `zero` 折进码值（`Σ (code − zero)·x`），于是不需要每组再拿 `Σ x` 去修正——`x` 的和在两条
/// 通道里都不再出现。
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
    // 组号在 R 行之间共用（每行都有这一组的标度、权重行也都按同一组号切），不是"遍历某一条
    // 切片"——按下标走组号，行号由内层 `r` 给出。
    #[allow(clippy::needless_range_loop)]
    for g in 0..groups {
        let at = g * per_group;
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
        for (r, (acc, running)) in acc.iter().zip(running.iter_mut()).enumerate() {
            *running = acc.mul_add(Simd::splat(s[r * groups + g]), *running);
        }
    }
    running.map(SimdFloat::reduce_sum)
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
