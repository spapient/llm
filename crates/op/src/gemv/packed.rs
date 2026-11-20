//! 打包权重（每字节 `cpb` 个码）的取样与解包点积助手。
//!
//! int2（4 码/字节）与 int4（2 码/字节）只在码宽、查表与粒度倍数上不同；打包位域的字节布局
//! 与按行点积收口到这里，避免两条打包格式各抄一遍。行遍历、守卫与行切片在骨架
//! `gemv::forward_rows` 里只有一份——本模块只把“行宽按 `cpb` 折算”接上去。
//! int8 不打包，不走本模块。

/// 确定性打包权重：`n` 行、每行 `m / cpb` 个字节，同一形状取值可复现。
pub fn weights(m: usize, n: usize, cpb: usize) -> Vec<u8> {
    (0..n * (m / cpb))
        .map(|index| ((index * 37) % 251) as u8)
        .collect()
}

/// 一行的点积：按分组推进，组内解包 `cpb` 个码查 `dequant` 表，整组共用同一个标度。
///
/// 累加顺序与朴素写法逐位一致（同一字节的低位码在前）；每组 `group` 为 `cpb` 倍数，
/// 解包只读整组码，不触碰补齐位。
pub fn dot_row(
    packed_row: &[u8],
    x: &[f32],
    s_row: &[f32],
    per_group: usize,
    group: usize,
    cpb: usize,
    dequant: &[f32],
) -> f32 {
    // 位宽走配置表派生：`width = 8 / cpb`，与 [`crate::quant::width_cpb`] 同一公式，不在这里第二份除法。
    let width = crate::quant::width_cpb(cpb);
    let mask = (1u8 << width) - 1;
    let mut acc = 0.0f32;
    for ((packed, scale), xs) in packed_row
        .chunks_exact(per_group)
        .zip(s_row)
        .zip(x.chunks_exact(group))
    {
        let mut dot = 0.0f32;
        for (byte, codes) in packed.iter().zip(xs.chunks(cpb)) {
            for (slot, &code_slot) in codes.iter().enumerate() {
                let code = usize::from((byte >> (slot * width)) & mask);
                dot += dequant[code] * code_slot;
            }
        }
        acc += dot * scale;
    }
    acc
}

/// 把骨架的逐行点积接口接到 [`dot_row`]：骨架只给分组元素数 `group`，解包点积还要每组字节数
/// `group / cpb`、每字节码数与反量化表。
fn row_dot(cpb: usize, dequant: &[f32]) -> impl Fn(&[u8], &[f32], &[f32], usize) -> f32 + '_ {
    move |w_row, x, s_row, group| dot_row(w_row, x, s_row, group / cpb, group, cpb, dequant)
}

/// 打包格式的单行入口：行宽按 `cpb` 折算，逐行点积走 `row_dot`。
///
/// 守卫、行切片与遍历都在骨架 `gemv::forward_rows` 里；这里只接上“一行 `m / cpb` 个字节”
/// 与解包算术，解包位域与累加顺序和 [`dot_row`] 同一套，参考与内核对拍逐位一致。
pub fn forward_rows(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    out: &mut [f32],
    cpb: usize,
    dequant: &[f32],
) -> diagnostic::Result<()> {
    super::forward_rows(x, weights, scales, out, cpb, row_dot(cpb, dequant))
}

/// 打包格式的行批入口：与 [`forward_rows`] 同一套算术与同一份 `row_dot`，只换遍历嵌套。
///
/// 输出行 `j` 在外（权重行 `j` 由此只解包一次），输入行 `r` 在内——量化权重在 prefill 下不会
/// 被按行数重复读，而每个元素仍是同一累加顺序，批结果与逐行调用逐位相同。
pub fn forward_batch(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    geo: super::Geometry,
    out: &mut [f32],
    cpb: usize,
    dequant: &[f32],
) -> diagnostic::Result<()> {
    super::forward_batch_rows(x, weights, scales, geo, out, cpb, row_dot(cpb, dequant))
}
