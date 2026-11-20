//! 打包权重（每字节 `cpb` 个码）的取样、分组校验与解包点积助手。
//!
//! int2（4 码/字节）与 int4（2 码/字节）只在码宽、查表与粒度倍数上不同；打包位域的字节布局、
//! 分组推导与按行点积收口到这里，避免两条打包格式各抄一遍。int8 不打包，不走本模块。

use super::{finite_scales, required_weights, scaled_groups};
use diagnostic::{Code, Error};

/// 确定性打包权重：`n` 行、每行 `m / cpb` 个字节，同一形状取值可复现。
pub fn weights(m: usize, n: usize, cpb: usize) -> Vec<u8> {
    (0..n * (m / cpb))
        .map(|index| ((index * 37) % 251) as u8)
        .collect()
}

/// 分组推导 + "分组必须是 `cpb` 的倍数" + 打包权重长度下界。
///
/// 返回 `(groups, group)`：每行多少组、每组多少元素。
pub fn groups(
    m: usize,
    n: usize,
    scales: usize,
    weights: usize,
    cpb: usize,
) -> Result<(usize, usize), Error> {
    let (groups, group) = scaled_groups(m, n, scales)?;
    if !group.is_multiple_of(cpb) {
        return Err(Error::new(
            Code::Invalid,
            format!("分组 {group} 必须是 {cpb} 的倍数"),
        ));
    }
    required_weights(weights, n, m / cpb)?;
    Ok((groups, group))
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

/// 打包格式的行遍历：分组与长度守卫通过后，每行解包点积一次写回调用方缓冲。
///
/// 行切片、分组推进与格式无关；格式只给每字节码数 `cpb` 与反量化表 `dequant`。
/// 解包位域与累加顺序和 [`dot_row`] 同一套算术，参考与内核对拍逐位一致。
pub fn forward_rows(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    out: &mut [f32],
    cpb: usize,
    dequant: &[f32],
) -> diagnostic::Result<()> {
    let m = x.len();
    let n = out.len();
    let (groups, group) = groups(m, n, scales.len(), weights.len(), cpb)?;
    finite_scales(scales, groups)?;
    let row_bytes = m / cpb;
    let per_group = group / cpb;
    for (row, value) in out.iter_mut().enumerate() {
        let w_row = &weights[row * row_bytes..][..row_bytes];
        let s_row = &scales[row * groups..][..groups];
        *value = dot_row(w_row, x, s_row, per_group, group, cpb, dequant);
    }
    Ok(())
}

/// 打包格式的行批：一行权重解包一遍，服务 `rows` 行输入，写进 `rows × n` 的行主序输出。
///
/// 与 [`forward_rows`] 同一套算术与同一份 [`dot_row`]，只换遍历嵌套：输出行 `j` 在外（权重行
/// `j` 由此只解包一次），输入行 `r` 在内。于是量化权重在 prefill 下不会被按行数重复读，
/// 而每个元素仍是同一累加顺序——批结果与逐行调用的结果逐位相同。
pub fn forward_batch(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    geo: super::Geometry,
    out: &mut [f32],
    cpb: usize,
    dequant: &[f32],
) -> diagnostic::Result<()> {
    let (groups, group) =
        batch_packed_context(x.len(), out.len(), scales, weights.len(), geo, cpb)?;
    let row_bytes = geo.m / cpb;
    let per_group = group / cpb;
    for j in 0..geo.n {
        let w_row = &weights[j * row_bytes..][..row_bytes];
        let s_row = &scales[j * groups..][..groups];
        for r in 0..geo.rows {
            out[r * geo.n + j] = dot_row(
                w_row,
                &x[r * geo.m..][..geo.m],
                s_row,
                per_group,
                group,
                cpb,
                dequant,
            );
        }
    }
    Ok(())
}

/// 打包行批的守卫收口：几何、分组与标度三项一次判齐，返回 `(groups, group)`。
fn batch_packed_context(
    x: usize,
    out: usize,
    scales: &[f32],
    weights: usize,
    geo: super::Geometry,
    cpb: usize,
) -> diagnostic::Result<(usize, usize)> {
    super::batch_geometry(x, out, geo)?;
    let (groups, group) = groups(geo.m, geo.n, scales.len(), weights, cpb)?;
    finite_scales(scales, groups)?;
    Ok((groups, group))
}

/// 逐元素（不打包）格式的行遍历：守卫与行切片收口到骨架，每行点积算术由闭包给出。
///
/// 与 [`forward_rows`] 对称，差别只是权重不按字节打包、一行即 `m` 个元素；格式只给
/// 逐行点积函数，不在自己的 `forward` 里再抄一遍分组/标度/权重长度守卫。
pub fn forward_plain(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    out: &mut [f32],
    row_dot: impl Fn(&[u8], &[f32], &[f32], usize) -> f32,
) -> diagnostic::Result<()> {
    super::forward_scaled_rows(x, weights, scales, out, row_dot)
}
