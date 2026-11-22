//! 打包权重（每字节 `cpb` 个码）的取样与解包点积助手。
/// 确定性打包权重：`n` 行、每行 `m / cpb` 个字节，同一形状取值可复现。
pub fn weights(m: usize, n: usize, cpb: usize) -> Vec<u8> {
    (0..n * (m / cpb))
        .map(|index| ((index * 37) % 251) as u8)
        .collect()
}

/// 一行的点积：按分组推进，组内解包 `cpb` 个码查 `dequant` 表，整组共用同一个标度。
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
pub fn forward_rows(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    out: &mut [f32],
    cpb: usize,
    dequant: &[f32],
) -> diagnostic::Result<()> {
    let ctx = super::forward::scaled_row_ctx(x.len(), out.len(), scales, weights.len(), cpb)?;
    if let Some(affine) = super::simd::admit(cpb, out.len(), ctx, dequant) {
        super::simd::forward_batch(&[x], weights, scales, affine, out);
        return Ok(());
    }
    super::forward::forward_rows_scaled(x, weights, scales, out, ctx, row_dot(cpb, dequant));
    Ok(())
}

/// 打包格式的行批入口：与 [`forward_rows`] 同一套算术与同一份 `row_dot`，只换遍历嵌套。
pub fn forward_batch(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    geo: super::forward::Geometry,
    out: &mut [f32],
    cpb: usize,
    dequant: &[f32],
) -> diagnostic::Result<()> {
    super::forward::batch_geometry(x.len(), out.len(), geo)?;
    let ctx = super::forward::scaled_row_ctx(geo.m, geo.n, scales, weights.len(), cpb)?;
    if let Some(affine) = super::simd::admit(cpb, geo.n, ctx, dequant) {
        let rows: Vec<&[f32]> = x.chunks(geo.m).collect();
        super::simd::forward_batch(&rows, weights, scales, affine, out);
        return Ok(());
    }
    super::forward::forward_batch_rows_scaled(
        x,
        weights,
        scales,
        geo,
        out,
        ctx,
        row_dot(cpb, dequant),
    );
    Ok(())
}
