//! 前向内核：按行（解码）与按批（prefill）两条路径，点积本身分「带标度」与「稠密」两种。
use diagnostic::{Code, Error};

use super::format::Plain;
use super::shape::{finite_scales, required_weights, scaled_groups};

/// 带标度行骨架：分组、粒度、权重与标度守卫连同行切片统一收口，逐行点积由 `row_dot` 给出。
pub(super) fn forward_rows<E>(
    x: &[f32],
    weights: &[E],
    scales: &[f32],
    out: &mut [f32],
    cpb: usize,
    row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
) -> diagnostic::Result<()> {
    let ctx = scaled_row_ctx(x.len(), out.len(), scales, weights.len(), cpb)?;
    forward_rows_scaled(x, weights, scales, out, ctx, row_dot);
    Ok(())
}

/// 逐行遍历一遍，算术由 `row_dot` 给：守卫已经过了（`ctx` 就是证据），这里只剩切片与调用。
pub(crate) fn forward_rows_scaled<E>(
    x: &[f32],
    weights: &[E],
    scales: &[f32],
    out: &mut [f32],
    ctx: ScaledRow,
    row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
) {
    for (index, value) in out.iter_mut().enumerate() {
        let w_row = &weights[index * ctx.row..][..ctx.row];
        let s_row = &scales[index * ctx.groups..][..ctx.groups];
        *value = row_dot(w_row, x, s_row, ctx.group);
    }
}

/// 行批骨架：输出行在外、输入行在里，一行权重只读一遍。
fn forward_batch_rows<E>(
    x: &[f32],
    weights: &[E],
    scales: &[f32],
    geo: Geometry,
    out: &mut [f32],
    cpb: usize,
    row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
) -> diagnostic::Result<()> {
    let ctx = scaled_row_ctx(geo.m, geo.n, scales, weights.len(), cpb)?;
    forward_batch_rows_scaled(x, weights, scales, geo, out, ctx, row_dot);
    Ok(())
}

/// 稠密行批：无标度、输出行在外使一行权重只读一遍，守卫与 [`forward_dense`] 逐条对齐。
fn forward_batch_dense<C: Plain>(
    x: &[f32],
    weights: &[C::Elem],
    scales: &[f32],
    geo: Geometry,
    out: &mut [f32],
) -> diagnostic::Result<()> {
    dense_scales(scales)?;
    required_weights(weights.len(), geo.n, geo.m)?;
    for row in 0..geo.n {
        let w_row = &weights[row * geo.m..][..geo.m];
        for r in 0..geo.rows {
            out[r * geo.n + row] = C::dot(w_row, &x[r * geo.m..][..geo.m]);
        }
    }
    Ok(())
}

/// 稠密格式的标度守卫：稠密没有标度，收到任何一条都是调用方的形状错误。
fn dense_scales(scales: &[f32]) -> diagnostic::Result<()> {
    if scales.is_empty() {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!("稠密格式没有标度，收到 {} 个", scales.len()),
    ))
}

/// 行批的逐行遍历，算术由 `row_dot` 给：与 [`forward_rows_scaled`] 同一个"守卫已过"约定。
pub(crate) fn forward_batch_rows_scaled<E>(
    x: &[f32],
    weights: &[E],
    scales: &[f32],
    geo: Geometry,
    out: &mut [f32],
    ctx: ScaledRow,
    row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
) {
    for j in 0..geo.n {
        let w_row = &weights[j * ctx.row..][..ctx.row];
        let s_row = &scales[j * ctx.groups..][..ctx.groups];
        for r in 0..geo.rows {
            out[r * geo.n + j] = row_dot(w_row, &x[r * geo.m..][..geo.m], s_row, ctx.group);
        }
    }
}

/// 带标度格式的行守卫：分组、粒度倍数、权重长度与标度有限四项一次判齐，返回 `(groups, group)`。
pub(super) fn scaled_row_ctx(
    m: usize,
    n: usize,
    scales: &[f32],
    weights: usize,
    cpb: usize,
) -> diagnostic::Result<ScaledRow> {
    let (groups, group) = scaled_groups(m, n, scales.len())?;
    if !group.is_multiple_of(cpb) {
        return Err(Error::new(
            Code::Invalid,
            format!("分组 {group} 必须是 {cpb} 的倍数"),
        ));
    }
    required_weights(weights, n, m / cpb)?;
    finite_scales(scales, groups)?;
    Ok(ScaledRow {
        groups,
        group,
        row: m / cpb,
    })
}

/// 已过守卫的带标度行上下文：每行多少组、每组多少元素、每行多少个权重单位。
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScaledRow {
    /// 每行多少组。
    pub groups: usize,
    /// 每组多少元素（必须是 `cpb` 的倍数）。
    pub group: usize,
    /// 每行多少个权重单位（打包格式是 `m / cpb` 个字节）。
    pub row: usize,
}

/// 行批逐元素入口：按配置选择带标度 / 无标度两条骨架，与 [`Format::forward`] 同一分叉。
pub fn forward_plain_batch<C: Plain>(
    x: &[f32],
    weights: &[C::Elem],
    scales: &[f32],
    geo: Geometry,
    out: &mut [f32],
) -> diagnostic::Result<()> {
    batch_geometry(x.len(), out.len(), geo)?;
    if C::SCALES.is_some() {
        forward_batch_rows(x, weights, scales, geo, out, 1, scaled_dot::<C>)
    } else {
        forward_batch_dense::<C>(x, weights, scales, geo, out)
    }
}

/// 行批几何：`rows` 行输入、输入行宽 `m`、输出行宽 `n`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    /// 输入行数。
    pub rows: usize,
    /// 输入行宽（权重行宽）。
    pub m: usize,
    /// 输出行宽。
    pub n: usize,
}

impl Geometry {
    /// 由三个数直接构造；合法性留给入口守卫，构造本身不报错。
    pub const fn new(rows: usize, m: usize, n: usize) -> Self {
        Self { rows, m, n }
    }
}

/// 行批几何：行数非零，输入与输出的长度分别对上 `rows × m` 与 `rows × n`。
pub(super) fn batch_geometry(x: usize, out: usize, geo: Geometry) -> diagnostic::Result<()> {
    if geo.rows == 0 {
        return Err(Error::new(Code::Invalid, "行批至少一行"));
    }
    let (want_x, want_out) = (
        geo.rows.saturating_mul(geo.m),
        geo.rows.saturating_mul(geo.n),
    );
    if x != want_x || out != want_out {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "行批尺寸不符：x={x}，out={out}，期望 {}×{}={want_x}、{}×{}={want_out}",
                geo.rows, geo.m, geo.rows, geo.n
            ),
        ));
    }
    Ok(())
}

/// 无标度逐元素格式的行遍历：换错算子（给了标度）必须报错，而不是静默忽略。
pub(super) fn forward_dense<C: Plain>(
    x: &[f32],
    weights: &[C::Elem],
    scales: &[f32],
    out: &mut [f32],
) -> diagnostic::Result<()> {
    dense_scales(scales)?;
    let (m, n) = (x.len(), out.len());
    required_weights(weights.len(), n, m)?;
    for (row, value) in out.iter_mut().enumerate() {
        let w_row = &weights[row * m..][..m];
        *value = C::dot(w_row, x);
    }
    Ok(())
}

/// 一行的带标度点积：按组推进，每组解包点积一次乘标度（[`Plain::dot`] 承担单组点积）。
pub(super) fn scaled_dot<C: Plain>(
    w_row: &[C::Elem],
    x: &[f32],
    s_row: &[f32],
    group: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for ((w_chunk, scale), x_chunk) in w_row.chunks(group).zip(s_row).zip(x.chunks(group)) {
        acc += C::dot(w_chunk, x_chunk) * scale;
    }
    acc
}
