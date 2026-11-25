//! 前向内核：按行（解码）与按批（prefill）两条路径，点积本身分「带标度」与「稠密」两种。
use diagnostic::{Code, Error};

use super::format::Plain;
use super::shape::{finite_scales, required_weights, scaled_groups};
use super::simd::COLS;

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
    for (idx, value) in out.iter_mut().enumerate() {
        let w_row = &weights[idx * ctx.row..][..ctx.row];
        let s_row = &scales[idx * ctx.groups..][..ctx.groups];
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

/// 行批逐元素入口：按配置选择带标度 / 无标度两条骨架，与 `Format::forward` 同一分叉。
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
    let full = n - n % COLS;
    for j in (0..full).step_by(COLS) {
        let block = &weights[j * m..][..COLS * m];
        C::dot_four(block, x, m, &mut out[j..][..COLS]);
    }
    for j in full..n {
        out[j] = C::dot(&weights[j * m..][..m], x);
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

pub mod packed {
    //! 打包权重的确定性取样与解包点积助手。

    /// 确定性打包权重：`n` 行、每行 `m / cpb` 个字节。
    pub fn weights(m: usize, n: usize, cpb: usize) -> Vec<u8> {
        (0..n * (m / cpb))
            .map(|idx| ((idx * 37) % 251) as u8)
            .collect()
    }

    /// 一行的点积：组内解包 `cpb` 个码查 `dequant` 表，整组共用同一个标度。
    pub fn dot_row(
        packed_row: &[u8],
        x: &[f32],
        s_row: &[f32],
        per_group: usize,
        group: usize,
        cpb: usize,
        dequant: &[f32],
    ) -> f32 {
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
        let ctx =
            crate::gemv::forward::scaled_row_ctx(x.len(), out.len(), scales, weights.len(), cpb)?;
        if let Some(affine) = crate::gemv::simd::admit(cpb, out.len(), ctx, dequant) {
            crate::gemv::simd::forward_batch(&[x], weights, scales, affine, out);
            return Ok(());
        }
        crate::gemv::forward::forward_rows_scaled(
            x,
            weights,
            scales,
            out,
            ctx,
            row_dot(cpb, dequant),
        );
        Ok(())
    }

    /// 打包格式的行批入口：与单行入口共用算术，只换遍历嵌套。
    pub fn forward_batch(
        x: &[f32],
        weights: &[u8],
        scales: &[f32],
        geo: crate::gemv::forward::Geometry,
        out: &mut [f32],
        cpb: usize,
        dequant: &[f32],
    ) -> diagnostic::Result<()> {
        crate::gemv::forward::batch_geometry(x.len(), out.len(), geo)?;
        let ctx = crate::gemv::forward::scaled_row_ctx(geo.m, geo.n, scales, weights.len(), cpb)?;
        if let Some(affine) = crate::gemv::simd::admit(cpb, geo.n, ctx, dequant) {
            let rows: Vec<&[f32]> = x.chunks(geo.m).collect();
            crate::gemv::simd::forward_batch(&rows, weights, scales, affine, out);
            return Ok(());
        }
        crate::gemv::forward::forward_batch_rows_scaled(
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
}

pub(in crate::gemv) mod shape {
    //! 形状上下文、确定性样本与标量参考：`#[annotation::model]` 要的三件产物。
    use diagnostic::{Code, Error};

    use crate::model::{Buf, Cost, Sample, ShapeCtx};

    use crate::gemv::format::{Element, Format};

    /// 与权重格式无关的输入向量：两条算子共用同一份取值。
    fn input(m: usize) -> Vec<f32> {
        (0..m)
            .map(|idx| ((idx * 7) % 13) as f32 * 0.25 - 1.0)
            .collect()
    }

    /// 量化格式共用的标度取值：`0.03125` 的 1..=4 倍，落在 2 的负幂网格，f32 精确。
    pub(crate) fn scale_of(idx: usize) -> f32 {
        0.031_25 * (1 + (idx % 4)) as f32
    }

    /// 把形状对翻译成投影骨架的具名轴上下文：`m` 是规约轴、`n` 是输出轴。
    pub fn project_shape(m: usize, n: usize) -> Result<ShapeCtx, Error> {
        ShapeCtx::new([('m', m), ('n', n)])
    }

    /// 标量代价：完全由 `FormulaOwner::EXPR` 与操作数位宽推出，本函数不含任何流量常数。
    ///
    /// 调用方只给基本轴；派生轴由格式自己声明——分组是算子语义，不是每个层都该抄一遍的常数。
    ///
    /// # Errors
    ///
    /// 形状与表达式不自洽（缺轴、长度为 0，或 `m` 不是 [`Format::GRAIN`] 的整数倍）时返回说明。
    pub fn cost<F: Format>(shape: &ShapeCtx) -> Result<Cost, Error> {
        let extents = match F::DERIVED {
            Some((label, base, divisor)) => shape.extents_with_group(label, base, divisor)?,
            None => shape.extents().clone(),
        };
        F::formula()?
            .cost(&extents)
            .map_err(|error| error.with("nn::gemv::cost"))
    }

    /// 确定性样本：同一形状永远得到同一组输入。
    pub fn sample<F: Format>(m: usize, n: usize) -> Sample {
        let mut bufs = vec![Buf::f32("x", input(m)), F::Elem::buf("w", F::weights(m, n))];
        if let Some(scale) = F::SCALES {
            let groups = m / F::GRAIN.max(1);
            bufs.push(Buf::f32("scales", (0..n * groups).map(scale).collect()));
        }
        bufs.push(Buf::f32("y", vec![0.0; n]));
        Sample {
            m,
            n,
            scalars: vec![("M", m as i64), ("N", n as i64), ("gsz", F::STEP as i64)],
            bufs,
        }
    }

    /// 从样本里取权重：名字固定是 `w`，元素类型由格式给出。
    fn weights<F: Format>(sample: &Sample) -> diagnostic::Result<&[F::Elem]> {
        let buf = sample
            .buf("w")
            .ok_or_else(|| Error::new(Code::Missing, "样本里没有缓冲 `w`"))?;
        F::Elem::borrow(&buf.values).ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("缓冲 `w` 不是 {}：这份格式的权重元素类型", F::Elem::NAME),
            )
        })
    }

    /// 输出维度不能为 0：三条格式与参考实现共同的前置约束。
    pub(crate) fn non_empty(n: usize) -> Result<(), Error> {
        if n == 0 {
            return Err(Error::new(Code::Invalid, "输出维度不能为 0"));
        }
        Ok(())
    }

    /// 带标度格式的公共前置约束：`scales` 能按行均分、分组能整除输入宽度。
    pub(crate) fn scaled_groups(
        m: usize,
        n: usize,
        scales: usize,
    ) -> Result<(usize, usize), Error> {
        non_empty(n)?;
        if !scales.is_multiple_of(n) {
            return Err(Error::new(
                Code::Invalid,
                format!("scale 长度 {scales} 不是 {n} 的整数倍"),
            ));
        }
        let groups = scales / n;
        if groups == 0 || !m.is_multiple_of(groups) {
            return Err(Error::new(
                Code::Invalid,
                format!("scale 长度 {scales} 推不出能整除 {m} 的分组数"),
            ));
        }
        Ok((groups, m / groups))
    }

    /// 标度值守卫：每组分标度必须是有限值；非有限标度会把整组乘积带进 NaN/Inf，且与格式无关。
    pub(crate) fn finite_scales(scales: &[f32], groups: usize) -> Result<(), Error> {
        let bad = scales.iter().any(|scale| !scale.is_finite());
        if bad {
            return Err(Error::new(
                Code::Invalid,
                format!("标度含非有限值（{groups} 组）"),
            ));
        }
        Ok(())
    }

    /// 权重至少要够 `n` 行、每行 `row` 个元素，返回这个下界。
    pub(crate) fn required_weights(weights: usize, n: usize, row: usize) -> Result<usize, Error> {
        let want = n.saturating_mul(row);
        if weights < want {
            return Err(Error::new(
                Code::Invalid,
                format!("权重长度 {weights} < {want}"),
            ));
        }
        Ok(want)
    }

    /// 标量参考：校验由骨架给出，算术由格式给出。
    ///
    /// 骨架负责的部分与格式无关——输出非空、输入够长、权重够长、结果写进一块新缓冲；格式负责
    /// 自己的解包与分组约束。于是"再加一份精度"不需要重写这些检查。
    ///
    /// # Errors
    ///
    /// 缓冲缺失、元素类型不符、长度不足，或格式自己的约束（分组、标度个数）不成立时返回说明。
    pub fn reference<F: Format>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let weights = weights::<F>(sample)?;
        let scales = scales::<F>(sample)?;
        let (m, n) = (sample.m, sample.n);
        gauge(x.len(), m, n)?;
        run::<F>(&x[..m], weights, scales, m, n)
    }

    /// 标度缓冲：只有格式声明了 [`Format::SCALES`] 才读；没有标度的格式拿空切片。
    fn scales<F: Format>(sample: &Sample) -> diagnostic::Result<&[f32]> {
        match F::SCALES {
            Some(_) => sample.f32("scales"),
            None => Ok(&[]),
        }
    }

    /// 骨架侧的尺寸判据：输出非空、输入够长。只问答不答格式——权重够不够由 [`run`] 判。
    fn gauge(x: usize, m: usize, n: usize) -> diagnostic::Result<()> {
        non_empty(n)?;
        if x < m {
            return Err(Error::new(Code::Invalid, format!("输入长度 {x} < {m}")));
        }
        Ok(())
    }

    /// 权重下界够了之后，把结果交给格式自己的算术（分配也在这里，格式只写缓冲）。
    fn run<F: Format>(
        x: &[f32],
        weights: &[F::Elem],
        scales: &[f32],
        m: usize,
        n: usize,
    ) -> diagnostic::Result<Vec<f32>> {
        let want = required_weights(weights.len(), n, F::row(m))?;
        let mut y = vec![0.0f32; n];
        F::forward(
            tensor::View::one(x),
            &weights[..want],
            tensor::View::one(scales),
            &mut y,
        )?;
        Ok(y)
    }
}
