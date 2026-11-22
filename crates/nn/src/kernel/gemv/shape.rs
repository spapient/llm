//! 形状上下文、确定性样本与标量参考：`#[annotation::model]` 要的三件产物。
use diagnostic::{Code, Error};

use crate::model::{Buf, Cost, Sample, ShapeCtx, Values};

use super::format::{Element, Format};

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

/// 标量代价：完全由 [`FormulaOwner::EXPR`] 与操作数位宽推出，本函数不含任何流量常数。
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
    let mut bufs = vec![
        Buf {
            name: "x",
            values: Values::F32(input(m)),
        },
        F::Elem::buf("w", F::weights(m, n)),
    ];
    if let Some(scale) = F::SCALES {
        let groups = m / F::GRAIN.max(1);
        bufs.push(Buf {
            name: "scales",
            values: Values::F32((0..n * groups).map(scale).collect()),
        });
    }
    bufs.push(Buf {
        name: "y",
        values: Values::F32(vec![0.0; n]),
    });
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
pub(crate) fn scaled_groups(m: usize, n: usize, scales: usize) -> Result<(usize, usize), Error> {
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
    F::forward(x, &weights[..want], scales, &mut y)?;
    Ok(y)
}
