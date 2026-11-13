//! 归约骨架：`y[0] = fold(a[0..n])`，输出单个元素。
//!
//! 输入轴 `n` 被规约消掉，输出轴 `o` 长度恒为 1。`reduce_sum_f32` / `reduce_max_f32` 是同一副
//! 骨架的两种 [`Reduce`] 实现，骨架（表达式缓存、代价、样本、校验、遍历）只有一份，按类型单态化。

use crate::einsum::{Bits, Operand};
use crate::gemv::OperandSpec;
use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};
use std::sync::OnceLock;

/// 归约算子随语义变化的部分。
pub trait Reduce {
    /// 表达式各项的位宽与方向（表达式固定为 `n,o->o`）。
    const OPERANDS: &'static [OperandSpec];

    /// 表达式缓存。
    fn formula() -> Result<&'static crate::einsum::Formula, Error>;

    /// 累加初值。
    fn start() -> f32;

    /// 一次折叠：把当前累加值与下一个元素合并。
    fn fold(acc: f32, x: f32) -> f32;

    /// 确定性输入。
    fn input(len: usize) -> Vec<f32>;
}

/// 归约骨架的表达式缓存。
pub struct ReduceCache(OnceLock<Result<crate::einsum::Formula, Error>>);

impl ReduceCache {
    /// 空缓存。
    #[must_use]
    pub const fn new() -> Self {
        Self(OnceLock::new())
    }

    /// 取表达式：`n,o->o`，输入读、输出写，输出轴长度 1。
    ///
    /// # Errors
    ///
    /// 表达式与操作数不自洽时返回说明。
    pub fn get<R: Reduce>(&'static self) -> Result<&'static crate::einsum::Formula, Error> {
        self.0
            .get_or_init(|| {
                let operands = R::OPERANDS
                    .iter()
                    .map(|spec| Operand::new(spec.name, spec.bits, spec.access))
                    .collect();
                crate::einsum::Expr::parse("n,o->o")
                    .and_then(|expr| crate::einsum::Formula::new(expr, operands))
                    .map_err(|error| error.with("op::reduce::formula"))
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

impl Default for ReduceCache {
    fn default() -> Self {
        Self::new()
    }
}

/// 两条轴：输入轴 `n`（被规约），输出轴 `o`。
pub const AXES: &[AxisSpec] = &[
    AxisSpec {
        role: AxisRole::Input,
        label: 'n',
    },
    AxisSpec {
        role: AxisRole::Output,
        label: 'o',
    },
];

/// 形状：输入轴 `n`=len，输出轴 `o` 恒为 1。
///
/// # Errors
///
/// 当前实现不会失败。
pub fn project_shape(len: usize, _: usize) -> Result<ShapeContext, Error> {
    ShapeContext::new([('n', len), ('o', 1)])
}

/// 标量代价。
pub fn cost<R: Reduce>(shape: &ShapeContext) -> Result<Cost, Error> {
    R::formula()?
        .cost(shape.extents())
        .map_err(|error| error.with("op::reduce::cost"))
}

/// 确定性样本：输入 len 个，输出 1 个。
pub fn sample<R: Reduce>(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: 1,
        scalars: vec![("N", len as i64)],
        buffers: vec![
            Buffer {
                name: "a",
                values: Values::F32(R::input(len)),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0]),
            },
        ],
    }
}

/// 标量参考：折叠整个输入。
pub fn reference<R: Reduce>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let a = sample.f32("a")?;
    let mut acc = R::start();
    for x in a {
        acc = R::fold(acc, *x);
    }
    Ok(vec![acc])
}

/// 内核本体：结果写进调用方给的单元素缓冲。
///
/// # Errors
///
/// 输出缓冲不是单元素时返回说明。
pub fn forward<R: Reduce>(a: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    if out.len() != 1 {
        return Err(Error::new(
            Code::Invalid,
            format!("归约输出必须是单元素，得到 {} 个", out.len()),
        ));
    }
    let mut acc = R::start();
    for x in a {
        acc = R::fold(acc, *x);
    }
    out[0] = acc;
    Ok(())
}

/// f32 归约操作数：输入读（轴 n），输出写（轴 o）。
#[must_use]
pub const fn f32_operands() -> [OperandSpec; 2] {
    [
        OperandSpec::read("a", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ]
}
