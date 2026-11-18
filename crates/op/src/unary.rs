//! 逐元素一元骨架：`y[n] = f(a[n])`。
//!
//! 与 [`crate::elementwise`] 同构，差别在只读一个张量：一次循环读一个数、做一次逐元素函数、
//! 写一个数。`silu_f32` / `gelu_f32` 是同一副骨架的两种 [`Unary`] 实现，骨架（表达式缓存、
//! 代价、样本、校验、遍历）只有一份，按实现类型单态化。

use crate::einsum::{Bits, Formula, FormulaCache, OperandSpec};
use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

/// 逐元素一元算子随语义变化的那部分。
pub trait Unary {
    /// 一条 einsum 表达式：箭头左两项（一个读、一个写），共享一条轴。
    ///
    /// 一元逐元素形状只有这一种，默认值即标准规格；需要别的表达式时在实现里覆盖。
    const EXPR: &'static str = "n,n->n";

    /// 表达式各项的位宽与方向，顺序与 [`Unary::EXPR`] 一一对应。
    ///
    /// 默认 f32 一读一写；位宽不同的格式在实现里覆盖。
    const OPERANDS: &'static [OperandSpec] = &f32_operands();

    /// 缓存单元格（每个实现一份），见 [`crate::gemv::Format::cache`]。
    fn cache() -> &'static FormulaCache;

    /// 表达式：第一次调用时解析并校验，之后复用。
    ///
    /// # Errors
    ///
    /// 表达式与操作数表不自洽时返回说明（带 `op::unary::formula` 帧）。
    fn formula() -> Result<&'static Formula, Error> {
        Self::cache().get(Self::EXPR, Self::OPERANDS, "op::unary::formula")
    }

    /// 逐元素函数：一个输入值，一个输出值。
    fn apply(x: f32) -> f32;

    /// 输入的确定性取值。
    fn input(len: usize) -> Vec<f32>;
}

/// 一元骨架只有一条轴：既是输入轴也是输出轴。
pub const AXES: &[AxisSpec] = &[AxisSpec {
    role: AxisRole::Input,
    label: 'n',
}];

/// 单轴形状：轴 `n` 取第一个参数（元素数）。
///
/// # Errors
///
/// 形状上下文构造失败时返回说明（当前实现不会失败）。
pub fn project_shape(len: usize, _: usize) -> Result<ShapeContext, Error> {
    ShapeContext::new([('n', len)])
}

/// 标量代价：完全由表达式与操作数位宽推出。
pub fn cost<U: Unary>(shape: &ShapeContext) -> Result<Cost, Error> {
    U::formula()?
        .cost(shape.extents())
        .map_err(|error| error.with("op::unary::cost"))
}

/// 确定性样本：第一个参数是元素数。
pub fn sample<U: Unary>(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64)],
        buffers: vec![
            Buffer {
                name: "a",
                values: Values::F32(U::input(len)),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 标量参考：逐元素应用 [`Unary::apply`]。
pub fn reference<U: Unary>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let a = sample.f32("a")?;
    Ok(a.iter().map(|x| U::apply(*x)).collect())
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输入与输出等长，否则返回说明。
pub fn forward<U: Unary>(a: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    if a.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不一致：a {}、out {}", a.len(), out.len()),
        ));
    }
    // 切片 zip：连续内存、无下标边界检查，便于自动向量化。
    for (out, x) in out.iter_mut().zip(a) {
        *out = U::apply(*x);
    }
    Ok(())
}

/// f32 一元操作数的标准规格：一个读、一个写。
#[must_use]
pub const fn f32_operands() -> [OperandSpec; 2] {
    [
        OperandSpec::read("a", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ]
}
