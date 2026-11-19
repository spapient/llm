//! 逐元素一元骨架：`y[n] = f(a[n])`。
//!
//! 与 [`crate::elementwise`] 同构，差别在只读一个张量：一次循环读一个数、做一次逐元素函数、
//! 写一个数。`silu_f32` / `gelu_f32` 是同一副骨架的两种 [`Unary`] 实现，骨架（表达式缓存、
//! 代价、样本、校验、遍历）只有一份，按实现类型单态化。

use crate::einsum::{Formula, FormulaCache, OperandSpec};
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
    const OPERANDS: &'static [OperandSpec] = &crate::einsum::f32_io_operands();

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

    /// 输入的确定性取值：默认线性夹具 `−4.0 + 0.25·i`；需要不同取值时在实现里覆盖。
    fn input(len: usize) -> Vec<f32> {
        crate::model::linear(len, 0.25, -4.0)
    }
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

/// 生成一份 f32 一元逐元素算子的公开面：标记类型、[`Unary`] 实现、[`annotation::model`] 别名、
/// `forward` 入口与测试挂载。`scale` / `gelu` / `silu` 只差一行 `apply`（可选样本夹具）；
/// 新增一种一元运算加一次调用即可。
#[macro_export]
macro_rules! unary_f32 {
    (
        $name:ident, $alias:ident, $op:tt, $test:tt,
        $apply:expr
        $(, $input:expr)?
    ) => {
        #[doc = concat!("`", $op, "` 的标记类型：逐元素一元运算见 `apply`，骨架见 `crate::unary`。")]
        pub struct $name;

        impl $crate::unary::Unary for $name {
            fn cache() -> &'static $crate::einsum::FormulaCache {
                static CACHE: $crate::einsum::FormulaCache = $crate::einsum::FormulaCache::new();
                &CACHE
            }

            fn apply(x: f32) -> f32 {
                const APPLY: fn(f32) -> f32 = $apply;
                APPLY(x)
            }

            $( fn input(len: usize) -> Vec<f32> {
                const INPUT: fn(usize) -> Vec<f32> = $input;
                INPUT(len)
            } )?
        }

        #[annotation::model(name = $op, shape = [4096, 4096], dtype = "f32", skeleton = "unary")]
        #[doc = concat!("`", $op, "` 的语义别名。")]
        pub type $alias = $name;

        /// 内核入口：结果写进调用方给的缓冲。
        ///
        /// # Errors
        ///
        /// 输入与输出等长，否则返回说明。
        pub fn forward(a: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
            $crate::unary::forward::<$name>(a, out)
        }

        #[cfg(test)]
        #[path = $test]
        mod tests;
    };
}

#[cfg(test)]
pub(crate) mod test_support {
    /// 生成 scale/silu 等一元算子共有的三则测试：参考对拍、长度负例、饱和与重复边界。
    macro_rules! unary_op_tests {
        ($ty:ty, $forward:ident) => {
            #[test]
            fn matches_the_scalar_reference() {
                let sample = crate::unary::sample::<$ty>(128, 1);
                let expected = crate::unary::reference::<$ty>(&sample).expect("参考可计算");
                let mut out = vec![0.0; 128];
                let a = sample.f32("a").expect("a 存在");
                $forward(a, &mut out).expect("缓冲等长");
                assert_eq!(out, expected);
            }

            #[test]
            fn rejects_mismatched_length_and_allows_empty() {
                let mut out = vec![0.0; 3];
                assert!($forward(&[1.0, 2.0], &mut out).is_err(), "输出比输入长");
                let mut empty = vec![];
                $forward(&[], &mut empty).expect("空行逐元素放行");
            }

            #[test]
            fn saturating_and_repeated_inputs_match_reference() {
                let cases: &[&[f32]] = &[
                    &[0.0, 0.0],
                    &[1.0, 1.0, 1.0],
                    &[-1.0, -2.0, -3.0],
                    &[f32::MAX, f32::MAX],
                    &[f32::MIN, f32::MIN],
                ];
                for (index, a) in cases.iter().enumerate() {
                    let mut sample = crate::unary::sample::<$ty>(a.len(), 1);
                    sample.buffers[0].values = crate::model::Values::F32(a.to_vec());
                    let expected = crate::unary::reference::<$ty>(&sample).expect("参考可计算");
                    let mut out = vec![0.0; a.len()];
                    $forward(a, &mut out).expect("等长");
                    assert_eq!(out, expected, "用例 {index}");
                    assert!(out.iter().all(|v| v.is_finite()), "用例 {index}：结果有限");
                }
            }
        };
    }
    pub(crate) use unary_op_tests;
}
