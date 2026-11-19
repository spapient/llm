//! 归约骨架：`y[0] = fold(a[0..n])`，输出单个元素。
//!
//! 输入轴 `n` 被规约消掉，输出轴 `o` 长度恒为 1。`reduce_sum_f32` / `reduce_max_f32` 是同一副
//! 骨架的两种 [`Reduce`] 实现，骨架（表达式缓存、代价、样本、校验、遍历）只有一份，按类型单态化。

use crate::einsum::{Formula, FormulaCache, OperandSpec};
use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

/// 归约算子随语义变化的部分。
pub trait Reduce {
    /// 表达式各项的位宽与方向（表达式固定为 `n,o->o`）。
    ///
    /// 默认 f32 一读一写；位宽不同的格式在实现里覆盖。
    const OPERANDS: &'static [OperandSpec] = &crate::einsum::f32_io_operands();

    /// 缓存单元格（每个实现一份），见 [`crate::gemv::Format::cache`]。
    fn cache() -> &'static FormulaCache;

    /// 表达式：固定的 `n,o->o`，第一次调用时解析并校验，之后复用。
    ///
    /// 表达式不随实现变化，所以它不是关联常量：多一份可以写歪的事实，而这条骨架本来就只有
    /// 一种求和形状（输出轴长度恒为 1）。
    ///
    /// # Errors
    ///
    /// 表达式与操作数表不自洽时返回说明（带 `op::reduce::formula` 帧）。
    fn formula() -> Result<&'static Formula, Error> {
        Self::cache().get("n,o->o", Self::OPERANDS, "op::reduce::formula")
    }

    /// 累加初值。
    fn start() -> f32;

    /// 一次折叠：把当前累加值与下一个元素合并。
    fn fold(acc: f32, x: f32) -> f32;

    /// 确定性输入：默认线性夹具 `−1.0 + 0.5·i`；需要不同取值时在实现里覆盖。
    fn input(len: usize) -> Vec<f32> {
        crate::model::linear(len, 0.5, -1.0)
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

/// 生成一份 f32 归约算子的公开面：标记类型、[`Reduce`] 实现、[`annotation::model`] 别名、
/// `forward` 入口与测试挂载。`reduce_max` / `reduce_sum` 只差初值与折叠闭包；
/// 新增一种归约加一次调用即可。
#[macro_export]
macro_rules! reduce_f32 {
    ($name:ident, $alias:ident, $op:tt, $test:tt, $start:expr, $fold:expr) => {
        #[doc = concat!("`", $op, "` 的标记类型：初值与折叠见 `start`/`fold`，骨架见 `crate::reduce`。")]
        pub struct $name;

        impl $crate::reduce::Reduce for $name {
            fn cache() -> &'static $crate::einsum::FormulaCache {
                static CACHE: $crate::einsum::FormulaCache = $crate::einsum::FormulaCache::new();
                &CACHE
            }

            fn start() -> f32 {
                $start
            }

            fn fold(acc: f32, x: f32) -> f32 {
                const FOLD: fn(f32, f32) -> f32 = $fold;
                FOLD(acc, x)
            }
        }

        #[annotation::model(name = $op, shape = [4096, 1], dtype = "f32", skeleton = "reduce")]
        #[doc = concat!("`", $op, "` 的语义别名。")]
        pub type $alias = $name;

        /// 内核入口：结果写进调用方给的单元素缓冲。
        ///
        /// # Errors
        ///
        /// 输出不是单元素时返回说明。
        pub fn forward(a: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
            $crate::reduce::forward::<$name>(a, out)
        }

        #[cfg(test)]
        #[path = $test]
        mod tests;
    };
}

#[cfg(test)]
pub(crate) mod test_support {
    /// 生成 reduce_max/reduce_sum 共有的两则测试：参考对拍、非单元素输出与空输入折叠。
    macro_rules! reduce_op_tests {
        ($ty:ty, $forward:ident, $empty:expr) => {
            #[test]
            fn matches_the_scalar_reference() {
                let sample = crate::reduce::sample::<$ty>(128, 1);
                let expected = crate::reduce::reference::<$ty>(&sample).expect("参考可计算");
                let mut out = vec![0.0];
                $forward(sample.f32("a").expect("a"), &mut out).expect("单元素输出");
                assert_eq!(out, expected);
            }

            #[test]
            fn rejects_non_singleton_output_and_handles_empty_input() {
                let mut wide = vec![0.0; 2];
                assert!(
                    $forward(&[1.0, 2.0], &mut wide).is_err(),
                    "归约输出必须是单元素"
                );
                let mut out = vec![0.0];
                $forward(&[], &mut out).expect("空输入折叠到初值");
                assert_eq!(out[0], $empty);
            }
        };
    }
    pub(crate) use reduce_op_tests;
}
