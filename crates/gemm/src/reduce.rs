//! 归约骨架：`y[0] = fold(a[0..n])`，输出单个元素。

use crate::formula::FormulaOwner;
use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

/// 归约算子随语义变化的部分。
pub trait Reduce: FormulaOwner {
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
    ($name:ident, $alias:ident, $op:tt, $cases:expr, $start:expr, $fold:expr) => {
        #[doc = concat!("`", $op, "` 的标记类型：初值与折叠见 `start`/`fold`，骨架见 `crate::reduce`。")]
        pub struct $name;

        impl $crate::formula::FormulaOwner for $name {
            const EXPR: &'static str = "n,o->o";

            const OPERANDS: &'static [$crate::einsum::OperandSpec] =
                &$crate::einsum::f32_io_operands();

            const FRAME: &'static str = "op::reduce::formula";

            fn cache() -> &'static $crate::einsum::FormulaCache {
                $crate::formula_cell!()
            }
        }

        impl $crate::reduce::Reduce for $name {
            fn start() -> f32 {
                $start
            }

            fn fold(acc: f32, x: f32) -> f32 {
                const FOLD: fn(f32, f32) -> f32 = $fold;
                FOLD(acc, x)
            }
        }

        // 基座契约（ADR 0050）：只转发族骨架，不写第二份算术。迁移期与 `#[model]` 生成的
        // `MODEL` 并存，两者一致性由 `operator::tests` 逐字段锁住。
        $crate::operator_via!($name, $op, 1, (4096, 1), true, super, $crate::reduce::AXES);

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
        mod tests {
            use super::*;
            $crate::reduce::test_support::reduce_op_tests!($name, forward, $start);
            $crate::reduce::test_support::reduce_saturating_tests!($name, forward, $cases);
        }
    };
}

#[cfg(test)]
#[path = "../tests/reduce_test_support.rs"]
pub(crate) mod test_support;

// 声明式算子表：同一副归约骨架的两种算子（最大值 / 求和），每个算子一个 `pub mod` 保持
// `op::<name>::MODEL` / `op::<name>::forward` 路径不变（`forward` 同名，必须隔在各自的模块里）。
// 新增一种归约在此补一段。

pub mod reduce_max_f32 {
    //! 最大值归约：`y[0] = max(a[0..n])`。

    crate::reduce_f32!(
        Max,
        ReduceMaxF32,
        "reduce_max_f32",
        &[
            &[0.0, 0.0, 0.0],
            &[f32::MIN, f32::MAX, -1.0],
            &[f32::MIN, f32::MIN],
            &[2.0, 2.0, 2.0],
            &[-3.0, -1.0, -2.0],
        ],
        f32::MIN,
        |acc, x| if x > acc { x } else { acc }
    );
}

pub mod reduce_sum_f32 {
    //! 求和归约：`y[0] = sum(a[0..n])`。

    crate::reduce_f32!(
        Sum,
        ReduceSumF32,
        "reduce_sum_f32",
        &[
            &[0.0, 0.0, 0.0],
            &[f32::MAX, f32::MAX],
            &[f32::MIN, f32::MIN],
            &[1.0, 1.0, 1.0, 1.0],
            &[-3.0, 0.0, 3.0],
        ],
        0.0,
        |acc, x| acc + x
    );
}
