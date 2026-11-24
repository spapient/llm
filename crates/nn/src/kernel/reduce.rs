//! 归约骨架：`y[0] = fold(a[0..n])`，输出单个元素。

use crate::formula::FormulaOwner;
use crate::kernel::scalar::Scalar;
use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx, Values};
use diagnostic::{Code, Error};
use tensor::View;

/// 归约算子随语义变化的部分。
///
/// 初值与折叠泛型到浮点标量档 `K: Scalar`（ADR 0054 §1 的「元素档 K」）：同一份折叠
/// 服务四档浮点。对拍通道归一 f32，[`start_f32`](Self::start_f32) / [`fold_f32`](Self::fold_f32)
/// 即 `start::<f32>` / `fold::<f32>` 的默认别名。
pub trait Reduce: FormulaOwner {
    /// 累加初值，泛型到浮点标量档 `K`。
    fn start<K: Scalar>() -> K;

    /// 一次折叠：把当前累加值与下一个元素合并，泛型到浮点标量档 `K`。
    fn fold<K: Scalar>(acc: K, x: K) -> K;

    /// f32 标量初值：对拍通道归一 f32 的入口，默认即 [`start`](Self::start) 的 f32 单态化。
    fn start_f32() -> f32 {
        Self::start()
    }

    /// f32 折叠：对拍通道归一 f32 的入口，默认即 [`fold`](Self::fold) 的 f32 单态化。
    fn fold_f32(acc: f32, x: f32) -> f32 {
        Self::fold(acc, x)
    }

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
pub fn project_shape(len: usize, _: usize) -> Result<ShapeCtx, Error> {
    ShapeCtx::new([('n', len), ('o', 1)])
}

/// 标量代价。
pub fn cost<R: Reduce>(shape: &ShapeCtx) -> Result<Cost, Error> {
    R::formula()?
        .cost(shape.extents())
        .map_err(|error| error.with("nn::reduce::cost"))
}

/// 确定性样本：输入 len 个，输出 1 个。
pub fn sample<R: Reduce>(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: 1,
        scalars: vec![("N", len as i64)],
        bufs: vec![
            Buf {
                name: "a",
                values: Values::F32(R::input(len)),
            },
            Buf {
                name: "y",
                values: Values::F32(vec![0.0]),
            },
        ],
    }
}

/// 标量参考：用 f32 折叠整个输入（对拍通道归一 f32）。
pub fn reference<R: Reduce>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let a = sample.f32("a")?;
    let mut acc = R::start_f32();
    for x in a {
        acc = R::fold_f32(acc, *x);
    }
    Ok(vec![acc])
}

/// 内核本体：结果写进调用方给的单元素缓冲，泛型到浮点标量档 `K`。
///
/// # Errors
///
/// 输出缓冲不是单元素时返回说明。
pub fn forward<K: Scalar, R: Reduce>(a: View<'_, K>, out: &mut [K]) -> diagnostic::Result<()> {
    let a = a.as_slice();
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
    ($name:ident, $alias:ident, $op:tt, $cases:expr, $start:expr, $fold:expr, $diff:literal) => {
        #[doc = concat!("`", $op, "` 的标记类型：初值与折叠见 `start`/`fold`，骨架见 `crate::reduce`。")]
        pub struct $name;

        $crate::formula_owner!(
            $name,
            "n,o->o",
            $crate::einsum::f32_io_operands(),
            "nn::reduce::formula"
        );

        impl $crate::reduce::Reduce for $name {
            fn start<K: $crate::kernel::scalar::Scalar>() -> K {
                K::from_f64(($start) as f64)
            }

            fn fold<K: $crate::kernel::scalar::Scalar>(acc: K, x: K) -> K {
                let fold: fn(K, K) -> K = $fold;
                fold(acc, x)
            }
        }

        // 基座契约（ADR 0050）：只转发族骨架，不写第二份算术。迁移期与 `#[model]` 生成的
        // `MODEL` 并存，两者一致性由 `operator::tests` 逐字段锁住。
        $crate::operator_via!($name, $op, 1, (4096, 1), $diff, super, $crate::reduce::AXES);

        #[annotation::model(name = $op, shape = [4096, 1], dtype = "f32", skeleton = "reduce")]
        #[doc = concat!("`", $op, "` 的语义别名。")]
        pub type $alias = $name;

        /// 内核入口：结果写进调用方给的单元素缓冲。
        ///
        /// # Errors
        ///
        /// 输出不是单元素时返回说明。
        pub fn forward(a: tensor::View<'_, f32>, out: &mut [f32]) -> diagnostic::Result<()> {
            $crate::reduce::forward::<f32, $name>(a, out)
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            $crate::reduce::test_support::reduce_op_tests!($name, forward, $start);
            $crate::reduce::test_support::reduce_saturating_tests!($name, forward, $cases);
        }
    };
    // 缺省可微：既有算子沿用 `true`；不可微调子在显式位传 `false`。
    ($name:ident, $alias:ident, $op:tt, $cases:expr, $start:expr, $fold:expr) => {
        $crate::reduce_f32! { $name, $alias, $op, $cases, $start, $fold, true }
    };
}

#[cfg(test)]
mod test_support {
    include!("../../tests/kernel_reduce_test_support.rs");
}

// 声明式算子表：同一副归约骨架的两种算子（最大值 / 求和），每个算子一个 `pub mod` 保持
// `nn::<name>::MODEL` / `nn::<name>::forward` 路径不变（`forward` 同名，必须隔在各自的模块里）。
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

pub mod reduce_prod_f32 {
    //! 连乘归约：`y[0] = ∏ a[0..n]`；空输入折叠到初值 1。

    crate::reduce_f32!(
        Prod,
        ReduceProdF32,
        "reduce_prod_f32",
        &[
            &[1.0, 1.0, 1.0],
            &[2.0, 3.0, 2.0],
            &[0.0, 4.0],
            &[-1.0, -2.0, 3.0],
        ],
        1.0,
        |acc, x| acc * x,
        false
    );
}
