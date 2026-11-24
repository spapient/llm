//! 逐元素一元骨架：`y[n] = f(a[n])`。

use crate::formula::FormulaOwner;
use crate::kernel::scalar::Scalar;
use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx, Values};
use diagnostic::{Code, Error};
use tensor::View;

pub(crate) mod ops;

/// 逐元素一元算子随语义变化的那部分。
///
/// 逐元素函数泛型到浮点标量档 `K: Scalar`（ADR 0054 §1 的「元素档 K」）：同一份闭包
/// 服务 `f32` / `f64` / `f16` / `bf16` 四档，无需为每档另写算子。对拍通道（样本 / 参考 /
/// 目录位宽）当前归一 f32，[`apply_f32`](Self::apply_f32) 即 `apply::<f32>` 的默认别名。
pub trait Unary: FormulaOwner {
    /// 逐元素函数：一个输入值，一个输出值，泛型到浮点标量档 `K`。
    fn apply<K: Scalar>(x: K) -> K;

    /// f32 标量核：对拍通道归一 f32 的入口，默认即 [`apply`](Self::apply) 的 f32 单态化。
    fn apply_f32(x: f32) -> f32 {
        Self::apply(x)
    }

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
pub fn project_shape(len: usize, _: usize) -> Result<ShapeCtx, Error> {
    ShapeCtx::new([('n', len)])
}

/// 标量代价：完全由表达式与操作数位宽推出。
pub fn cost<U: Unary>(shape: &ShapeCtx) -> Result<Cost, Error> {
    U::formula()?
        .cost(shape.extents())
        .map_err(|error| error.with("nn::unary::cost"))
}

/// 确定性样本：第一个参数是元素数。
pub fn sample<U: Unary>(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64)],
        bufs: vec![
            Buf {
                name: "a",
                values: Values::F32(U::input(len)),
            },
            Buf {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 标量参考：逐元素应用 f32 标量核 [`Unary::apply_f32`]。样本通道当前只有 f32，
/// 非 f32 档随取值通道批次接入。
pub fn reference<U: Unary>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let a = sample.f32("a")?;
    Ok(a.iter().map(|x| U::apply_f32(*x)).collect())
}

/// 内核本体：结果写进调用方给的缓冲，泛型到浮点标量档 `K`。
///
/// # Errors
///
/// 输入与输出等长，否则返回说明。
pub fn forward<K: Scalar, U: Unary>(a: View<'_, K>, out: &mut [K]) -> diagnostic::Result<()> {
    let a = a.as_slice();
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
        $name:ident, $alias:ident, $op:tt,
        $kind:ident, $edge:tt,
        $apply:expr, $diff:literal
        $(, $input:expr)?
    ) => {
        #[doc = concat!("`", $op, "` 的标记类型：逐元素一元运算见 `apply`，骨架见 `crate::unary`。")]
        pub struct $name;

        $crate::formula_owner!(
            $name,
            "n,n->n",
            $crate::einsum::f32_io_operands(),
            "nn::unary::formula"
        );

        impl $crate::unary::Unary for $name {
            // 闭包先绑定到 `fn(K) -> K` 函数指针：参数类型由该签名反推为泛型 `K`，
            // 不在调用点标注 `K`（调用点还进不了方法泛型作用域）。与二元族同一模式。
            fn apply<K: $crate::kernel::scalar::Scalar>(x: K) -> K {
                let apply: fn(K) -> K = $apply;
                apply(x)
            }

            $( fn input(len: usize) -> Vec<f32> {
                const INPUT: fn(usize) -> Vec<f32> = $input;
                INPUT(len)
            } )?
        }

        // 基座契约（ADR 0050）：只转发族骨架，不写第二份算术；骨架走绝对路径，算子模块
        // 嵌在哪一层都不依赖 `super` 的父模块解析。
        $crate::operator_via!(
            @impl $name, $op, 1, (4096, 4096), $diff, 1, $crate::unary::AXES,
            $crate::unary::project_shape,
            $crate::unary::cost::<$name>,
            $crate::unary::sample::<$name>,
            $crate::unary::reference::<$name>
        );

        #[annotation::model(name = $op, shape = [4096, 4096], dtype = "f32", skeleton = "unary")]
        #[doc = concat!("`", $op, "` 的语义别名。")]
        pub type $alias = $name;

        /// 内核入口：结果写进调用方给的缓冲。
        ///
        /// # Errors
        ///
        /// 输入与输出等长，否则返回说明。
        pub fn forward(a: tensor::View<'_, f32>, out: &mut [f32]) -> diagnostic::Result<()> {
            $crate::unary::forward::<f32, $name>(a, out)
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            $crate::unary::test_support::$kind!($name, forward);

            #[test]
            fn edge_cases_match_reference() {
                $edge
            }
        }
    };
    // 缺省可微：既有算子沿用 `true`；不可微调子在显式位传 `false`。
    (
        $name:ident, $alias:ident, $op:tt,
        $kind:ident, $edge:tt,
        $apply:expr
        $(, $input:expr)?
    ) => {
        $crate::unary_f32! {
            $name, $alias, $op, $kind, $edge, $apply, true $(, $input)?
        }
    };
}

#[cfg(test)]
pub(crate) mod test_support {
    include!("../../tests/kernel_unary_test_support.rs");
}
