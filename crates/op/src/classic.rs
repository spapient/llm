//! 池化 / 插值 / 归一化的算子声明：几何是声明档固定常量，算术与标量参考直接复用
//! [`crate::conv`] 与 [`crate::norm`] 的真实现。
//!
//! 为什么要有这一层：`nn` 的层契约要求"一层至少引用一个已注册算子"，而池化的窗口取最大、
//! 归一化的行内减均值都不属于 `Skeleton` 闭集里的任何一种骨架。`attention_f32` 的既有做法是
//! "归到 elementwise、数学走标量函数"，这里同一口径：声明骨架 `elementwise`，语义表用
//! [`crate::model::cost_n`] 的显式口径（读输入一遍、写输出一遍），样本与参考是**真窗口/真统计**。
//!
//! 几何是声明档常量（窗口 2×2、3×3、4×4→2×2、2×2→4×4 等），内核模板按该档几何写；层侧的
//! 实际几何由 `op::conv` / `op::norm` 在计算时决定，两者不共享这份常量。
//!
//! 这些算子都标 `differentiable = true`：`train` 的反向公式目录（`train::backprop`）已为池化 /
//! 插值 / 归一化 / 二维卷积补齐梯度，与 `classic_op!` 传给 [`crate::operator_here`] 的 `true` 一致。
//!
//! 精度轴：每个窗口/归一化模块的 `run` 是 `<S: Scalar>` 泛型，四种精度（f32/f64/f16/bf16）
//! 共用同一条算术；公开的 `forward`/`reference` 仍以 f32 挂在算子注册表上（`DTYPE` 与参考缓冲
//! 都是 f32）。二维卷积暂不泛型：它下沉到 im2col + GEMM 的 f32/i8 路径，f16/bf16 无对应 GEMM。

use crate::Scalar;
use crate::conv::ConvGeom;
use crate::model::{Buffer, Cost, Sample, ShapeContext, Values, x_buffer};
use crate::unary::{AXES, project_shape};
use diagnostic::Error;

/// 一次算子装配：标记类型、语义别名与 [`crate::operator_here`] 的基座实现，外加 `#[annotation::model]`
/// 自检常量。`$m`/`$n` 同时喂给 `operator_here!`（元组形状）与注解（数组形状），两处从同一对字面量派生。
///
/// 模块内必须提供同名 `cost` / `sample` / `reference` 与 `project_shape` / `AXES`：基座宏按路径取，
/// 不在这里再抄一份。
macro_rules! classic_op {
    ($ty:ident, $alias:ident, $op:tt, $arity:literal, $m:tt, $n:tt) => {
        #[doc = concat!("`", $op, "` 的标记类型。")]
        #[derive(Debug, Clone, Copy)]
        pub struct $ty;

        crate::operator_here!($ty, $op, $arity, ($m, $n), true, reference);

        #[annotation::model(name = $op, shape = [$m, $n], dtype = "f32", skeleton = "elementwise")]
        /// 语义别名（`#[annotation::model]` 在此生成 `MODEL` 自检常量）。
        pub type $alias = $ty;
    };
}

/// 窗口族（池化 / 自适应 / 插值）的公开面：`sample` / `reference` / `forward` / `cost` 四个入口
/// 由本宏统一生成。算术在模块自带的 `run` 里（不藏进宏），缓冲搬运与代价口径复用
/// [`window_sample`] / [`window_reference`] / [`crate::model::cost_n`]。
macro_rules! window_op {
    ($ty:ident, $alias:ident, $op:tt, $m:tt, $n:tt, $in_len:expr, $out_len:expr,
     $flops:expr, $reads:expr) => {
        /// 声明档样本：`x` 与 `y` 的长度由模块几何常量给出。
        pub fn sample(_: usize, _: usize) -> Sample {
            window_sample($in_len, $out_len)
        }

        /// 标量参考：读 `x`、跑模块的 `run::<f32>`、返回 `y`（真窗口）。
        pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
            window_reference(sample, $out_len, run::<f32>)
        }

        /// 声明档前向：与参考共用同一条 `run`，几何与样本一致。
        pub fn forward(x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
            run::<f32>(x, out)
        }

        /// 标量代价：显式口径，浮点运算与字节读写见本模块装配行的注释。
        pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
            crate::model::cost_n(shape, $op, $flops, $reads)
        }

        classic_op!($ty, $alias, $op, 1, $m, $n);
    };
}

/// 归一化族（批 / 组 / 实例 / 层）的公开面：与 [`window_op!`] 同构，但 `sample` / `reference`
/// 多带 `gamma` / `beta` 两个参数缓冲，`forward` 签名多两个入参。
macro_rules! norm_op {
    ($ty:ident, $alias:ident, $op:tt, $m:tt, $n:tt,
     $total:expr, $channels:expr, $scalars:expr, $gamma:expr, $beta:expr,
     $flops:expr, $reads:expr) => {
        /// 声明档样本：`x` 与 `gamma`/`beta` 的长度由模块几何常量给出。
        pub fn sample(_: usize, _: usize) -> Sample {
            norm_sample($total, $channels, $scalars, $gamma, $beta)
        }

        /// 标量参考：读 `x`/`gamma`/`beta`、跑模块的 `run::<f32>`、返回 `y`（真统计）。
        pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
            norm_reference(sample, $total, run::<f32>)
        }

        /// 声明档前向：与参考共用同一条 `run`，几何与样本一致。
        pub fn forward(
            x: &[f32],
            gamma: &[f32],
            beta: &[f32],
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            run::<f32>(x, gamma, beta, out)
        }

        /// 标量代价：显式口径，浮点运算与字节读写见本模块装配行的注释。
        pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
            crate::model::cost_n(shape, $op, $flops, $reads)
        }

        classic_op!($ty, $alias, $op, 3, $m, $n);
    };
}

/// 窗口骨架的确定性样本：`x`（长 `in_len`）与 `y`（长 `out_len`），`N` 记输出长度。
///
/// 池化 / 自适应 / 插值五类算子共享这一份：几何不同，但样本结构同构。
fn window_sample(in_len: usize, out_len: usize) -> Sample {
    Sample {
        m: out_len,
        n: out_len,
        scalars: vec![("N", out_len as i64)],
        buffers: vec![
            x_buffer(in_len),
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; out_len]),
            },
        ],
    }
}

/// 窗口骨架的标量参考：读 `x`、跑 `run`、返回 `y`。算术在 `run` 里，这里只做缓冲搬运。
fn window_reference(
    sample: &Sample,
    out_len: usize,
    run: impl Fn(&[f32], &mut [f32]) -> diagnostic::Result<()>,
) -> diagnostic::Result<Vec<f32>> {
    let x = sample.f32("x")?;
    let mut out = vec![0.0f32; out_len];
    run(x, &mut out)?;
    Ok(out)
}

/// 归一化骨架的确定性样本：`x`（总长 `total`）、`gamma`/`beta`（各 `channels` 个）、`y`。
///
/// 批 / 组 / 实例 / 层四类归一化共享这一份：标量键与 `gamma`/`beta` 取值不同，结构同构。
fn norm_sample(
    total: usize,
    channels: usize,
    scalars: Vec<(&'static str, i64)>,
    gamma: f32,
    beta: f32,
) -> Sample {
    Sample {
        m: total,
        n: total,
        scalars,
        buffers: vec![
            x_buffer(total),
            Buffer {
                name: "gamma",
                values: Values::F32(vec![gamma; channels]),
            },
            Buffer {
                name: "beta",
                values: Values::F32(vec![beta; channels]),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; total]),
            },
        ],
    }
}

/// 归一化骨架的标量参考：读 `x`/`gamma`/`beta`、跑 `run`、返回 `y`。
fn norm_reference(
    sample: &Sample,
    total: usize,
    run: impl Fn(&[f32], &[f32], &[f32], &mut [f32]) -> diagnostic::Result<()>,
) -> diagnostic::Result<Vec<f32>> {
    let x = sample.f32("x")?;
    let gamma = sample.f32("gamma")?;
    let beta = sample.f32("beta")?;
    let mut out = vec![0.0f32; total];
    run(x, gamma, beta, &mut out)?;
    Ok(out)
}

/// 二维卷积（声明档 `4×4, cin 2, cout 3, k=3, s=1, p=1`，same 输出）。
pub mod conv2d;

pub mod pool;

/// 双线性插值（声明档 2×2 → 4×4，`align_corners = false`）。
pub mod interpolate;
pub mod norm;
// 公开路径保持为 `classic::{max_pool1d, …, layer_norm}`，`lib.rs` 与测试据此取名。
pub use norm::{
    batch as batch_norm, group as group_norm, instance as instance_norm, layer as layer_norm,
};
pub use pool::{
    adaptive as adaptive_pool, avg2d as avg_pool2d, max1d as max_pool1d, max2d as max_pool2d,
};

#[cfg(test)]
#[path = "../tests/classic_tests.rs"]
mod tests;
