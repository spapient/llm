//! 池化 / 插值 / 归一化的算子声明：几何是声明档固定常量，算术与标量参考直接复用
//! [`crate::conv`] 与 [`crate::norm`] 的真实现。

use crate::Scalar;
use crate::conv::ConvGeom;
use crate::model::{Buffer, Cost, Sample, ShapeContext, Values, x_buffer};
use diagnostic::Error;
use llm_gemm::unary::{AXES, project_shape};

/// 一次算子装配：标记类型、语义别名与 [`crate::operator_here`] 的基座实现，外加 `#[annotation::model]`
/// 自检常量。`$m`/`$n` 同时喂给 `operator_here!`（元组形状）与注解（数组形状），两处从同一对字面量派生。
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
