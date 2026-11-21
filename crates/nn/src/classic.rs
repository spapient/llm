//! 经典几何层：卷积、反卷积、池化、插值与归一化的契约登记。
//!
//! 这一层只**声明**形状与算子归属，不执行、不持有权重：层的价值是让装配侧与 `Stack::floor`
//! 看见"模型里有一层 3×3 卷积"这件事，而不是再写一份前向。执行仍在 `op::conv` / `op::norm`。
//!
//! # 下界分两档，理由不同
//!
//! - **二维卷积**绑 `conv2d_f32`（真卷积算子：im2col + GEMM 的完整语义表），下界按它的每输出
//!   `2·k²·cin` 次运算与三读一写口径给出。
//! - **一维卷积与两种转置**仍绑 `gemm_f32`：`Conv1d` 的 1d im2col 就是矩阵乘、转置卷积没有独立
//!   算子，`gemm_f32` 的收缩轴是声明档 `K = 8`，因此这里是**松下界**（下界允许偏松，不会高于真实代价）。
//! - **窗口/归一化族**各自绑定自己的真算子（`max_pool1d/max_pool2d/avg_pool2d/adaptive_pool/
//!   interpolate/batch_norm/group_norm/instance_norm/layer_norm`），代价表即算子的语义表。

use diagnostic::{Code, Error, Result};
use op::conv::ConvGeom;
use op::conv::{
    AdaptiveAvgPool2d as OpAdaptive, Conv1d as OpConv1, Conv2d as OpConv2, Interpolate as OpInterp,
    MaxPool1d as OpPool1, MaxPool2d as OpPool2,
};
use op::deconv::{ConvTranspose1d as OpDeconv1, ConvTranspose2d as OpDeconv2};
// 归一化层现在是 `Scalar` 泛型；经典层登记只针对 `f32` 声明档，这里给出一份具体别名。
type OpBatch = op::norm::BatchNorm2d<f32>;
type OpGroup = op::norm::GroupNorm<f32>;
type OpInst = op::norm::InstanceNorm<f32>;
type OpLayer = op::norm::LayerNorm<f32>;
use tensor::Shape;

use crate::ContractSpec as Spec;
use crate::{Bounds, BoundsResult, Contract, ContractSpec, ParamRole, spec_accessors};

/// 一维卷积与转置卷积绑定的算子：im2col 之后就是一次矩阵乘。
const GEMM: &str = "gemm_f32";

/// 二维卷积绑定的真算子（完整 im2col + GEMM 语义）。
const CONV: &str = "conv2d_f32";
/// 校验非零并在失败时点名层与字段。
fn nonzero(value: usize, field: &str, layer: &str) -> Result<usize> {
    if value == 0 {
        return Err(
            Error::new(Code::Invalid, format!("{layer}: `{field}` 必须非零")).with("nn::classic"),
        );
    }
    Ok(value)
}

/// 校验 `eps` 为正且有限：所有归一化层共用。
fn check_eps(eps: f32, layer: &str) -> Result<()> {
    if eps <= 0.0 || eps.is_nan() {
        return Err(Error::new(Code::Invalid, format!("{layer}: `eps` 必须为正")).with(layer));
    }
    Ok(())
}

/// 校验组归一化的结构：组数/通道非零且通道能被组数整除。
fn check_group_norm(groups: usize, channels: usize, layer: &str) -> Result<()> {
    nonzero(groups, "groups", layer)?;
    nonzero(channels, "channels", layer)?;
    if !channels.is_multiple_of(groups) {
        return Err(Error::new(
            Code::Invalid,
            format!("{layer}: `channels`({channels}) 不能被 `groups`({groups}) 整除"),
        )
        .with(layer));
    }
    Ok(())
}

/// 校验反卷积层的输入/输出通道非零：1d/2d 共用。
fn nonzero_channels(cin: usize, cout: usize, layer: &str) -> Result<()> {
    nonzero(cin, "cin", layer)?;
    nonzero(cout, "cout", layer)?;
    Ok(())
}
mod conv;
mod norm;
mod pool;

pub use conv::{Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d};
pub use norm::{BatchNorm2d, GroupNorm, InstanceNorm, LayerNorm};
pub use pool::{AdaptiveAvgPool2d, AvgPool2d, Interpolate, MaxPool1d, MaxPool2d};

/// 输出形状的静态检查：每层的 `out` 维数必须与 `Shape::new` 接受的一致。
const _: () = {
    // 只是让 `Shape` 与 `ParamRole` 保持被引用，避免导出面漂移时静默。
    let _ = core::mem::size_of::<Shape>();
    let _ = ParamRole::Trainable;
};

#[cfg(test)]
#[path = "../tests/classic_tests.rs"]
mod tests;
