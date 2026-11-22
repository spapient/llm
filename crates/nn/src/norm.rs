//! 经典几何层之二：归一化的契约登记。
//!
//! 与 `spatial` 同属经典几何层族。拆开的判据是这一族**不碰窗口**：没有滑窗几何，只有
//! 通道与组的整除关系（`check_group_norm` 就是这条约束的校验），故两族的共享代数不重叠。
//! 层只声明形状与算子归属，执行仍在 `op::norm`。

use diagnostic::{Code, Error, Result};
// 归一化层现在是 `Scalar` 泛型；经典层登记只针对 `f32` 声明档，这里给出一份具体别名。
type OpBatch = op::norm::BatchNorm2d<f32>;
type OpGroup = op::norm::GroupNorm<f32>;
type OpInst = op::norm::InstanceNorm<f32>;
type OpLayer = op::norm::LayerNorm<f32>;
use crate::ContractSpec as Spec;
use crate::{Bounds, BoundsResult, Contract, ContractSpec, nonzero, spec_accessors};

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

/// 批归一化层（NHWC，最后维 `channels`）。
#[derive(Debug, Clone, PartialEq)]
pub struct BatchNorm2d {
    spec: ContractSpec,
    norm: OpBatch,
    channels: usize,
}

impl BatchNorm2d {
    /// 构造：`eps` 与 `momentum` 由调用方给。
    ///
    /// # Errors
    ///
    /// 通道为零、`eps ≤ 0` 或 `momentum` 越界时返回说明。
    pub fn new(name: impl Into<String>, channels: usize, eps: f32, momentum: f32) -> Result<Self> {
        let layer = "nn::BatchNorm2d::new";
        nonzero(channels, "channels", layer)?;
        check_eps(eps, layer)?;
        if !(0.0..=1.0).contains(&momentum) {
            return Err(Error::new(
                Code::Invalid,
                format!("{layer}: `momentum` 必须落在 [0, 1]"),
            )
            .with(layer));
        }
        let norm = OpBatch {
            channels,
            eps,
            momentum,
        };
        let spec = Spec::new(name, crate::Shape::new([channels]), ["batch_norm_f32"])?;
        Ok(Self {
            spec,
            norm,
            channels,
        })
    }

    /// 通道数。
    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl Contract for BatchNorm2d {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_ctx("batch_norm_f32", self.channels, 1).map(Some)
    }
}

/// 组归一化层：`(n, group)` 内按 `(C/G, spatial)` 统计。规格构造口径同 [`BatchNorm2d`]。
#[derive(Debug, Clone, PartialEq)]
pub struct GroupNorm {
    spec: ContractSpec,
    norm: OpGroup,
    channels: usize,
}

impl GroupNorm {
    /// 构造：`channels` 必须能被 `groups` 整除。
    ///
    /// # Errors
    ///
    /// 组数/通道为零、不能整除或 `eps ≤ 0` 时返回说明。
    pub fn new(name: impl Into<String>, groups: usize, channels: usize, eps: f32) -> Result<Self> {
        let layer = "nn::GroupNorm::new";
        check_group_norm(groups, channels, layer)?;
        check_eps(eps, layer)?;
        let norm = OpGroup {
            groups,
            channels,
            eps,
        };
        let spec = Spec::new(name, crate::Shape::new([channels]), ["group_norm_f32"])?;
        Ok(Self {
            spec,
            norm,
            channels,
        })
    }

    /// 通道数。
    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl Contract for GroupNorm {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_ctx("group_norm_f32", self.channels, 1).map(Some)
    }
}

/// 实例归一化层：等价于 `GroupNorm(groups = channels)`。规格构造口径同 [`BatchNorm2d`]。
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceNorm {
    spec: ContractSpec,
    norm: OpInst,
    channels: usize,
}

impl InstanceNorm {
    /// 构造。
    ///
    /// # Errors
    ///
    /// 通道为零或 `eps ≤ 0` 时返回说明。
    pub fn new(name: impl Into<String>, channels: usize, eps: f32) -> Result<Self> {
        let layer = "nn::InstanceNorm::new";
        nonzero(channels, "channels", layer)?;
        check_eps(eps, layer)?;
        let norm = OpInst { channels, eps };
        let spec = Spec::new(name, crate::Shape::new([channels]), ["instance_norm_f32"])?;
        Ok(Self {
            spec,
            norm,
            channels,
        })
    }

    /// 通道数。
    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl Contract for InstanceNorm {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_ctx("instance_norm_f32", self.channels, 1).map(Some)
    }
}

/// 层归一化层：逐行在最后一维 `width` 上统计。规格构造口径同 [`BatchNorm2d`]。
#[derive(Debug, Clone, PartialEq)]
pub struct LayerNorm {
    spec: ContractSpec,
    norm: OpLayer,
    width: usize,
}

impl LayerNorm {
    /// 构造。
    ///
    /// # Errors
    ///
    /// 宽度为零或 `eps ≤ 0` 时返回说明。
    pub fn new(name: impl Into<String>, width: usize, eps: f32) -> Result<Self> {
        let layer = "nn::LayerNorm::new";
        nonzero(width, "width", layer)?;
        check_eps(eps, layer)?;
        let norm = OpLayer { width, eps };
        let spec = Spec::new(name, crate::Shape::new([width]), ["layer_norm_f32"])?;
        Ok(Self { spec, norm, width })
    }

    /// 归一化宽度。
    pub fn width(&self) -> usize {
        self.width
    }
}

impl Contract for LayerNorm {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_ctx("layer_norm_f32", self.width, 1).map(Some)
    }
}
