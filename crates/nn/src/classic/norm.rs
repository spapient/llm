//! `classic` 的归一化层族：Batch/Group/Instance/Layer。

use super::*;

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
        let spec = Spec::new(
            name,
            crate::Shape::new([channels]),
            vec!["batch_norm_f32".to_string()],
        )?;
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
        let spec = Spec::new(
            name,
            crate::Shape::new([channels]),
            vec!["group_norm_f32".to_string()],
        )?;
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
        let spec = Spec::new(
            name,
            crate::Shape::new([channels]),
            vec!["instance_norm_f32".to_string()],
        )?;
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
        let spec = Spec::new(
            name,
            crate::Shape::new([width]),
            vec!["layer_norm_f32".to_string()],
        )?;
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
