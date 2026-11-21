//! 职责：`Geometry::declared` 与十个经典算子的声明档表。
//!
//! 声明档是梯度对拍的基准：固定几何逐字段对齐 `op::classic`，值全部取自 `op::*` 的
//! `pub const`，不在训练侧另写一份算术常量。

use diagnostic::{Code, Error, Result};

use super::Geometry;

impl Geometry {
    /// 声明档实例：与 `op::classic` 里的固定几何逐字段对齐（梯度对拍的基准）。
    ///
    /// # Errors
    ///
    /// `op` 不在十个经典算子之列时返回 [`Code::Unsupported`]。
    pub fn declared(op: &str) -> Result<Self> {
        DECLARED
            .iter()
            .find(|(name, _)| *name == op)
            .map(|(_, geom)| *geom)
            .ok_or_else(|| {
                Error::new(Code::Unsupported, format!("`{op}` 没有声明档实例几何"))
                    .with("train::Geometry::declared")
            })
    }
}

/// 十个经典算子的声明档实例：`declared` 的事实来源，一份表、一个数据源。
///
/// 值全部取自 `op::*` 的 `pub const`，不在训练侧另写一份算术常量。
const DECLARED: &[(&str, Geometry)] = &[
    (
        "max_pool1d_f32",
        Geometry::Pool1d {
            k: 2,
            s: 2,
            p: 0,
            l: op::max_pool1d::L,
            c: 1,
        },
    ),
    (
        "max_pool2d_f32",
        Geometry::Pool2d {
            k: 3,
            s: 1,
            p: 1,
            h: op::max_pool2d::H,
            w: op::max_pool2d::W,
            c: 1,
        },
    ),
    (
        "avg_pool2d_f32",
        Geometry::Avg2d {
            k: 3,
            s: 1,
            p: 1,
            h: op::avg_pool2d::H,
            w: op::avg_pool2d::W,
            c: 1,
            count_include_pad: false,
        },
    ),
    (
        "adaptive_pool_f32",
        Geometry::Adaptive {
            oh: op::adaptive_pool::OH,
            ow: op::adaptive_pool::OW,
            h: op::adaptive_pool::H,
            w: op::adaptive_pool::W,
            c: 1,
        },
    ),
    (
        "interpolate_f32",
        Geometry::Interp {
            oh: op::interpolate::OH,
            ow: op::interpolate::OW,
            h: op::interpolate::H,
            w: op::interpolate::W,
            c: 1,
            linear: true,
        },
    ),
    (
        "conv2d_f32",
        Geometry::Conv2d {
            k: op::conv2d::K,
            s: op::conv2d::S,
            p: op::conv2d::P,
            h: op::conv2d::H,
            w: op::conv2d::W,
            cin: op::conv2d::CIN,
            cout: op::conv2d::COUT,
        },
    ),
    (
        "batch_norm_f32",
        Geometry::Norm {
            n: op::batch_norm::N,
            spatial: op::batch_norm::S,
            channels: op::batch_norm::C,
        },
    ),
    (
        "group_norm_f32",
        Geometry::Group {
            groups: op::group_norm::G,
            n: op::group_norm::N,
            spatial: op::group_norm::S,
            channels: op::group_norm::C,
        },
    ),
    (
        "instance_norm_f32",
        Geometry::Group {
            groups: op::instance_norm::C,
            n: op::instance_norm::N,
            spatial: op::instance_norm::S,
            channels: op::instance_norm::C,
        },
    ),
    (
        "layer_norm_f32",
        Geometry::Layer {
            rows: op::layer_norm::R,
            width: op::layer_norm::W,
        },
    ),
];
