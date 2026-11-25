//! 池化 / 插值 / 归一化的算子声明：几何是声明档固定常量，算术与标量参考直接复用
//! [`crate::conv`] 与 [`crate::norm`] 的真实现。

use crate::Scalar;
use crate::conv::ConvGeom;
use crate::model::{Buf, Cost, Sample, ShapeCtx, x_buf};
use crate::unary::{AXES, project_shape};
use diagnostic::Error;

/// 池化声明共用的 same 3×3 几何：单位步长、单侧补零 1。
const POOL_SAME_3X3: ConvGeom = ConvGeom {
    kh: 3,
    kw: 3,
    sh: 1,
    sw: 1,
    ph: 1,
    pw: 1,
};

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

/// 窗口族公开面：形状、样本、参考、前向与代价在一处平铺声明；算术仍由模块的 `run` 提供。
macro_rules! window_op {
    ($ty:ident, $alias:ident, $op:tt, $m:tt, $n:tt, $in_len:expr, $out_len:expr,
     $flops:expr, $reads:expr) => {
        /// 构造这一窗口声明档的确定性输入与输出缓冲。
        pub fn sample(_: usize, _: usize) -> Sample {
            Sample::f32_io(
                $out_len,
                $out_len,
                $in_len,
                $out_len,
                vec![("N", $out_len as i64)],
            )
        }

        /// 用模块的真实窗口实现计算参考输出。
        pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
            let x = sample.f32("x")?;
            let mut out = vec![0.0f32; $out_len];
            run::<f32>(x, &mut out)?;
            Ok(out)
        }

        /// 声明档前向：与参考共用同一条 `run`，几何与样本一致。
        pub fn forward(x: tensor::View<'_, f32>, out: &mut [f32]) -> diagnostic::Result<()> {
            run::<f32>(x.as_slice(), out)
        }

        /// 标量代价：由声明的浮点运算数与读操作数推出。
        pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
            crate::model::cost_n(shape, $op, $flops, $reads)
        }

        classic_op!($ty, $alias, $op, 1, $m, $n);
    };
}

/// 归一化族公开面：共享 64×64 注解档；批／组／实例可省略共同的 `N*S*C` 与 `C` 几何。
macro_rules! norm_op {
    ($ty:ident, $alias:ident, $op:tt, $total:expr, $channels:expr, $scalars:expr,
     $gamma:expr, $beta:expr, $flops:expr, $reads:expr) => {
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
            x: tensor::View<'_, f32>,
            gamma: tensor::View<'_, f32>,
            beta: tensor::View<'_, f32>,
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            run::<f32>(x.as_slice(), gamma.as_slice(), beta.as_slice(), out)
        }

        /// 标量代价：显式口径，浮点运算与字节读写见本模块装配行的注释。
        pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
            crate::model::cost_n(shape, $op, $flops, $reads)
        }

        classic_op!($ty, $alias, $op, 3, 64, 64);
    };
    ($ty:ident, $alias:ident, $op:tt, $scalars:expr, $gamma:expr, $beta:expr,
     $flops:expr, $reads:expr) => {
        norm_op!(
            $ty,
            $alias,
            $op,
            N * S * C,
            C,
            $scalars,
            $gamma,
            $beta,
            $flops,
            $reads
        );
    };
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
        bufs: vec![
            x_buf(total),
            Buf::f32("gamma", vec![gamma; channels]),
            Buf::f32("beta", vec![beta; channels]),
            Buf::f32("y", vec![0.0; total]),
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
pub mod conv2d {
    use super::*;

    /// 输入高度。
    pub const H: usize = 4;
    /// 输入宽度或归一化宽度。
    pub const W: usize = 4;
    /// 输入通道数。
    pub const CIN: usize = 2;
    /// 输出通道数。
    pub const COUT: usize = 3;
    /// 卷积核边长。
    pub const K: usize = 3;
    /// 步长或空间尺寸。
    pub const S: usize = 1;
    /// 单侧补零宽度。
    pub const P: usize = 1;
    /// 输出高度。
    pub const OH: usize = 4;
    /// 输出宽度。
    pub const OW: usize = 4;
    /// 单个输出元素对应的卷积输入元素数。
    pub const PATCH: usize = K * K * CIN;

    fn conv() -> crate::conv::Conv2d {
        crate::conv::Conv2d::new(
            CIN,
            COUT,
            ConvGeom {
                kh: K,
                kw: K,
                sh: S,
                sw: S,
                ph: P,
                pw: P,
            },
        )
    }

    /// 构造确定性的卷积输入、权重、偏置与输出样本。
    pub fn sample(_: usize, _: usize) -> Sample {
        Sample {
            m: OH * OW,
            n: COUT,
            scalars: vec![("N", 1)],
            bufs: vec![
                x_buf(H * W * CIN),
                Buf::f32(
                    "weight",
                    (0..PATCH * COUT)
                        .map(|i| 0.05 + (i % 7) as f32 * 0.03)
                        .collect(),
                ),
                Buf::f32("bias", vec![0.4; COUT]),
                Buf::f32("y", vec![0.0; OH * OW * COUT]),
            ],
        }
    }

    /// 标量参考：读样本输入 / 权重 / 偏置，跑声明档二维卷积，返回输出向量。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let weight = sample.f32("weight")?;
        let bias = sample.f32("bias")?;
        let mut out = vec![0.0f32; OH * OW * COUT];
        conv().forward_f32(x, weight, bias, 1, (H, W), &mut out)?;
        Ok(out)
    }

    /// 执行声明档二维卷积前向计算。
    pub fn forward(
        x: tensor::View<'_, f32>,
        weight: tensor::View<'_, f32>,
        bias: tensor::View<'_, f32>,
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        conv().forward_f32(
            x.as_slice(),
            weight.as_slice(),
            bias.as_slice(),
            1,
            (H, W),
            out,
        )
    }

    /// 按声明档形状推导出算子代价（FLOPs 与权重条数）。
    pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
        crate::model::cost_n(shape, "conv2d_f32", 2.0 * PATCH as f64, 3)
    }

    classic_op!(Conv2dModel, Conv2d, "conv2d_f32", 3, 16, 3);
}

/// 窗口族共用 `window_op!`；子模块只保留各自几何与真执行入口。
pub mod pool {
    /// 自适应平均池化（声明档 4×4 → 2×2）。
    pub mod adaptive {
        use super::super::*;

        /// 输入高度。
        pub const H: usize = 4;
        /// 输入宽度或归一化宽度。
        pub const W: usize = 4;
        /// 输出高度。
        pub const OH: usize = 2;
        /// 输出宽度。
        pub const OW: usize = 2;

        pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
            crate::conv::AdaptiveAvgPool2d { oh: OH, ow: OW }.forward(x, 1, H, W, 1, out)
        }

        window_op!(
            AdaptivePoolModel,
            AdaptivePool,
            "adaptive_pool_f32",
            64,
            64,
            H * W,
            OH * OW,
            5.0,
            2
        );
    }

    /// 二维平均池化（声明档几何 `3×3, s=1, p=1`，补零不计入分母）。
    pub mod avg2d {
        use super::super::*;

        /// 输入高度。
        pub const H: usize = 4;
        /// 输入宽度或归一化宽度。
        pub const W: usize = 4;

        fn pool() -> crate::conv::AvgPool2d {
            crate::conv::AvgPool2d {
                geom: POOL_SAME_3X3,
                count_include_pad: false,
            }
        }

        pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
            pool().forward(x, 1, H, W, 1, out)
        }

        window_op!(
            AvgPool2dModel,
            AvgPool2d,
            "avg_pool2d_f32",
            64,
            64,
            H * W,
            H * W,
            10.0,
            4
        );
    }

    /// 一维最大池化（声明档几何 `k=2, s=2, p=0`，单通道）。
    pub mod max1d {
        use super::super::*;

        /// 输出序列长度。
        pub const OL: usize = 8;
        /// 输入序列长度。
        pub const L: usize = 16;

        pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
            crate::conv::MaxPool1d { k: 2, s: 2, p: 0 }.forward(x, 1, L, 1, out)
        }

        window_op!(
            MaxPool1dModel,
            MaxPool1d,
            "max_pool1d_f32",
            64,
            64,
            L,
            OL,
            2.0,
            2
        );
    }

    /// 二维最大池化（声明档几何 `3×3, s=1, p=1`，same 输出）。
    pub mod max2d {
        use super::super::*;

        /// 输入高度。
        pub const H: usize = 4;
        /// 输入宽度或归一化宽度。
        pub const W: usize = 4;

        pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
            crate::conv::MaxPool2d {
                geom: POOL_SAME_3X3,
            }
            .forward(x, 1, H, W, 1, out)
        }

        window_op!(
            MaxPool2dModel,
            MaxPool2d,
            "max_pool2d_f32",
            64,
            64,
            H * W,
            H * W,
            8.0,
            4
        );
    }
}

/// 双线性插值（声明档 2×2 → 4×4，`align_corners = false`）。
pub mod interpolate {
    use super::*;

    /// 输入高度。
    pub const H: usize = 2;
    /// 输入宽度或归一化宽度。
    pub const W: usize = 2;
    /// 输出高度。
    pub const OH: usize = 4;
    /// 输出宽度。
    pub const OW: usize = 4;

    pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
        crate::conv::Interpolate {
            sampler: crate::conv::Sampler::Linear,
            oh: OH,
            ow: OW,
        }
        .forward(x, 1, H, W, 1, out)
    }

    window_op!(
        InterpolateModel,
        Interpolate,
        "interpolate_f32",
        64,
        64,
        H * W,
        OH * OW,
        8.0,
        4
    );
}

/// 四种归一化只保留各自统计域与声明档参数。
pub mod norm {
    /// 批／组／实例归一化共享的通道数。
    pub const C: usize = 4;
    /// 共享的空间尺寸。
    pub const S: usize = 2;
    /// 共享的批大小。
    pub const N: usize = 2;
    /// 四种归一化共享的数值稳定性 epsilon。
    pub const EPS: f32 = 1e-5;

    /// 批归一化（声明档 `C=4, spatial=2, n=2`）。
    pub mod batch {
        use super::super::*;
        pub use super::{C, EPS, N, S};

        pub(crate) fn run<S: Scalar>(
            x: &[S],
            gamma: &[S],
            beta: &[S],
            out: &mut [S],
        ) -> diagnostic::Result<()> {
            let mut mean = vec![S::ZERO; C];
            let mut var = vec![S::ONE; C];
            crate::norm::BatchNorm2d {
                channels: C,
                eps: S::from_f64(f64::from(EPS)),
                momentum: S::from_f64(0.1),
            }
            .forward_train(x, gamma, beta, N, 1, S, &mut mean, &mut var, out)
        }

        norm_op!(
            BatchNormModel,
            BatchNorm,
            "batch_norm_f32",
            vec![("N", (N * S * C) as i64), ("C", C as i64), ("S", S as i64)],
            1.5,
            0.25,
            4.0,
            3
        );
    }

    /// 组归一化（声明档 `G=2, C=4, spatial=2, n=2`）。
    pub mod group {
        use super::super::*;
        pub use super::{C, EPS, N, S};

        /// 归一化分组数。
        pub const G: usize = 2;

        pub(crate) fn run<S: Scalar>(
            x: &[S],
            gamma: &[S],
            beta: &[S],
            out: &mut [S],
        ) -> diagnostic::Result<()> {
            crate::norm::GroupNorm {
                groups: G,
                channels: C,
                eps: S::from_f64(f64::from(EPS)),
            }
            .forward(x, gamma, beta, N, S, out)
        }

        norm_op!(
            GroupNormModel,
            GroupNorm,
            "group_norm_f32",
            vec![
                ("N", (N * S * C) as i64),
                ("G", G as i64),
                ("C", C as i64),
                ("S", S as i64),
            ],
            1.25,
            0.5,
            4.0,
            3
        );
    }

    /// 实例归一化（声明档 `C=4, spatial=2, n=2`）。
    pub mod instance {
        use super::super::*;
        pub use super::{C, EPS, N, S};

        pub(crate) fn run<S: Scalar>(
            x: &[S],
            gamma: &[S],
            beta: &[S],
            out: &mut [S],
        ) -> diagnostic::Result<()> {
            crate::norm::InstanceNorm {
                channels: C,
                eps: S::from_f64(f64::from(EPS)),
            }
            .forward(x, gamma, beta, N, S, out)
        }

        norm_op!(
            InstanceNormModel,
            InstanceNorm,
            "instance_norm_f32",
            vec![("N", (N * S * C) as i64), ("C", C as i64), ("S", S as i64)],
            0.75,
            -0.25,
            4.0,
            3
        );
    }

    /// 层归一化（声明档 `3 行 × 4 宽`）。
    pub mod layer {
        use super::super::*;
        pub use super::EPS;

        /// 层归一化行数。
        pub const R: usize = 3;
        /// 输入宽度或归一化宽度。
        pub const W: usize = 4;

        pub(crate) fn run<S: Scalar>(
            x: &[S],
            gamma: &[S],
            beta: &[S],
            out: &mut [S],
        ) -> diagnostic::Result<()> {
            crate::norm::LayerNorm {
                width: W,
                eps: S::from_f64(f64::from(EPS)),
            }
            .forward(x, gamma, beta, out)
        }

        norm_op!(
            LayerNormModel,
            LayerNorm,
            "layer_norm_f32",
            R * W,
            W,
            vec![("N", (R * W) as i64), ("R", R as i64), ("W", W as i64)],
            1.0,
            0.0,
            4.0,
            3
        );
    }
}

// 公开路径保持为 `classic::{max_pool1d, …, layer_norm}`，`lib.rs` 与测试据此取名。
pub use norm::{
    batch as batch_norm, group as group_norm, instance as instance_norm, layer as layer_norm,
};
pub use pool::{
    adaptive as adaptive_pool, avg2d as avg_pool2d, max1d as max_pool1d, max2d as max_pool2d,
};

#[cfg(test)]
mod tests {
    include!("../../tests/conv_classic_tests.rs");
}
