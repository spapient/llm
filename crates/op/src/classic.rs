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
pub mod conv2d {
    use super::*;

    /// 声明档输入高。
    pub const H: usize = 4;
    /// 声明档输入宽。
    pub const W: usize = 4;
    /// 声明档输入通道。
    pub const CIN: usize = 2;
    /// 声明档输出通道。
    pub const COUT: usize = 3;
    /// 声明档核边长。
    pub const K: usize = 3;
    /// 声明档步长。
    pub const S: usize = 1;
    /// 声明档补零（`(K-1)/2`，保持 same 输出）。
    pub const P: usize = 1;
    /// 声明档输出高（same 卷积与输入同形）。
    pub const OH: usize = 4;
    /// 声明档输出宽。
    pub const OW: usize = 4;
    /// 每行 im2col 的收缩长度：`K·K·CIN`。
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

    /// 声明档样本：固定几何下的输入、权重、偏置与输出缓冲。
    pub fn sample(_: usize, _: usize) -> Sample {
        Sample {
            m: OH * OW,
            n: COUT,
            scalars: vec![("N", 1)],
            buffers: vec![
                x_buffer(H * W * CIN),
                Buffer {
                    name: "weight",
                    values: Values::F32(
                        (0..PATCH * COUT)
                            .map(|i| 0.05 + (i % 7) as f32 * 0.03)
                            .collect(),
                    ),
                },
                Buffer {
                    name: "bias",
                    values: Values::F32(vec![0.4; COUT]),
                },
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; OH * OW * COUT]),
                },
            ],
        }
    }

    /// 标量参考：复用 `im2col + GEMM` 的真实现算一遍（不是第二份卷积循环）。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let weight = sample.f32("weight")?;
        let bias = sample.f32("bias")?;
        let mut out = vec![0.0f32; OH * OW * COUT];
        conv().forward_f32(x, weight, bias, 1, (H, W), &mut out)?;
        Ok(out)
    }

    /// 声明档前向：单样本 `H×W×CIN → OH×OW×COUT`。
    pub fn forward(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        conv().forward_f32(x, weight, bias, 1, (H, W), out)
    }

    /// 标量代价：每输出 `2·K²·CIN` 次浮点运算；读 x/weight/bias 三遍、写 y 一遍。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        crate::model::cost_n(shape, "conv2d_f32", 2.0 * PATCH as f64, 3)
    }

    classic_op!(Conv2dModel, Conv2d, "conv2d_f32", 3, 16, 3);
}

/// 窗口族：最大池化、平均池化、自适应池化。
pub mod pool {
    //! 窗口族：最大池化（1d/2d）、平均池化与自适应池化，共用 [`crate::classic`] 的 `window_op!` 装配。
    /// 自适应平均池化（声明档 4×4 → 2×2）。
    pub mod adaptive {
        use super::super::*;

        /// 声明档输入边长。
        pub const H: usize = 4;
        /// 声明档输入宽（或行主序的行宽）。
        pub const W: usize = 4;
        /// 声明档输出边长。
        pub const OH: usize = 2;
        /// 声明档输出宽。
        pub const OW: usize = 2;

        /// 声明档算术：`<S: Scalar>` 泛型，四种精度共用一条自适应求平均的路径。
        pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
            crate::conv::AdaptiveAvgPool2d { oh: OH, ow: OW }.forward(x, 1, H, W, 1, out)
        }

        // 每格 4 元素求和 + 平均，读 2 倍、写一遍。
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

        /// 声明档输入边长。
        pub const H: usize = 4;
        /// 声明档输入宽（或行主序的行宽）。
        pub const W: usize = 4;

        fn pool() -> crate::conv::AvgPool2d {
            crate::conv::AvgPool2d {
                geom: ConvGeom {
                    kh: 3,
                    kw: 3,
                    sh: 1,
                    sw: 1,
                    ph: 1,
                    pw: 1,
                },
                count_include_pad: false,
            }
        }

        /// 声明档算术：`<S: Scalar>` 泛型，四种精度共用一条窗口求平均的路径。
        pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
            pool().forward(x, 1, H, W, 1, out)
        }

        // 窗口 9 次加法 + 1 次除法，读 4 倍、写一遍。
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

        /// 声明档输出位置数。
        pub const OL: usize = 8;
        /// 声明档输入长度：`(OL − 1)·S + K = 16`。
        pub const L: usize = 16;

        /// 声明档算术：`<S: Scalar>` 泛型，四种精度共用一条取窗口最大的路径。
        pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
            crate::conv::MaxPool1d { k: 2, s: 2, p: 0 }.forward(x, 1, L, 1, out)
        }

        // 每输出读两个输入元素、写一个；读两遍字节、写一遍。
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

    /// 二维最大池化（声明档几何 `3×3, s=1, p=1`，4×4 输入、same 输出）。
    pub mod max2d {
        use super::super::*;

        /// 声明档边长。
        pub const H: usize = 4;
        /// 声明档输入宽（或行主序的行宽）。
        pub const W: usize = 4;

        fn geom() -> ConvGeom {
            ConvGeom {
                kh: 3,
                kw: 3,
                sh: 1,
                sw: 1,
                ph: 1,
                pw: 1,
            }
        }

        /// 声明档算术：`<S: Scalar>` 泛型，四种精度共用一条取窗口最大的路径。
        pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
            crate::conv::MaxPool2d { geom: geom() }.forward(x, 1, H, W, 1, out)
        }

        // 窗口 9 次比较、读 9 倍字节上界、写一遍。
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

/// 归一化族：批 / 组 / 实例 / 层。
pub mod norm {
    //! 归一化族：批 / 组 / 实例 / 层，四种统计范围共用 [`crate::classic`] 的 `norm_op!` 装配。
    /// 批归一化（声明档 `C=4, spatial=2, n=2`，训练口径：批统计自上而下、滑动统计按动量混入）。
    pub mod batch {
        use super::super::*;

        /// 声明档通道数。
        pub const C: usize = 4;
        /// 声明档空间元素数。
        pub const S: usize = 2;
        /// 声明档批大小（或总元素数）。
        pub const N: usize = 2;
        /// 归一化分母的稳定项（前向与反向共用同一份常量）。
        pub const EPS: f32 = 1e-5;

        /// 声明档算术：`<S: Scalar>` 泛型，四种精度共用一条批归一化训练路径。
        pub(crate) fn run<S: Scalar>(
            x: &[S],
            gamma: &[S],
            beta: &[S],
            out: &mut [S],
        ) -> diagnostic::Result<()> {
            let mut mean = vec![S::ZERO; C];
            let mut var = vec![S::ONE; C];
            // 声明档是 `n=2, h=1, w=2`：`h·w = spatial`。
            crate::norm::BatchNorm2d {
                channels: C,
                eps: S::from_f64(EPS as f64),
                momentum: S::from_f64(0.1),
            }
            .forward_train(x, gamma, beta, N, 1, S, &mut mean, &mut var, out)
        }

        // 两趟读 x（均值、方差）+ 读 gamma/beta + 写 y，每元素约 4 次运算。
        norm_op!(
            BatchNormModel,
            BatchNorm,
            "batch_norm_f32",
            64,
            64,
            N * S * C,
            C,
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

        /// 声明档组数。
        pub const G: usize = 2;
        /// 声明档通道数。
        pub const C: usize = 4;
        /// 声明档空间元素数。
        pub const S: usize = 2;
        /// 声明档批大小（或总元素数）。
        pub const N: usize = 2;
        /// 归一化分母的稳定项（前向与反向共用同一份常量）。
        pub const EPS: f32 = 1e-5;

        /// 声明档算术：`<S: Scalar>` 泛型，四种精度共用一条组归一化路径。
        pub(crate) fn run<S: Scalar>(
            x: &[S],
            gamma: &[S],
            beta: &[S],
            out: &mut [S],
        ) -> diagnostic::Result<()> {
            crate::norm::GroupNorm {
                groups: G,
                channels: C,
                eps: S::from_f64(EPS as f64),
            }
            .forward(x, gamma, beta, N, S, out)
        }

        // 两趟统计 + 一趟归一化，读 3 遍、写一遍。
        norm_op!(
            GroupNormModel,
            GroupNorm,
            "group_norm_f32",
            64,
            64,
            N * S * C,
            C,
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

    /// 实例归一化（声明档 `C=4, spatial=2, n=2`）：等价于 `GroupNorm(G = C)`。
    pub mod instance {
        use super::super::*;

        /// 声明档通道数。
        pub const C: usize = 4;
        /// 声明档空间元素数。
        pub const S: usize = 2;
        /// 声明档批大小（或总元素数）。
        pub const N: usize = 2;
        /// 归一化分母的稳定项（前向与反向共用同一份常量）。
        pub const EPS: f32 = 1e-5;

        /// 声明档算术：`<S: Scalar>` 泛型，四种精度共用一条实例归一化路径。
        pub(crate) fn run<S: Scalar>(
            x: &[S],
            gamma: &[S],
            beta: &[S],
            out: &mut [S],
        ) -> diagnostic::Result<()> {
            crate::norm::InstanceNorm {
                channels: C,
                eps: S::from_f64(EPS as f64),
            }
            .forward(x, gamma, beta, N, S, out)
        }

        // 与组归一化同构（统计域更小、次数不变）。
        norm_op!(
            InstanceNormModel,
            InstanceNorm,
            "instance_norm_f32",
            64,
            64,
            N * S * C,
            C,
            vec![("N", (N * S * C) as i64), ("C", C as i64), ("S", S as i64)],
            0.75,
            -0.25,
            4.0,
            3
        );
    }

    /// 层归一化（声明档 `3 行 × 4 宽`）：逐行在最后一维统计。
    pub mod layer {
        use super::super::*;

        /// 声明档行数。
        pub const R: usize = 3;
        /// 声明档输入宽（或行主序的行宽）。
        pub const W: usize = 4;
        /// 归一化分母的稳定项（前向与反向共用同一份常量）。
        pub const EPS: f32 = 1e-5;

        /// 声明档算术：`<S: Scalar>` 泛型，四种精度共用一条层归一化路径。
        pub(crate) fn run<S: Scalar>(
            x: &[S],
            gamma: &[S],
            beta: &[S],
            out: &mut [S],
        ) -> diagnostic::Result<()> {
            crate::norm::LayerNorm {
                width: W,
                eps: S::from_f64(EPS as f64),
            }
            .forward(x, gamma, beta, out)
        }

        // 两趟行内统计 + 一趟归一化，读 3 遍、写一遍。
        norm_op!(
            LayerNormModel,
            LayerNorm,
            "layer_norm_f32",
            64,
            64,
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
/// 双线性插值（声明档 2×2 → 4×4，`align_corners = false`）。
pub mod interpolate {
    use super::*;

    /// 声明档输入边长。
    pub const H: usize = 2;
    /// 声明档输入宽（或行主序的行宽）。
    pub const W: usize = 2;
    /// 声明档输出边长。
    pub const OH: usize = 4;
    /// 声明档输出宽。
    pub const OW: usize = 4;

    /// 声明档算术：`<S: Scalar>` 泛型，四种精度共用一条双线性插值的路径。
    pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
        crate::conv::Interpolate {
            sampler: crate::conv::Sampler::Linear,
            oh: OH,
            ow: OW,
        }
        .forward(x, 1, H, W, 1, out)
    }

    // 每输出 4 邻 2 次线性插值，读 4 倍、写一遍。
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
