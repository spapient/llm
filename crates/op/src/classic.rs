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
//! 这些算子当前标 `differentiable = false`：`train` 的反向公式目录里还没有池化/插值的梯度，
//! 标 true 会立刻撞上"可微算子必须有反向公式"的编译期断言。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values, x_buffer};
use crate::conv::ConvGeom;
use diagnostic::Error;

/// 一次算子装配：标记类型、语义别名与 [`crate::operator_here`] 的基座实现。
///
/// 模块内必须提供同名 `cost` / `sample` / `reference` 与 `project_shape` / `AXES`：基座宏按路径取，
/// 不在这里再抄一份。
macro_rules! classic_op {
    ($ty:ident, $op:literal, $arity:literal, $shape:expr) => {
        #[doc = concat!("`", $op, "` 的标记类型。")]
        #[derive(Debug, Clone, Copy)]
        pub struct $ty;

        crate::operator_here!($ty, $op, $arity, $shape, false, reference);
    };
}

/// 一维最大池化（声明档几何 `k=2, s=2, p=0`，单通道）。
pub mod max_pool1d_f32 {

    use super::*;
    pub use crate::unary::{AXES, project_shape};

    /// 声明档输出位置数。
    const OL: usize = 8;
    /// 声明档输入长度：`(OL − 1)·S + K = 16`。
    const L: usize = 16;

    /// 声明档样本：固定几何下的输入与输出缓冲。
    pub fn sample(_: usize, _: usize) -> Sample {
        Sample {
            m: OL,
            n: OL,
            scalars: vec![("N", OL as i64)],
            buffers: vec![
                x_buffer(L),
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; OL]),
                },
            ],
        }
    }

    /// 标量参考：复用 conv/norm 的真实现算一遍（真窗口 / 真统计）。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let mut out = vec![0.0f32; OL];
        crate::conv::MaxPool1d { k: 2, s: 2, p: 0 }.forward(x, 1, L, 1, &mut out)?;
        Ok(out)
    }

    /// 声明档前向：单通道 `L → OL`（几何与样本一致）。
    pub fn forward(x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        crate::conv::MaxPool1d { k: 2, s: 2, p: 0 }.forward(x, 1, L, 1, out)
    }

    /// 标量代价：每输出读两个输入元素、写一个；读两遍字节、写一遍。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        crate::model::cost_n(shape, "max_pool1d_f32", 2.0, 2)
    }

    classic_op!(MaxPool1dModel, "max_pool1d_f32", 1, (64, 64));
    #[annotation::model(name = "max_pool1d_f32", shape = [64, 64], dtype = "f32", skeleton = "elementwise")]
    /// 语义别名（`#[annotation::model]` 在此生成 `MODEL` 自检常量）。
    pub type MaxPool1dF32 = MaxPool1dModel;
}

/// 二维最大池化（声明档几何 `3×3, s=1, p=1`，4×4 输入、same 输出）。
pub mod max_pool2d_f32 {

    use super::*;
    pub use crate::unary::{AXES, project_shape};

    /// 声明档边长。
    const H: usize = 4;
    const W: usize = 4;

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

    /// 声明档样本：固定几何下的输入与输出缓冲。
    pub fn sample(_: usize, _: usize) -> Sample {
        Sample {
            m: H * W,
            n: H * W,
            scalars: vec![("N", (H * W) as i64)],
            buffers: vec![
                x_buffer(H * W),
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; H * W]),
                },
            ],
        }
    }

    /// 标量参考：复用 conv/norm 的真实现算一遍（真窗口 / 真统计）。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let mut out = vec![0.0f32; H * W];
        crate::conv::MaxPool2d { geom: geom() }.forward(x, 1, H, W, 1, &mut out)?;
        Ok(out)
    }

    /// 声明档前向：单通道 `H×W`，same 输出。
    pub fn forward(x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        crate::conv::MaxPool2d { geom: geom() }.forward(x, 1, H, W, 1, out)
    }

    /// 标量代价：窗口 9 次比较、读 9 倍字节上界、写一遍。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        crate::model::cost_n(shape, "max_pool2d_f32", 8.0, 4)
    }

    classic_op!(MaxPool2dModel, "max_pool2d_f32", 1, (64, 64));
    #[annotation::model(name = "max_pool2d_f32", shape = [64, 64], dtype = "f32", skeleton = "elementwise")]
    /// 语义别名（`#[annotation::model]` 在此生成 `MODEL` 自检常量）。
    pub type MaxPool2dF32 = MaxPool2dModel;
}

/// 二维平均池化（声明档几何 `3×3, s=1, p=1`，补零不计入分母）。
pub mod avg_pool2d_f32 {

    use super::*;
    pub use crate::unary::{AXES, project_shape};

    const H: usize = 4;
    const W: usize = 4;

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

    /// 声明档样本：固定几何下的输入与输出缓冲。
    pub fn sample(_: usize, _: usize) -> Sample {
        Sample {
            m: H * W,
            n: H * W,
            scalars: vec![("N", (H * W) as i64)],
            buffers: vec![
                x_buffer(H * W),
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; H * W]),
                },
            ],
        }
    }

    /// 标量参考：复用 conv/norm 的真实现算一遍（真窗口 / 真统计）。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let mut out = vec![0.0f32; H * W];
        pool().forward(x, 1, H, W, 1, &mut out)?;
        Ok(out)
    }

    /// 声明档前向：单通道 `H×W`，补零不计入分母。
    pub fn forward(x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        pool().forward(x, 1, H, W, 1, out)
    }

    /// 标量代价：窗口 9 次加法 + 1 次除法，读 4 倍、写一遍。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        crate::model::cost_n(shape, "avg_pool2d_f32", 10.0, 4)
    }

    classic_op!(AvgPool2dModel, "avg_pool2d_f32", 1, (64, 64));
    #[annotation::model(name = "avg_pool2d_f32", shape = [64, 64], dtype = "f32", skeleton = "elementwise")]
    /// 语义别名（`#[annotation::model]` 在此生成 `MODEL` 自检常量）。
    pub type AvgPool2dF32 = AvgPool2dModel;
}

/// 自适应平均池化（声明档 4×4 → 2×2）。
pub mod adaptive_pool_f32 {

    use super::*;
    pub use crate::unary::{AXES, project_shape};

    const H: usize = 4;
    const W: usize = 4;
    const OH: usize = 2;
    const OW: usize = 2;

    /// 声明档样本：固定几何下的输入与输出缓冲。
    pub fn sample(_: usize, _: usize) -> Sample {
        Sample {
            m: OH * OW,
            n: OH * OW,
            scalars: vec![("N", (OH * OW) as i64)],
            buffers: vec![
                x_buffer(H * W),
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; OH * OW]),
                },
            ],
        }
    }

    /// 标量参考：复用 conv/norm 的真实现算一遍（真窗口 / 真统计）。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let mut out = vec![0.0f32; OH * OW];
        crate::conv::AdaptiveAvgPool2d { oh: OH, ow: OW }.forward(x, 1, H, W, 1, &mut out)?;
        Ok(out)
    }

    /// 声明档前向：单通道 `H×W → OH×OW`。
    pub fn forward(x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        crate::conv::AdaptiveAvgPool2d { oh: OH, ow: OW }.forward(x, 1, H, W, 1, out)
    }

    /// 标量代价：每格 4 元素求和 + 平均，读 2 倍、写一遍。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        crate::model::cost_n(shape, "adaptive_pool_f32", 5.0, 2)
    }

    classic_op!(AdaptivePoolModel, "adaptive_pool_f32", 1, (64, 64));
    #[annotation::model(name = "adaptive_pool_f32", shape = [64, 64], dtype = "f32", skeleton = "elementwise")]
    /// 语义别名（`#[annotation::model]` 在此生成 `MODEL` 自检常量）。
    pub type AdaptivePoolF32 = AdaptivePoolModel;
}

/// 双线性插值（声明档 2×2 → 4×4，`align_corners = false`）。
pub mod interpolate_f32 {

    use super::*;
    pub use crate::unary::{AXES, project_shape};

    const H: usize = 2;
    const W: usize = 2;
    const OH: usize = 4;
    const OW: usize = 4;

    fn sampler() -> crate::conv::Sampler {
        crate::conv::Sampler::Linear
    }

    /// 声明档样本：固定几何下的输入与输出缓冲。
    pub fn sample(_: usize, _: usize) -> Sample {
        Sample {
            m: OH * OW,
            n: OH * OW,
            scalars: vec![("N", (OH * OW) as i64)],
            buffers: vec![
                x_buffer(H * W),
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; OH * OW]),
                },
            ],
        }
    }

    /// 标量参考：复用 conv/norm 的真实现算一遍（真窗口 / 真统计）。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let mut out = vec![0.0f32; OH * OW];
        crate::conv::Interpolate {
            sampler: sampler(),
            oh: OH,
            ow: OW,
        }
        .forward(x, 1, H, W, 1, &mut out)?;
        Ok(out)
    }

    /// 声明档前向：单通道 `H×W → OH×OW`（双线性）。
    pub fn forward(x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        crate::conv::Interpolate {
            sampler: sampler(),
            oh: OH,
            ow: OW,
        }
        .forward(x, 1, H, W, 1, out)
    }

    /// 标量代价：每输出 4 邻 2 次线性插值，读 4 倍、写一遍。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        crate::model::cost_n(shape, "interpolate_f32", 8.0, 4)
    }

    classic_op!(InterpolateModel, "interpolate_f32", 1, (64, 64));
    #[annotation::model(name = "interpolate_f32", shape = [64, 64], dtype = "f32", skeleton = "elementwise")]
    /// 语义别名（`#[annotation::model]` 在此生成 `MODEL` 自检常量）。
    pub type InterpolateF32 = InterpolateModel;
}

/// 批归一化（声明档 `C=4, spatial=2, n=2`，训练口径：批统计自上而下、滑动统计按动量混入）。
pub mod batch_norm_f32 {

    use super::*;
    pub use crate::unary::{AXES, project_shape};

    const C: usize = 4;
    const S: usize = 2;
    const N: usize = 2;

    fn norm() -> crate::norm::BatchNorm2d {
        crate::norm::BatchNorm2d {
            channels: C,
            eps: 1e-5,
            momentum: 0.1,
        }
    }

    /// 声明档样本：固定几何下的输入与输出缓冲。
    pub fn sample(_: usize, _: usize) -> Sample {
        Sample {
            m: N * S * C,
            n: N * S * C,
            scalars: vec![("N", (N * S * C) as i64), ("C", C as i64), ("S", S as i64)],
            buffers: vec![
                x_buffer(N * S * C),
                Buffer {
                    name: "gamma",
                    values: Values::F32(vec![1.5; C]),
                },
                Buffer {
                    name: "beta",
                    values: Values::F32(vec![0.25; C]),
                },
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; N * S * C]),
                },
            ],
        }
    }

    /// 标量参考：复用 conv/norm 的真实现算一遍（真窗口 / 真统计）。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let gamma = sample.f32("gamma")?;
        let beta = sample.f32("beta")?;
        let mut out = vec![0.0f32; N * S * C];
        let mut mean = vec![0.0f32; C];
        let mut var = vec![1.0f32; C];
        // 声明档是 `n=2, h=1, w=2`：`h·w = spatial`。
        norm().forward_train(x, gamma, beta, N, 1, S, &mut mean, &mut var, &mut out)?;
        Ok(out)
    }

    /// 声明档前向：批统计走训练口径（`running` 缓冲就地构造、不外露）。
    pub fn forward(x: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        let mut mean = vec![0.0f32; C];
        let mut var = vec![1.0f32; C];
        norm().forward_train(x, gamma, beta, N, 1, S, &mut mean, &mut var, out)
    }

    /// 标量代价：两趟读 x（均值、方差）+ 读 gamma/beta + 写 y，每元素约 4 次运算。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        crate::model::cost_n(shape, "batch_norm_f32", 4.0, 3)
    }

    classic_op!(BatchNormModel, "batch_norm_f32", 3, (64, 64));
    #[annotation::model(name = "batch_norm_f32", shape = [64, 64], dtype = "f32", skeleton = "elementwise")]
    /// 语义别名（`#[annotation::model]` 在此生成 `MODEL` 自检常量）。
    pub type BatchNormF32 = BatchNormModel;
}

/// 组归一化（声明档 `G=2, C=4, spatial=2, n=2`）。
pub mod group_norm_f32 {

    use super::*;
    pub use crate::unary::{AXES, project_shape};

    const G: usize = 2;
    const C: usize = 4;
    const S: usize = 2;
    const N: usize = 2;

    fn norm() -> crate::norm::GroupNorm {
        crate::norm::GroupNorm {
            groups: G,
            channels: C,
            eps: 1e-5,
        }
    }

    /// 声明档样本：固定几何下的输入与输出缓冲。
    pub fn sample(_: usize, _: usize) -> Sample {
        Sample {
            m: N * S * C,
            n: N * S * C,
            scalars: vec![
                ("N", (N * S * C) as i64),
                ("G", G as i64),
                ("C", C as i64),
                ("S", S as i64),
            ],
            buffers: vec![
                x_buffer(N * S * C),
                Buffer {
                    name: "gamma",
                    values: Values::F32(vec![1.25; C]),
                },
                Buffer {
                    name: "beta",
                    values: Values::F32(vec![0.5; C]),
                },
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; N * S * C]),
                },
            ],
        }
    }

    /// 标量参考：复用 conv/norm 的真实现算一遍（真窗口 / 真统计）。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let gamma = sample.f32("gamma")?;
        let beta = sample.f32("beta")?;
        let mut out = vec![0.0f32; N * S * C];
        norm().forward(x, gamma, beta, N, S, &mut out)?;
        Ok(out)
    }

    /// 声明档前向：`G` 组、每组 `(C/G, S)` 统计。
    pub fn forward(x: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        norm().forward(x, gamma, beta, N, S, out)
    }

    /// 标量代价：两趟统计 + 一趟归一化，读 3 遍、写一遍。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        crate::model::cost_n(shape, "group_norm_f32", 4.0, 3)
    }

    classic_op!(GroupNormModel, "group_norm_f32", 3, (64, 64));
    #[annotation::model(name = "group_norm_f32", shape = [64, 64], dtype = "f32", skeleton = "elementwise")]
    /// 语义别名（`#[annotation::model]` 在此生成 `MODEL` 自检常量）。
    pub type GroupNormF32 = GroupNormModel;
}

/// 实例归一化（声明档 `C=4, spatial=2, n=2`）：等价于 `GroupNorm(G = C)`。
pub mod instance_norm_f32 {

    use super::*;
    pub use crate::unary::{AXES, project_shape};

    const C: usize = 4;
    const S: usize = 2;
    const N: usize = 2;

    fn norm() -> crate::norm::InstanceNorm {
        crate::norm::InstanceNorm {
            channels: C,
            eps: 1e-5,
        }
    }

    /// 声明档样本：固定几何下的输入与输出缓冲。
    pub fn sample(_: usize, _: usize) -> Sample {
        Sample {
            m: N * S * C,
            n: N * S * C,
            scalars: vec![("N", (N * S * C) as i64), ("C", C as i64), ("S", S as i64)],
            buffers: vec![
                x_buffer(N * S * C),
                Buffer {
                    name: "gamma",
                    values: Values::F32(vec![0.75; C]),
                },
                Buffer {
                    name: "beta",
                    values: Values::F32(vec![-0.25; C]),
                },
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; N * S * C]),
                },
            ],
        }
    }

    /// 标量参考：复用 conv/norm 的真实现算一遍（真窗口 / 真统计）。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let gamma = sample.f32("gamma")?;
        let beta = sample.f32("beta")?;
        let mut out = vec![0.0f32; N * S * C];
        norm().forward(x, gamma, beta, N, S, &mut out)?;
        Ok(out)
    }

    /// 声明档前向：`C` 个独立通道实例统计。
    pub fn forward(x: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        norm().forward(x, gamma, beta, N, S, out)
    }

    /// 标量代价：与组归一化同构（统计域更小、次数不变）。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        crate::model::cost_n(shape, "instance_norm_f32", 4.0, 3)
    }

    classic_op!(InstanceNormModel, "instance_norm_f32", 3, (64, 64));
    #[annotation::model(name = "instance_norm_f32", shape = [64, 64], dtype = "f32", skeleton = "elementwise")]
    /// 语义别名（`#[annotation::model]` 在此生成 `MODEL` 自检常量）。
    pub type InstanceNormF32 = InstanceNormModel;
}

/// 层归一化（声明档 `3 行 × 4 宽`）：逐行在最后一维统计。
pub mod layer_norm_f32 {

    use super::*;
    pub use crate::unary::{AXES, project_shape};

    const R: usize = 3;
    const W: usize = 4;

    fn norm() -> crate::norm::LayerNorm {
        crate::norm::LayerNorm {
            width: W,
            eps: 1e-5,
        }
    }

    /// 声明档样本：固定几何下的输入与输出缓冲。
    pub fn sample(_: usize, _: usize) -> Sample {
        Sample {
            m: R * W,
            n: R * W,
            scalars: vec![("N", (R * W) as i64), ("R", R as i64), ("W", W as i64)],
            buffers: vec![
                x_buffer(R * W),
                Buffer {
                    name: "gamma",
                    values: Values::F32(vec![1.0; W]),
                },
                Buffer {
                    name: "beta",
                    values: Values::F32(vec![0.0; W]),
                },
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; R * W]),
                },
            ],
        }
    }

    /// 标量参考：复用 conv/norm 的真实现算一遍（真窗口 / 真统计）。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let gamma = sample.f32("gamma")?;
        let beta = sample.f32("beta")?;
        let mut out = vec![0.0f32; R * W];
        norm().forward(x, gamma, beta, &mut out)?;
        Ok(out)
    }

    /// 声明档前向：`R` 行、每行 `W` 宽。
    pub fn forward(x: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        norm().forward(x, gamma, beta, out)
    }

    /// 标量代价：两趟行内统计 + 一趟归一化，读 3 遍、写一遍。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        crate::model::cost_n(shape, "layer_norm_f32", 4.0, 3)
    }

    classic_op!(LayerNormModel, "layer_norm_f32", 3, (64, 64));
    #[annotation::model(name = "layer_norm_f32", shape = [64, 64], dtype = "f32", skeleton = "elementwise")]
    /// 语义别名（`#[annotation::model]` 在此生成 `MODEL` 自检常量）。
    pub type LayerNormF32 = LayerNormModel;
}

#[cfg(test)]
mod tests {
    use super::{batch_norm_f32, layer_norm_f32, max_pool1d_f32};

    /// 每个算子的样本 -> 参考必须自洽（参考不能报错，长度与 `y` 缓冲一致）。
    fn check(name: &str) {
        let model = crate::model(name).expect("已注册");
        let sample = (model.sample)(model.shape.0, model.shape.1);
        let values = (model.reference)(&sample).expect("参考可计算");
        let y = sample.f32("y").expect("y 存在");
        assert_eq!(values.len(), y.len(), "{name}: 参考长度必须等于 y 缓冲");
    }

    #[test]
    fn every_classic_sample_matches_its_reference() {
        for name in [
            "max_pool1d_f32",
            "max_pool2d_f32",
            "avg_pool2d_f32",
            "adaptive_pool_f32",
            "interpolate_f32",
            "batch_norm_f32",
            "group_norm_f32",
            "instance_norm_f32",
            "layer_norm_f32",
        ] {
            check(name);
        }
    }

    #[test]
    fn norm_references_are_statistically_sane() {
        // LayerNorm 参考：每行均值 0、方差 ≈ 1（eps 让方差略小于 1）。
        let sample = layer_norm_f32::sample(1, 1);
        let y = layer_norm_f32::reference(&sample).expect("参考可计算");
        for row in 0..3 {
            let part = &y[row * 4..(row + 1) * 4];
            let mean: f32 = part.iter().sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-5, "行 {row} 均值 {mean}");
            let var: f32 = part.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
            assert!((var - 1.0).abs() < 1e-3, "行 {row} 方差 {var}");
        }
        // BatchNorm 参考：批内两样本 × 空间 2 × 通道 4，输出必须全部有限。
        let sample = batch_norm_f32::sample(1, 1);
        let y = batch_norm_f32::reference(&sample).expect("参考可计算");
        assert_eq!(y.len(), 16);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn pool_references_pick_exact_window_extremes() {
        let sample = max_pool1d_f32::sample(1, 1);
        let x = sample.f32("x").expect("x");
        let y = max_pool1d_f32::reference(&sample).expect("参考可计算");
        for (o, value) in y.iter().enumerate() {
            let want = x[2 * o].max(x[2 * o + 1]);
            assert_eq!(*value, want, "位置 {o}");
        }
    }
}
