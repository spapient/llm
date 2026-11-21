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
        llm_gemm::norm::BatchNorm2d {
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
        llm_gemm::norm::GroupNorm {
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
        llm_gemm::norm::InstanceNorm {
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
        llm_gemm::norm::LayerNorm {
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
