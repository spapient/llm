use super::*;

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
