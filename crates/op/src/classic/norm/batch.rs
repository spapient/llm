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
