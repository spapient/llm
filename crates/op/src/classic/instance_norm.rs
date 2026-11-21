use super::*;

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
