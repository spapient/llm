use super::*;

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
