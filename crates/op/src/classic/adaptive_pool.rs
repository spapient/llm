use super::*;

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
