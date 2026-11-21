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
