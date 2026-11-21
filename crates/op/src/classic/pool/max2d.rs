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
