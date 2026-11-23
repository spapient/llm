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
