//! 窗口族：最大池化（1d/2d）、平均池化与自适应池化，共用 [`crate::classic`] 的 `window_op!` 装配。

/// 自适应平均池化（声明档 4×4 → 2×2）。
pub mod adaptive {
    use super::super::*;

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
}

/// 二维平均池化（声明档几何 `3×3, s=1, p=1`，补零不计入分母）。
pub mod avg2d {
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
}

/// 一维最大池化（声明档几何 `k=2, s=2, p=0`，单通道）。
pub mod max1d {
    use super::super::*;

    /// 声明档输出位置数。
    pub const OL: usize = 8;
    /// 声明档输入长度：`(OL − 1)·S + K = 16`。
    pub const L: usize = 16;

    /// 声明档算术：`<S: Scalar>` 泛型，四种精度共用一条取窗口最大的路径。
    pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
        crate::conv::MaxPool1d { k: 2, s: 2, p: 0 }.forward(x, 1, L, 1, out)
    }

    // 每输出读两个输入元素、写一个；读两遍字节、写一遍。
    window_op!(
        MaxPool1dModel,
        MaxPool1d,
        "max_pool1d_f32",
        64,
        64,
        L,
        OL,
        2.0,
        2
    );
}

/// 二维最大池化（声明档几何 `3×3, s=1, p=1`，4×4 输入、same 输出）。
pub mod max2d {
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
}
