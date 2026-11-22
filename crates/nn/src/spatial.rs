//! 经典几何层之一：卷积、转置卷积、池化与插值的契约登记。
//!
//! 与 `norm` 同属经典几何层族。拆成两个文件的判据不是行数，而是**共享的几何代数不同**：
//! 这一族里的窗口滑动口径是同一套 `ConvGeom`——卷积读它算输出长度，池化读它算第几段，
//! 归一化族根本不碰窗口。层只**声明**形状与算子归属，不执行、不持有权重。
//!
//! # 下界分两档，理由不同
//!
//! - **二维卷积**绑 `conv2d_f32`（真卷积算子：im2col + GEMM 的完整语义表），下界按它的每输出
//!   `2·k²·cin` 次运算与三读一写口径给出。
//! - **一维卷积与两种转置**仍绑 `gemm_f32`：`Conv1d` 的 1d im2col 就是矩阵乘、转置卷积没有独立
//!   算子，`gemm_f32` 的收缩轴是声明档 `K = 8`，因此这里是**松下界**（下界允许偏松，不会高于真实代价）。
//! - **池化/插值**各绑自己的真算子，代价表即算子的语义表。

use crate::ContractSpec as Spec;
use crate::conv::ConvGeom;
use crate::conv::deconv::{ConvTranspose1d as OpDeconv1, ConvTranspose2d as OpDeconv2};
use crate::conv::{
    Conv1d as OpConv1, Conv2d as OpConv2, MaxPool1d as OpPool1, MaxPool2d as OpPool2,
};
use crate::{ContractSpec, nonzero};
use diagnostic::Result;

/// 一维卷积与转置卷积绑定的算子：im2col 之后就是一次矩阵乘。
const GEMM: &str = "gemm_f32";

/// 二维卷积绑定的真算子（完整 im2col + GEMM 语义）。
const CONV: &str = "conv2d_f32";

/// 校验反卷积层的输入/输出通道非零：1d/2d 共用。
fn nonzero_channels(cin: usize, cout: usize, layer: &str) -> Result<()> {
    nonzero(cin, "cin", layer)?;
    nonzero(cout, "cout", layer)?;
    Ok(())
}

/// 一维卷积层：NHWC 输入 `[l, cin]`，输出 `[ol, cout]`。
#[crate::spec(name = $name, out = [$ol, $cout], ops = [$op])]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conv1d {
    cin: usize,
    cout: usize,
    geom: ConvGeom,
    ol: usize,
}

impl Conv1d {
    /// 构造：按输入长度与几何算出输出长度。
    ///
    /// # Errors
    ///
    /// 通道/几何为零，或输入短于窗口时返回说明。
    pub fn new(
        name: impl Into<String>,
        cin: usize,
        cout: usize,
        geom: ConvGeom,
        l: usize,
    ) -> Result<Self> {
        let layer = "nn::Conv1d::new";
        nonzero_channels(cin, cout, layer)?;
        let kernel = OpConv1::new(cin, cout, geom.kw, geom.sw, geom.pw);
        let ol = kernel.len(l)?;
        let spec = __layer_spec_Conv1d!(name, ol, cout, GEMM)?;
        Ok(Self {
            spec,
            cin,
            cout,
            geom,
            ol,
        })
    }

    crate::accessors! {
        /// 输入通道。
        cin -> usize = cin,
        /// 输出通道。
        cout -> usize = cout,
        /// 卷积几何。
        geom -> ConvGeom = geom,
        /// 输出长度。
        #[allow(clippy::len_without_is_empty)]
        len -> usize = ol,
    }
}

crate::priced!(Conv1d, GEMM);

/// 二维卷积层：NHWC 输入 `[h, w, cin]`，输出 `[oh, ow, cout]`。
#[crate::spec(name = $name, out = [$oh, $ow, $cout], ops = [$op])]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conv2d {
    cin: usize,
    cout: usize,
    geom: ConvGeom,
    oh: usize,
    ow: usize,
}

impl Conv2d {
    /// 构造：按输入边长与几何算出输出边长。
    ///
    /// # Errors
    ///
    /// 通道/几何为零，或输入短于窗口时返回说明。
    pub fn new(
        name: impl Into<String>,
        cin: usize,
        cout: usize,
        geom: ConvGeom,
        h: usize,
        w: usize,
    ) -> Result<Self> {
        let layer = "nn::Conv2d::new";
        nonzero_channels(cin, cout, layer)?;
        let kernel = OpConv2::new(cin, cout, geom);
        let (oh, ow) = kernel.shape(h, w)?;
        let spec = __layer_spec_Conv2d!(name, oh, ow, cout, CONV)?;
        Ok(Self {
            spec,
            cin,
            cout,
            geom,
            oh,
            ow,
        })
    }

    crate::accessors! {
        /// 输入通道。
        cin -> usize = cin,
        /// 输出通道。
        cout -> usize = cout,
        /// 卷积几何。
        geom -> ConvGeom = geom,
    }

    /// 输出边长 `(oh, ow)`。
    pub fn shape(&self) -> (usize, usize) {
        (self.oh, self.ow)
    }
}

crate::priced!(Conv2d, CONV);

/// 一维反卷积层：`[l, cin]` → `[ol, cout]`，权重 `[cin, cout, k]`。
#[crate::spec(name = $name, out = [$ol, $cout], ops = [$op])]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvTranspose1d {
    deconv: OpDeconv1,
    ol: usize,
}

impl ConvTranspose1d {
    /// 构造：按输入长度与 `(k, s, p)` 算出输出长度。
    ///
    /// # Errors
    ///
    /// 通道/几何为零，或输出长度非正时返回说明。
    pub fn new(
        name: impl Into<String>,
        cin: usize,
        cout: usize,
        k: usize,
        s: usize,
        p: usize,
        l: usize,
    ) -> Result<Self> {
        let layer = "nn::ConvTranspose1d::new";
        nonzero_channels(cin, cout, layer)?;
        let deconv = OpDeconv1::new(cin, cout, k, s, p);
        let ol = deconv.len(l)?;
        let spec = __layer_spec_ConvTranspose1d!(name, ol, cout, GEMM)?;
        Ok(Self { spec, deconv, ol })
    }

    /// 输出长度。
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.ol
    }

    /// 输出通道。
    pub fn cout(&self) -> usize {
        self.deconv.cout
    }
}

crate::priced!(ConvTranspose1d, GEMM);

/// 二维反卷积层：`[h, w, cin]` → `[oh, ow, cout]`，权重 `[cin, cout, kh, kw]`。
#[crate::spec(name = $name, out = [$oh, $ow, $cout], ops = [$op])]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvTranspose2d {
    deconv: OpDeconv2,
    oh: usize,
    ow: usize,
}

impl ConvTranspose2d {
    /// 构造：按输入边长与几何算出输出边长。
    ///
    /// # Errors
    ///
    /// 通道/几何为零，或输出边长非正时返回说明。
    pub fn new(
        name: impl Into<String>,
        cin: usize,
        cout: usize,
        geom: ConvGeom,
        h: usize,
        w: usize,
    ) -> Result<Self> {
        let layer = "nn::ConvTranspose2d::new";
        nonzero_channels(cin, cout, layer)?;
        let deconv = OpDeconv2::new(cin, cout, geom);
        let oh = deconv.oh(h)?;
        let ow = deconv.ow(w)?;
        let spec = __layer_spec_ConvTranspose2d!(name, oh, ow, cout, GEMM)?;
        Ok(Self {
            spec,
            deconv,
            oh,
            ow,
        })
    }

    /// 输出边长 `(oh, ow)`。
    pub fn shape(&self) -> (usize, usize) {
        (self.oh, self.ow)
    }

    /// 输出通道。
    pub fn cout(&self) -> usize {
        self.deconv.cout
    }
}

crate::priced!(ConvTranspose2d, GEMM);

/// 一维最大池化层：`[l, c]` → `[ol, c]`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaxPool1d {
    spec: ContractSpec,
    ol: usize,
    c: usize,
}

impl MaxPool1d {
    /// 构造。
    ///
    /// # Errors
    ///
    /// 窗口/步长为零，或输入短于窗口时返回说明。
    pub fn new(
        name: impl Into<String>,
        k: usize,
        s: usize,
        p: usize,
        l: usize,
        c: usize,
    ) -> Result<Self> {
        let layer = "nn::MaxPool1d::new";
        nonzero(c, "channels", layer)?;
        // 算子结构体只在构造时用来算输出长度，之后没有访问器能读到它——存着就是一份
        // 会被拷贝、会被比较、却永远不参与判断的状态。
        let ol = OpPool1 { k, s, p }.len(l)?;
        let spec = Spec::new(name, crate::Shape::new([ol, c]), ["max_pool1d_f32"])?;
        Ok(Self { spec, ol, c })
    }

    crate::accessors! {
        /// 输出长度。
        #[allow(clippy::len_without_is_empty)]
        len -> usize = ol,
        /// 通道数。
        channels -> usize = c,
    }
}

crate::priced!(MaxPool1d, "max_pool1d_f32");

/// `[h, w, c] → [oh, ow, c]` 这一族的共用骨架。
///
/// 二维池化（最大/平均/自适应）与插值**只差一件事**：绑哪个算子。输出边长的算法属于算子自己，
/// 因此各自手写构造；而"存什么、怎么读、下界按什么算"三件事一模一样，只剩这一份。
///
/// 算子结构体不再作为字段保存：它只在构造时用来算输出边长，之后没有任何访问器能读到它，
/// 留着就是一份会被拷贝、会被比较、却永远不参与判断的状态。
macro_rules! planar {
    ($(#[$doc:meta])* $ty:ident, $op:literal) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $ty {
            spec: Spec,
            oh: usize,
            ow: usize,
            c: usize,
        }

        impl $ty {
            /// 输出边长。
            pub fn shape(&self) -> (usize, usize) {
                (self.oh, self.ow)
            }

            /// 通道数。
            pub fn channels(&self) -> usize {
                self.c
            }

            /// 四个构造函数的尾巴都是这一句：边长已由算子算好，这里只登记规格。
            fn plan(name: impl Into<String>, oh: usize, ow: usize, c: usize) -> Result<Self> {
                let spec = Spec::new(name, crate::Shape::new([oh, ow, c]), [$op])?;
                Ok(Self {
                    spec,
                    oh,
                    ow,
                    c,
                })
            }
        }

        crate::priced!($ty, $op);
    };
}

planar!(
    /// 二维最大池化层：`[h, w, c]` → `[oh, ow, c]`。规格构造口径同 [`MaxPool1d`]。
    MaxPool2d,
    "max_pool2d_f32"
);

impl MaxPool2d {
    /// 构造。
    ///
    /// # Errors
    ///
    /// 窗口/步长为零，或输入短于窗口时返回说明。
    pub fn new(
        name: impl Into<String>,
        geom: ConvGeom,
        h: usize,
        w: usize,
        c: usize,
    ) -> Result<Self> {
        let layer = "nn::MaxPool2d::new";
        nonzero(c, "channels", layer)?;
        let (oh, ow) = OpPool2 { geom }.shape(h, w)?;
        Self::plan(name, oh, ow, c)
    }
}

planar!(
    /// 二维平均池化层：口径同 [`MaxPool2d`]，`count_include_pad` 决定补零是否计入分母。
    AvgPool2d,
    "avg_pool2d_f32"
);

impl AvgPool2d {
    /// 构造。
    ///
    /// # Errors
    ///
    /// 窗口/步长为零，或输入短于窗口时返回说明。
    pub fn new(
        name: impl Into<String>,
        geom: ConvGeom,
        count_include_pad: bool,
        h: usize,
        w: usize,
        c: usize,
    ) -> Result<Self> {
        let layer = "nn::AvgPool2d::new";
        nonzero(c, "channels", layer)?;
        let (oh, ow) = crate::conv::AvgPool2d {
            geom,
            count_include_pad,
        }
        .shape(h, w)?;
        Self::plan(name, oh, ow, c)
    }
}

planar!(
    /// 自适应平均池化层：任意输入压到固定 `[oh, ow, c]`。规格构造口径同 [`MaxPool1d`]。
    AdaptiveAvgPool2d,
    "adaptive_pool_f32"
);

impl AdaptiveAvgPool2d {
    /// 构造。
    ///
    /// # Errors
    ///
    /// 目标尺寸或通道为零时返回说明。
    pub fn new(name: impl Into<String>, oh: usize, ow: usize, c: usize) -> Result<Self> {
        let layer = "nn::AdaptiveAvgPool2d::new";
        nonzero(oh, "oh", layer)?;
        nonzero(ow, "ow", layer)?;
        nonzero(c, "channels", layer)?;
        Self::plan(name, oh, ow, c)
    }
}

/// 插值层：`[h, w, c]` → `[oh, ow, c]`，取样方式见 [`crate::conv::Sampler`]。
///
/// 它比 `planar!` 那一族多一件**真状态**：`sampler` 不改输出边长，却决定算子怎么取样。
/// 因此不塞进宏，而是显式带上——并且必须能读回（`sampler()`），否则它又会退化成死字段。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interpolate {
    spec: Spec,
    sampler: crate::conv::Sampler,
    oh: usize,
    ow: usize,
    c: usize,
}

impl Interpolate {
    /// 构造。
    ///
    /// # Errors
    ///
    /// 目标尺寸或通道为零时返回说明。
    pub fn new(
        name: impl Into<String>,
        sampler: crate::conv::Sampler,
        oh: usize,
        ow: usize,
        c: usize,
    ) -> Result<Self> {
        let layer = "nn::Interpolate::new";
        nonzero(oh, "oh", layer)?;
        nonzero(ow, "ow", layer)?;
        nonzero(c, "channels", layer)?;
        let spec = Spec::new(name, crate::Shape::new([oh, ow, c]), ["interpolate_f32"])?;
        Ok(Self {
            spec,
            sampler,
            oh,
            ow,
            c,
        })
    }

    /// 输出边长。
    pub fn shape(&self) -> (usize, usize) {
        (self.oh, self.ow)
    }

    /// 通道数。
    pub fn channels(&self) -> usize {
        self.c
    }

    /// 取样方式。
    pub fn sampler(&self) -> crate::conv::Sampler {
        self.sampler
    }
}

crate::priced!(Interpolate, "interpolate_f32");

#[cfg(test)]
#[path = "../tests/classic_tests.rs"]
mod tests;
