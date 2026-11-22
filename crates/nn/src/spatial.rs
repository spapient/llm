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
use crate::{Bounds, BoundsResult, Contract, ContractSpec, nonzero, spec_accessors};
use diagnostic::Result;
use op::conv::ConvGeom;
use op::conv::{
    AdaptiveAvgPool2d as OpAdaptive, Conv1d as OpConv1, Conv2d as OpConv2, Interpolate as OpInterp,
    MaxPool1d as OpPool1, MaxPool2d as OpPool2,
};
use op::deconv::{ConvTranspose1d as OpDeconv1, ConvTranspose2d as OpDeconv2};

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

impl Contract for Conv1d {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_ctx(GEMM, self.ol, self.cout).map(Some)
    }
}

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

impl Contract for Conv2d {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_ctx(CONV, self.oh * self.ow, self.cout).map(Some)
    }
}

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

impl Contract for ConvTranspose1d {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_ctx(GEMM, self.ol, self.deconv.cout).map(Some)
    }
}

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

impl Contract for ConvTranspose2d {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_ctx(GEMM, self.oh * self.ow, self.deconv.cout).map(Some)
    }
}

/// 一维最大池化层：`[l, c]` → `[ol, c]`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaxPool1d {
    spec: ContractSpec,
    pool: OpPool1,
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
        let pool = OpPool1 { k, s, p };
        let ol = pool.len(l)?;
        let spec = Spec::new(name, crate::Shape::new([ol, c]), ["max_pool1d_f32"])?;
        Ok(Self { spec, pool, ol, c })
    }

    crate::accessors! {
        /// 输出长度。
        #[allow(clippy::len_without_is_empty)]
        len -> usize = ol,
        /// 通道数。
        channels -> usize = c,
    }
}

impl Contract for MaxPool1d {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_ctx("max_pool1d_f32", self.ol, self.c).map(Some)
    }
}

/// 二维最大池化层：`[h, w, c]` → `[oh, ow, c]`。规格构造口径同 [`MaxPool1d`]。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaxPool2d {
    spec: ContractSpec,
    pool: OpPool2,
    oh: usize,
    ow: usize,
    c: usize,
}

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
        let pool = OpPool2 { geom };
        let (oh, ow) = pool.shape(h, w)?;
        let spec = Spec::new(name, crate::Shape::new([oh, ow, c]), ["max_pool2d_f32"])?;
        Ok(Self {
            spec,
            pool,
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
}

impl Contract for MaxPool2d {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_ctx("max_pool2d_f32", self.oh * self.ow, self.c).map(Some)
    }
}

/// 二维平均池化层：口径同 [`MaxPool2d`]，`count_include_pad` 决定补零是否计入分母。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvgPool2d {
    spec: ContractSpec,
    oh: usize,
    ow: usize,
    c: usize,
}

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
        let pool = op::conv::AvgPool2d {
            geom,
            count_include_pad,
        };
        let (oh, ow) = pool.shape(h, w)?;
        let spec = Spec::new(name, crate::Shape::new([oh, ow, c]), ["avg_pool2d_f32"])?;
        Ok(Self { spec, oh, ow, c })
    }

    /// 输出边长。
    pub fn shape(&self) -> (usize, usize) {
        (self.oh, self.ow)
    }

    /// 通道数。
    pub fn channels(&self) -> usize {
        self.c
    }
}

impl Contract for AvgPool2d {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_ctx("avg_pool2d_f32", self.oh * self.ow, self.c).map(Some)
    }
}

/// 自适应平均池化层：任意输入压到固定 `[oh, ow, c]`。规格构造口径同 [`MaxPool1d`]。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveAvgPool2d {
    spec: ContractSpec,
    pool: OpAdaptive,
    c: usize,
}

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
        let pool = OpAdaptive { oh, ow };
        let spec = Spec::new(name, crate::Shape::new([oh, ow, c]), ["adaptive_pool_f32"])?;
        Ok(Self { spec, pool, c })
    }

    /// 输出边长。
    pub fn shape(&self) -> (usize, usize) {
        (self.pool.oh, self.pool.ow)
    }

    /// 通道数。
    pub fn channels(&self) -> usize {
        self.c
    }
}

impl Contract for AdaptiveAvgPool2d {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_ctx("adaptive_pool_f32", self.pool.oh * self.pool.ow, self.c).map(Some)
    }
}

/// 插值层：`[h, w, c]` → `[oh, ow, c]`，取样方式见 [`op::conv::Sampler`]。规格构造口径同 [`MaxPool1d`]。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interpolate {
    spec: ContractSpec,
    interp: OpInterp,
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
        sampler: op::conv::Sampler,
        oh: usize,
        ow: usize,
        c: usize,
    ) -> Result<Self> {
        let layer = "nn::Interpolate::new";
        nonzero(oh, "oh", layer)?;
        nonzero(ow, "ow", layer)?;
        nonzero(c, "channels", layer)?;
        let interp = OpInterp { sampler, oh, ow };
        let spec = Spec::new(name, crate::Shape::new([oh, ow, c]), ["interpolate_f32"])?;
        Ok(Self { spec, interp, c })
    }

    /// 输出边长。
    pub fn shape(&self) -> (usize, usize) {
        (self.interp.oh, self.interp.ow)
    }

    /// 通道数。
    pub fn channels(&self) -> usize {
        self.c
    }
}

impl Contract for Interpolate {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_ctx("interpolate_f32", self.interp.oh * self.interp.ow, self.c).map(Some)
    }
}

#[cfg(test)]
#[path = "../tests/classic_tests.rs"]
mod tests;
