//! `classic` 的池化与插值层族：Max/Avg/Adaptive/Interpolate。

use super::*;

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
