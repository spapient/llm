//! `classic` 的卷积层族：Conv1d/Conv2d 与转置卷积。

use super::*;

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
