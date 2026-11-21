//! 经典几何层：卷积、反卷积、池化、插值与归一化的契约登记。
//!
//! 这一层只**声明**形状与算子归属，不执行、不持有权重：层的价值是让装配侧与 `Stack::floor`
//! 看见"模型里有一层 3×3 卷积"这件事，而不是再写一份前向。执行仍在 `op::conv` / `op::norm`。
//!
//! # 下界分两档，理由不同
//!
//! - **二维卷积**绑 `conv2d_f32`（真卷积算子：im2col + GEMM 的完整语义表），下界按它的每输出
//!   `2·k²·cin` 次运算与三读一写口径给出。
//! - **一维卷积与两种转置**仍绑 `gemm_f32`：`Conv1d` 的 1d im2col 就是矩阵乘、转置卷积没有独立
//!   算子，`gemm_f32` 的收缩轴是声明档 `K = 8`，因此这里是**松下界**（下界允许偏松，不会高于真实代价）。
//! - **窗口/归一化族**各自绑定自己的真算子（`max_pool1d/max_pool2d/avg_pool2d/adaptive_pool/
//!   interpolate/batch_norm/group_norm/instance_norm/layer_norm`），代价表即算子的语义表。

use diagnostic::{Code, Error, Result};
use op::conv::ConvGeom;
use op::conv::{
    AdaptiveAvgPool2d as OpAdaptive, Conv1d as OpConv1, Conv2d as OpConv2, Interpolate as OpInterp,
    MaxPool1d as OpPool1, MaxPool2d as OpPool2,
};
use op::deconv::{ConvTranspose1d as OpDeconv1, ConvTranspose2d as OpDeconv2};
// 归一化层现在是 `Scalar` 泛型；经典层登记只针对 `f32` 声明档，这里给出一份具体别名。
type OpBatch = op::norm::BatchNorm2d<f32>;
type OpGroup = op::norm::GroupNorm<f32>;
type OpInst = op::norm::InstanceNorm<f32>;
type OpLayer = op::norm::LayerNorm<f32>;
use tensor::Shape;

use crate::ContractSpec as Spec;
use crate::{Bounds, BoundsResult, Contract, ContractSpec, ParamRole, spec_accessors};

/// 一维卷积与转置卷积绑定的算子：im2col 之后就是一次矩阵乘。
const GEMM: &str = "gemm_f32";

/// 二维卷积绑定的真算子（完整 im2col + GEMM 语义）。
const CONV: &str = "conv2d_f32";

/// 校验非零并在失败时点名层与字段。
fn nonzero(value: usize, field: &str, layer: &str) -> Result<usize> {
    if value == 0 {
        return Err(
            Error::new(Code::Invalid, format!("{layer}: `{field}` 必须非零")).with("nn::classic"),
        );
    }
    Ok(value)
}

/// 校验 `eps` 为正且有限：所有归一化层共用。
fn check_eps(eps: f32, layer: &str) -> Result<()> {
    if eps <= 0.0 || eps.is_nan() {
        return Err(Error::new(Code::Invalid, format!("{layer}: `eps` 必须为正")).with(layer));
    }
    Ok(())
}

/// 校验组归一化的结构：组数/通道非零且通道能被组数整除。
fn check_group_norm(groups: usize, channels: usize, layer: &str) -> Result<()> {
    nonzero(groups, "groups", layer)?;
    nonzero(channels, "channels", layer)?;
    if !channels.is_multiple_of(groups) {
        return Err(Error::new(
            Code::Invalid,
            format!("{layer}: `channels`({channels}) 不能被 `groups`({groups}) 整除"),
        )
        .with(layer));
    }
    Ok(())
}

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
        nonzero(cin, "cin", layer)?;
        nonzero(cout, "cout", layer)?;
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

    /// 输入通道。
    pub fn cin(&self) -> usize {
        self.cin
    }

    /// 输出通道。
    pub fn cout(&self) -> usize {
        self.cout
    }

    /// 卷积几何。
    pub fn geom(&self) -> ConvGeom {
        self.geom
    }

    /// 输出长度。
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.ol
    }
}

impl Contract for Conv1d {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_context(GEMM, self.ol, self.cout).map(Some)
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
        nonzero(cin, "cin", layer)?;
        nonzero(cout, "cout", layer)?;
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

    /// 输入通道。
    pub fn cin(&self) -> usize {
        self.cin
    }

    /// 输出通道。
    pub fn cout(&self) -> usize {
        self.cout
    }

    /// 卷积几何。
    pub fn geom(&self) -> ConvGeom {
        self.geom
    }

    /// 输出边长 `(oh, ow)`。
    pub fn shape(&self) -> (usize, usize) {
        (self.oh, self.ow)
    }
}

impl Contract for Conv2d {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_context(CONV, self.oh * self.ow, self.cout).map(Some)
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
        Bounds::of_context(GEMM, self.ol, self.deconv.cout).map(Some)
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
        Bounds::of_context(GEMM, self.oh * self.ow, self.deconv.cout).map(Some)
    }
}

/// 一维最大池化层：`[l, c]` → `[ol, c]`。
///
/// 规格走 [`ContractSpec::new`] 直接构造而不是 `#[crate::spec]`：池化没有可代理的已注册算子，
/// 注解层"一层至少一个算子"的硬约束在这里会把一个语义不符的名字塞进来，所以选择显式登记为
/// "0 算子 + 无法计价"，而不是凑数。
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
        let spec = Spec::new(
            name,
            crate::Shape::new([ol, c]),
            vec!["max_pool1d_f32".to_string()],
        )?;
        Ok(Self { spec, pool, ol, c })
    }

    /// 输出长度。
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.ol
    }

    /// 通道数。
    pub fn channels(&self) -> usize {
        self.c
    }
}

impl Contract for MaxPool1d {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_context("max_pool1d_f32", self.ol, self.c).map(Some)
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
        let spec = Spec::new(
            name,
            crate::Shape::new([oh, ow, c]),
            vec!["max_pool2d_f32".to_string()],
        )?;
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
        Bounds::of_context("max_pool2d_f32", self.oh * self.ow, self.c).map(Some)
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
        let spec = Spec::new(
            name,
            crate::Shape::new([oh, ow, c]),
            vec!["avg_pool2d_f32".to_string()],
        )?;
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
        Bounds::of_context("avg_pool2d_f32", self.oh * self.ow, self.c).map(Some)
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
        let spec = Spec::new(
            name,
            crate::Shape::new([oh, ow, c]),
            vec!["adaptive_pool_f32".to_string()],
        )?;
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
        Bounds::of_context("adaptive_pool_f32", self.pool.oh * self.pool.ow, self.c).map(Some)
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
        let spec = Spec::new(
            name,
            crate::Shape::new([oh, ow, c]),
            vec!["interpolate_f32".to_string()],
        )?;
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
        Bounds::of_context("interpolate_f32", self.interp.oh * self.interp.ow, self.c).map(Some)
    }
}

/// 批归一化层（NHWC，最后维 `channels`）。
///
/// 规格构造口径同 [`MaxPool1d`]：归一化没有可代理的已注册算子（RMSNorm 不含减均值，语义不同），
/// 因此登记为 0 算子层。
#[derive(Debug, Clone, PartialEq)]
pub struct BatchNorm2d {
    spec: ContractSpec,
    norm: OpBatch,
    channels: usize,
}

impl BatchNorm2d {
    /// 构造：`eps` 与 `momentum` 由调用方给。
    ///
    /// # Errors
    ///
    /// 通道为零、`eps ≤ 0` 或 `momentum` 越界时返回说明。
    pub fn new(name: impl Into<String>, channels: usize, eps: f32, momentum: f32) -> Result<Self> {
        let layer = "nn::BatchNorm2d::new";
        nonzero(channels, "channels", layer)?;
        check_eps(eps, layer)?;
        if !(0.0..=1.0).contains(&momentum) {
            return Err(Error::new(
                Code::Invalid,
                format!("{layer}: `momentum` 必须落在 [0, 1]"),
            )
            .with(layer));
        }
        let norm = OpBatch {
            channels,
            eps,
            momentum,
        };
        let spec = Spec::new(
            name,
            crate::Shape::new([channels]),
            vec!["batch_norm_f32".to_string()],
        )?;
        Ok(Self {
            spec,
            norm,
            channels,
        })
    }

    /// 通道数。
    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl Contract for BatchNorm2d {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_context("batch_norm_f32", self.channels, 1).map(Some)
    }
}

/// 组归一化层：`(n, group)` 内按 `(C/G, spatial)` 统计。规格构造口径同 [`BatchNorm2d`]。
#[derive(Debug, Clone, PartialEq)]
pub struct GroupNorm {
    spec: ContractSpec,
    norm: OpGroup,
    channels: usize,
}

impl GroupNorm {
    /// 构造：`channels` 必须能被 `groups` 整除。
    ///
    /// # Errors
    ///
    /// 组数/通道为零、不能整除或 `eps ≤ 0` 时返回说明。
    pub fn new(name: impl Into<String>, groups: usize, channels: usize, eps: f32) -> Result<Self> {
        let layer = "nn::GroupNorm::new";
        check_group_norm(groups, channels, layer)?;
        check_eps(eps, layer)?;
        let norm = OpGroup {
            groups,
            channels,
            eps,
        };
        let spec = Spec::new(
            name,
            crate::Shape::new([channels]),
            vec!["group_norm_f32".to_string()],
        )?;
        Ok(Self {
            spec,
            norm,
            channels,
        })
    }

    /// 通道数。
    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl Contract for GroupNorm {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_context("group_norm_f32", self.channels, 1).map(Some)
    }
}

/// 实例归一化层：等价于 `GroupNorm(groups = channels)`。规格构造口径同 [`BatchNorm2d`]。
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceNorm {
    spec: ContractSpec,
    norm: OpInst,
    channels: usize,
}

impl InstanceNorm {
    /// 构造。
    ///
    /// # Errors
    ///
    /// 通道为零或 `eps ≤ 0` 时返回说明。
    pub fn new(name: impl Into<String>, channels: usize, eps: f32) -> Result<Self> {
        let layer = "nn::InstanceNorm::new";
        nonzero(channels, "channels", layer)?;
        check_eps(eps, layer)?;
        let norm = OpInst { channels, eps };
        let spec = Spec::new(
            name,
            crate::Shape::new([channels]),
            vec!["instance_norm_f32".to_string()],
        )?;
        Ok(Self {
            spec,
            norm,
            channels,
        })
    }

    /// 通道数。
    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl Contract for InstanceNorm {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_context("instance_norm_f32", self.channels, 1).map(Some)
    }
}

/// 层归一化层：逐行在最后一维 `width` 上统计。规格构造口径同 [`BatchNorm2d`]。
#[derive(Debug, Clone, PartialEq)]
pub struct LayerNorm {
    spec: ContractSpec,
    norm: OpLayer,
    width: usize,
}

impl LayerNorm {
    /// 构造。
    ///
    /// # Errors
    ///
    /// 宽度为零或 `eps ≤ 0` 时返回说明。
    pub fn new(name: impl Into<String>, width: usize, eps: f32) -> Result<Self> {
        let layer = "nn::LayerNorm::new";
        nonzero(width, "width", layer)?;
        check_eps(eps, layer)?;
        let norm = OpLayer { width, eps };
        let spec = Spec::new(
            name,
            crate::Shape::new([width]),
            vec!["layer_norm_f32".to_string()],
        )?;
        Ok(Self { spec, norm, width })
    }

    /// 归一化宽度。
    pub fn width(&self) -> usize {
        self.width
    }
}

impl Contract for LayerNorm {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        Bounds::of_context("layer_norm_f32", self.width, 1).map(Some)
    }
}

/// 输出形状的静态检查：每层的 `out` 维数必须与 `Shape::new` 接受的一致。
const _: () = {
    // 只是让 `Shape` 与 `ParamRole` 保持被引用，避免导出面漂移时静默。
    let _ = core::mem::size_of::<Shape>();
    let _ = ParamRole::Trainable;
};

#[cfg(test)]
mod tests {
    use super::*;

    fn geom3() -> ConvGeom {
        ConvGeom {
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            ph: 1,
            pw: 1,
        }
    }

    #[test]
    fn conv2d_registers_its_shape_and_conv_operator() {
        let layer = Conv2d::new("stem.conv", 3, 64, geom3(), 32, 32).expect("合法");
        assert_eq!(layer.name(), "stem.conv");
        assert_eq!(layer.output().dims(), [32, 32, 64]);
        assert_eq!(layer.spca().ops(), [CONV]);
        assert_eq!(layer.shape(), (32, 32));
        // 下界来自 conv2d_f32 的语义表（每输出 2·k²·cin、三读一写），因此有正 FLOPs 与正字节。
        let floor = layer.bounds().expect("合法").expect("有算子");
        assert!(floor.flops() > 0.0 && floor.bytes() > 0);
    }

    #[test]
    fn conv2d_rejects_zero_channels() {
        assert!(Conv2d::new("x", 0, 8, geom3(), 8, 8).is_err());
    }

    #[test]
    fn conv1d_and_transpose_lengths_follow_the_geometry() {
        // 一维卷积只吃 `kw/sw/pw`：核 2、步长 1、无补零，长度 9 → 8。
        let layer = Conv1d::new(
            "a",
            4,
            8,
            ConvGeom {
                kh: 1,
                kw: 2,
                sh: 1,
                sw: 1,
                ph: 0,
                pw: 0,
            },
            9,
        )
        .expect("合法");
        assert_eq!(layer.len(), 8);
        assert_eq!(layer.output().dims(), [8, 8]);

        let up = ConvTranspose1d::new("up", 4, 6, 2, 2, 0, 9).expect("合法");
        assert_eq!(up.len(), 18);
        assert_eq!(up.output().dims(), [18, 6]);

        let up2 = ConvTranspose2d::new("up2", 4, 6, geom3(), 8, 8).expect("合法");
        assert_eq!(up2.shape(), (8, 8));
        assert_eq!(up2.output().dims(), [8, 8, 6]);
    }

    #[test]
    fn pool_and_interpolate_layers_bind_one_real_operator() {
        let pool = MaxPool1d::new("p1", 2, 2, 0, 8, 4).expect("合法");
        assert_eq!(pool.output().dims(), [4, 4]);
        assert_eq!(pool.spca().ops(), ["max_pool1d_f32"]);
        // 有真算子就能推出下界：读一遍输入、写一遍输出。
        let lower = pool.bounds().expect("合法").expect("可计价");
        assert!(lower.read() > 0 && lower.write() > 0 && lower.flops() > 0.0);

        let pool2 = MaxPool2d::new("p2", geom3(), 8, 8, 4).expect("合法");
        assert_eq!(pool2.output().dims(), [8, 8, 4]);
        assert_eq!(pool2.spca().ops(), ["max_pool2d_f32"]);

        let avg = AvgPool2d::new("p3", geom3(), false, 8, 8, 4).expect("合法");
        assert_eq!(avg.output().dims(), [8, 8, 4]);
        assert_eq!(avg.spca().ops(), ["avg_pool2d_f32"]);

        let adp = AdaptiveAvgPool2d::new("p4", 2, 2, 4).expect("合法");
        assert_eq!(adp.output().dims(), [2, 2, 4]);
        assert_eq!(adp.spca().ops(), ["adaptive_pool_f32"]);

        let up = Interpolate::new("i", op::conv::Sampler::Linear, 16, 16, 4).expect("合法");
        assert_eq!(up.output().dims(), [16, 16, 4]);
        assert_eq!(up.spca().ops(), ["interpolate_f32"]);
    }

    #[test]
    fn norm_layers_check_their_geometry() {
        let bn = BatchNorm2d::new("bn", 64, 1e-5, 0.1).expect("合法");
        assert_eq!(bn.output().dims(), [64]);
        assert_eq!(bn.spca().ops(), ["batch_norm_f32"]);
        assert!(bn.bounds().expect("合法").is_some());
        assert!(BatchNorm2d::new("bn", 64, 0.0, 0.1).is_err());
        assert!(BatchNorm2d::new("bn", 64, 1e-5, 1.5).is_err());

        let gn = GroupNorm::new("gn", 8, 64, 1e-5).expect("合法");
        assert_eq!(gn.output().dims(), [64]);
        assert!(GroupNorm::new("gn", 8, 60, 1e-5).is_err());

        let ins = InstanceNorm::new("in", 32, 1e-5).expect("合法");
        assert_eq!(ins.output().dims(), [32]);

        let ln = LayerNorm::new("ln", 512, 1e-5).expect("合法");
        assert_eq!(ln.output().dims(), [512]);
        assert!(LayerNorm::new("ln", 512, -1.0).is_err());
    }
}
