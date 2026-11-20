//! 经典几何层：卷积、反卷积、池化、插值与归一化的契约登记。
//!
//! 这一层只**声明**形状与算子归属，不执行、不持有权重：层的价值是让装配侧与 `Stack::floor`
//! 看见"模型里有一层 3×3 卷积"这件事，而不是再写一份前向。执行仍在 `op::conv` / `op::norm`。
//!
//! # 下界分两档，理由不同
//!
//! - **卷积族**（`Conv1d`/`Conv2d`/`ConvTranspose1d`/`ConvTranspose2d`）绑 `gemm_f32`：im2col 之后
//!   它们就是矩阵乘，语义同一副骨架。`gemm_f32` 的收缩轴是声明档 `K = 8`，而真实卷积的
//!   `K = kh·kw·cin` 随几何变化，所以这里给出的是**松下界**——下界本来就允许偏松，且它不会高于
//!   真实代价。
//! - **窗口/归一化族**（池化、插值、四种归一化）没有可代理的已注册算子（池化的窗口 `max`、
//!   归一化的减均值都不属于任何现有骨架），因此 `ops` 为空、下界走 [`Contract::bounds`] 的默认
//!   `Ok(None)`：层仍被登记与收口，只明确标"当前无法计价"，而不是绑一个语义不符的算子名把
//!   数字凑出来。

use diagnostic::{Code, Error, Result};
use op::conv::ConvGeom;
use op::deconv::{ConvTranspose1d as OpDeconv1, ConvTranspose2d as OpDeconv2};
use op::conv::{
    AdaptiveAvgPool2d as OpAdaptive, Conv1d as OpConv1, Conv2d as OpConv2, Interpolate as OpInterp,
    MaxPool1d as OpPool1, MaxPool2d as OpPool2,
};
use op::norm::{BatchNorm2d as OpBatch, GroupNorm as OpGroup, InstanceNorm as OpInst, LayerNorm as OpLayer};
use tensor::Shape;

use crate::{Bounds, BoundsResult, Contract, ContractSpec, ParamRole, spec_accessors};
use crate::ContractSpec as Spec;

/// 卷积/反卷积族绑定的算子：im2col 之后就是一次矩阵乘。
const GEMM: &str = "gemm_f32";

/// 校验非零并在失败时点名层与字段。
fn nonzero(value: usize, field: &str, layer: &str) -> Result<usize> {
    if value == 0 {
        return Err(Error::new(
            Code::Invalid,
            format!("{layer}: `{field}` 必须非零"),
        )
        .with("nn::classic"));
    }
    Ok(value)
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
        let spec = __layer_spec_Conv2d!(name, oh, ow, cout, GEMM)?;
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
        Bounds::of_context(GEMM, self.oh * self.ow, self.cout).map(Some)
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
        nonzero(cin, "cin", layer)?;
        nonzero(cout, "cout", layer)?;
        let deconv = OpDeconv1::new(cin, cout, k, s, p);
        let ol = deconv.len(l)?;
        let spec = __layer_spec_ConvTranspose1d!(name, ol, cout, GEMM)?;
        Ok(Self { spec, deconv, ol })
    }

    /// 输出长度。
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
        nonzero(cin, "cin", layer)?;
        nonzero(cout, "cout", layer)?;
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
    pub fn new(name: impl Into<String>, k: usize, s: usize, p: usize, l: usize, c: usize) -> Result<Self> {
        let layer = "nn::MaxPool1d::new";
        nonzero(c, "channels", layer)?;
        let pool = OpPool1 { k, s, p };
        let ol = pool.len(l)?;
        let spec = Spec::new(name, crate::Shape::new([ol, c]), Vec::new())?;
        Ok(Self { spec, pool, ol, c })
    }

    /// 输出长度。
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
    pub fn new(name: impl Into<String>, geom: ConvGeom, h: usize, w: usize, c: usize) -> Result<Self> {
        let layer = "nn::MaxPool2d::new";
        nonzero(c, "channels", layer)?;
        let pool = OpPool2 { geom };
        let (oh, ow) = pool.shape(h, w)?;
        let spec = Spec::new(name, crate::Shape::new([oh, ow, c]), Vec::new())?;
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
        let pool = op::conv::AvgPool2d { geom, count_include_pad };
        let (oh, ow) = pool.shape(h, w)?;
        let spec = Spec::new(name, crate::Shape::new([oh, ow, c]), Vec::new())?;
        Ok(Self {
            spec,
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

impl Contract for AvgPool2d {
    spec_accessors!();
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
        let spec = Spec::new(name, crate::Shape::new([oh, ow, c]), Vec::new())?;
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
        let spec = Spec::new(name, crate::Shape::new([oh, ow, c]), Vec::new())?;
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
        if !(eps > 0.0) {
            return Err(Error::new(Code::Invalid, format!("{layer}: `eps` 必须为正")).with(layer));
        }
        if !(0.0..=1.0).contains(&momentum) {
            return Err(
                Error::new(Code::Invalid, format!("{layer}: `momentum` 必须落在 [0, 1]"))
                    .with(layer),
            );
        }
        let norm = OpBatch {
            channels,
            eps,
            momentum,
        };
        let spec = Spec::new(name, crate::Shape::new([channels]), Vec::new())?;
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
        nonzero(groups, "groups", layer)?;
        nonzero(channels, "channels", layer)?;
        if !(eps > 0.0) {
            return Err(Error::new(Code::Invalid, format!("{layer}: `eps` 必须为正")).with(layer));
        }
        if channels % groups != 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("{layer}: `channels`({channels}) 不能被 `groups`({groups}) 整除"),
            )
            .with(layer));
        }
        let norm = OpGroup {
            groups,
            channels,
            eps,
        };
        let spec = Spec::new(name, crate::Shape::new([channels]), Vec::new())?;
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
        if !(eps > 0.0) {
            return Err(Error::new(Code::Invalid, format!("{layer}: `eps` 必须为正")).with(layer));
        }
        let norm = OpInst { channels, eps };
        let spec = Spec::new(name, crate::Shape::new([channels]), Vec::new())?;
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
        if !(eps > 0.0) {
            return Err(Error::new(Code::Invalid, format!("{layer}: `eps` 必须为正")).with(layer));
        }
        let norm = OpLayer { width, eps };
        let spec = Spec::new(name, crate::Shape::new([width]), Vec::new())?;
        Ok(Self { spec, norm, width })
    }

    /// 归一化宽度。
    pub fn width(&self) -> usize {
        self.width
    }
}

impl Contract for LayerNorm {
    spec_accessors!();
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
    fn conv2d_registers_its_shape_and_gemm_operator() {
        let layer = Conv2d::new("stem.conv", 3, 64, geom3(), 32, 32).expect("合法");
        assert_eq!(layer.name(), "stem.conv");
        assert_eq!(layer.output().dims(), [32, 32, 64]);
        assert_eq!(layer.spca().ops(), [GEMM]);
        assert_eq!(layer.shape(), (32, 32));
        // 下界来自 gemm 语义表，读侧应大于写侧。
        let floor = layer.bounds().expect("合法").expect("有算子");
        assert!(floor.read() > floor.write());
    }

    #[test]
    fn conv2d_rejects_zero_channels() {
        assert!(Conv2d::new("x", 0, 8, geom3(), 8, 8).is_err());
    }

    #[test]
    fn conv1d_and_transpose_lengths_follow_the_geometry() {
        let layer = Conv1d::new("a", 4, 8, ConvGeom { kh: 2, kw: 1, sh: 2, sw: 1, ph: 0, pw: 0 }, 9)
            .expect("合法");
        assert_eq!(layer.len(), 5);
        assert_eq!(layer.output().dims(), [5, 8]);

        let up = ConvTranspose1d::new("up", 4, 6, 2, 2, 0, 9).expect("合法");
        assert_eq!(up.len(), 18);
        assert_eq!(up.output().dims(), [18, 6]);

        let up2 = ConvTranspose2d::new("up2", 4, 6, geom3(), 8, 8).expect("合法");
        assert_eq!(up2.shape(), (8, 8));
        assert_eq!(up2.output().dims(), [8, 8, 6]);
    }

    #[test]
    fn pool_and_interpolate_layers_are_registered_without_operators() {
        let pool = MaxPool1d::new("p1", 2, 2, 0, 8, 4).expect("合法");
        assert_eq!(pool.output().dims(), [4, 4]);
        assert!(pool.spca().ops().is_empty());
        // 没有可代理算子 -> 明确"无法计价"，而不是凑一个数。
        assert!(pool.bounds().expect("合法").is_none());

        let pool2 = MaxPool2d::new("p2", geom3(), 8, 8, 4).expect("合法");
        assert_eq!(pool2.output().dims(), [8, 8, 4]);

        let pool2 = AvgPool2d::new("p3", geom3(), false, 8, 8, 4).expect("合法");
        assert_eq!(pool2.output().dims(), [8, 8, 4]);

        let pool2 = AdaptiveAvgPool2d::new("p4", 2, 2, 4).expect("合法");
        assert_eq!(pool2.output().dims(), [2, 2, 4]);

        let interp = Interpolate::new("i", op::conv::Sampler::Linear, 16, 16, 4).expect("合法");
        assert_eq!(interp.output().dims(), [16, 16, 4]);
    }

    #[test]
    fn norm_layers_check_their_geometry() {
        let bn = BatchNorm2d::new("bn", 64, 1e-5, 0.1).expect("合法");
        assert_eq!(bn.output().dims(), [64]);
        assert!(bn.bounds().expect("合法").is_none());
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
