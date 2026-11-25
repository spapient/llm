//! 经典几何层：卷积、转置卷积、池化、插值与归一化的契约登记。
//!
//! 这一族里窗口滑动口径是同一套 `ConvGeom`——卷积读它算输出长度，池化读它算第几段；归一化层
//! 不碰窗口，只碰通道与组的整除关系（`check_group_norm` 就是这条约束的校验）。两者共享同一条
//! 形状约定（末轴是通道、前面各轴合成外层行数），故同居一个文件：合并的判据是**声明口径相同**，
//! 而不是谁的行数更少。层只**声明**形状与算子归属，不执行、不持有权重。
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
use crate::{Bounds, BoundsResult, Contract, ContractSpec, LayerGeometry, nonzero, spec_accessors};
use diagnostic::{Code, Error, Result};

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

/// 空间层的平铺声明：字段、签名、校验、几何计划、输出轴与特有方法都在声明点可见。
/// `check: |layer|` 只绑定诊断位置；计划直接产出字段，公共骨架不隐藏任何几何算法。
macro_rules! spatial_layer {
    (
        $(#[$meta:meta])* $layer:ident, $spec:ident, $op:expr;
        ($($field:ident: $fty:ty),*);
        ($($arg:ident: $aty:ty),*);
        check: |$site:ident| $check:block;
        $plan:block;
        ($($axis:expr),*);
        {$($methods:tt)*}
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $layer {
            $($field: $fty),*
        }

        impl $layer {
            #[doc = concat!("构造 [`", stringify!($layer), "`]，按输入与几何计算输出形状。")]
            ///
            /// # Errors
            ///
            /// 参数或几何不合法、输入无法形成合法输出，或算子规格无效时返回说明。
            pub fn new(name: impl Into<String>, $($arg: $aty),*) -> Result<Self> {
                let $site = concat!("nn::", stringify!($layer), "::new");
                $check
                let ($($field),*) = $plan;
                let spec = $spec!(name, $($axis,)* $op)?;
                Ok(Self { spec, $($field),* })
            }

            $($methods)*
        }

        crate::priced!($layer, $op);
    };
}

spatial_layer!(
    /// 一维卷积层：NHWC 输入 `[l, cin]`，输出 `[ol, cout]`。
    #[crate::spec(name = $name, out = [$ol, $cout], ops = [$op])]
    Conv1d, __layer_spec_Conv1d, GEMM;
    (cin: usize, cout: usize, geom: ConvGeom, ol: usize);
    (cin: usize, cout: usize, geom: ConvGeom, l: usize);
    check: |layer| { nonzero_channels(cin, cout, layer)?; };
    {
        let kernel = OpConv1::new(cin, cout, geom.kw, geom.sw, geom.pw);
        let ol = kernel.len(l)?;
        (cin, cout, geom, ol)
    };
    (ol, cout);
    {
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
);

spatial_layer!(
    /// 二维卷积层：NHWC 输入 `[h, w, cin]`，输出 `[oh, ow, cout]`。
    #[crate::spec(name = $name, out = [$oh, $ow, $cout], ops = [$op])]
    Conv2d, __layer_spec_Conv2d, CONV;
    (cin: usize, cout: usize, geom: ConvGeom, oh: usize, ow: usize);
    (cin: usize, cout: usize, geom: ConvGeom, h: usize, w: usize);
    check: |layer| { nonzero_channels(cin, cout, layer)?; };
    {
        let kernel = OpConv2::new(cin, cout, geom);
        let (oh, ow) = kernel.shape(h, w)?;
        (cin, cout, geom, oh, ow)
    };
    (oh, ow, cout);
    {
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
);

spatial_layer!(
    /// 一维反卷积层：`[l, cin]` → `[ol, cout]`，权重 `[cin, cout, k]`。
    #[crate::spec(name = $name, out = [$ol, $cout], ops = [$op])]
    ConvTranspose1d, __layer_spec_ConvTranspose1d, GEMM;
    (deconv: OpDeconv1, ol: usize);
    (cin: usize, cout: usize, k: usize, s: usize, p: usize, l: usize);
    check: |layer| { nonzero_channels(cin, cout, layer)?; };
    {
        let deconv = OpDeconv1::new(cin, cout, k, s, p);
        let ol = deconv.len(l)?;
        (deconv, ol)
    };
    (ol, cout);
    {
        crate::accessors! {
            /// 输出长度。
            #[allow(clippy::len_without_is_empty)]
            len -> usize = ol,
        }

        /// 输出通道。
        pub fn cout(&self) -> usize {
            self.deconv.cout
        }
    }
);

spatial_layer!(
    /// 二维反卷积层：`[h, w, cin]` → `[oh, ow, cout]`，权重 `[cin, cout, kh, kw]`。
    #[crate::spec(name = $name, out = [$oh, $ow, $cout], ops = [$op])]
    ConvTranspose2d, __layer_spec_ConvTranspose2d, GEMM;
    (deconv: OpDeconv2, oh: usize, ow: usize);
    (cin: usize, cout: usize, geom: ConvGeom, h: usize, w: usize);
    check: |layer| { nonzero_channels(cin, cout, layer)?; };
    {
        let deconv = OpDeconv2::new(cin, cout, geom);
        let oh = deconv.oh(h)?;
        let ow = deconv.ow(w)?;
        (deconv, oh, ow)
    };
    (oh, ow, cout);
    {
        /// 输出边长 `(oh, ow)`。
        pub fn shape(&self) -> (usize, usize) {
            (self.oh, self.ow)
        }

        /// 输出通道。
        pub fn cout(&self) -> usize {
            self.deconv.cout
        }
    }
);

spatial_layer!(
    /// 一维最大池化层：`[l, c]` → `[ol, c]`。
    #[crate::spec(name = $name, out = [$ol, $c], ops = [$op])]
    MaxPool1d, __layer_spec_MaxPool1d, "max_pool1d_f32";
    (ol: usize, c: usize);
    (k: usize, s: usize, p: usize, l: usize, c: usize);
    check: |layer| { nonzero(c, "channels", layer)?; };
    {
        // 算子只负责构造期几何计算，不作为永远读不到的状态存进层。
        let ol = OpPool1 { k, s, p }.len(l)?;
        (ol, c)
    };
    (ol, c);
    {
        crate::accessors! {
            /// 输出长度。
            #[allow(clippy::len_without_is_empty)]
            len -> usize = ol,
            /// 通道数。
            channels -> usize = c,
        }
    }
);

spatial_layer!(
    /// 二维最大池化层：`[h, w, c]` → `[oh, ow, c]`。
    #[crate::spec(name = $name, out = [$oh, $ow, $c], ops = [$op])]
    MaxPool2d, __layer_spec_MaxPool2d, "max_pool2d_f32";
    (oh: usize, ow: usize, c: usize);
    (geom: ConvGeom, h: usize, w: usize, c: usize);
    check: |layer| { nonzero(c, "channels", layer)?; };
    {
        let (oh, ow) = OpPool2 { geom }.shape(h, w)?;
        (oh, ow, c)
    };
    (oh, ow, c);
    {
        /// 输出边长。
        pub fn shape(&self) -> (usize, usize) {
            (self.oh, self.ow)
        }

        crate::accessors! {
            /// 通道数。
            channels -> usize = c,
        }
    }
);

spatial_layer!(
    /// 二维平均池化层：`count_include_pad` 决定补零是否计入分母。
    #[crate::spec(name = $name, out = [$oh, $ow, $c], ops = [$op])]
    AvgPool2d, __layer_spec_AvgPool2d, "avg_pool2d_f32";
    (oh: usize, ow: usize, c: usize);
    (geom: ConvGeom, count_include_pad: bool, h: usize, w: usize, c: usize);
    check: |layer| { nonzero(c, "channels", layer)?; };
    {
        let (oh, ow) = crate::conv::AvgPool2d {
            geom,
            count_include_pad,
        }
        .shape(h, w)?;
        (oh, ow, c)
    };
    (oh, ow, c);
    {
        /// 输出边长。
        pub fn shape(&self) -> (usize, usize) {
            (self.oh, self.ow)
        }

        crate::accessors! {
            /// 通道数。
            channels -> usize = c,
        }
    }
);

spatial_layer!(
    /// 自适应平均池化层：任意输入压到固定 `[oh, ow, c]`。
    #[crate::spec(name = $name, out = [$oh, $ow, $c], ops = [$op])]
    AdaptiveAvgPool2d, __layer_spec_AdaptiveAvgPool2d, "adaptive_pool_f32";
    (oh: usize, ow: usize, c: usize);
    (oh: usize, ow: usize, c: usize);
    check: |layer| {
        nonzero(oh, "oh", layer)?;
        nonzero(ow, "ow", layer)?;
        nonzero(c, "channels", layer)?;
    };
    { (oh, ow, c) };
    (oh, ow, c);
    {
        /// 输出边长。
        pub fn shape(&self) -> (usize, usize) {
            (self.oh, self.ow)
        }

        crate::accessors! {
            /// 通道数。
            channels -> usize = c,
        }
    }
);

spatial_layer!(
    /// 插值层：`[h, w, c]` → `[oh, ow, c]`；`sampler` 是决定取样方式的真实状态。
    #[crate::spec(name = $name, out = [$oh, $ow, $c], ops = [$op])]
    Interpolate, __layer_spec_Interpolate, "interpolate_f32";
    (sampler: crate::conv::Sampler, oh: usize, ow: usize, c: usize);
    (sampler: crate::conv::Sampler, oh: usize, ow: usize, c: usize);
    check: |layer| {
        nonzero(oh, "oh", layer)?;
        nonzero(ow, "ow", layer)?;
        nonzero(c, "channels", layer)?;
    };
    { (sampler, oh, ow, c) };
    (oh, ow, c);
    {
        /// 输出边长。
        pub fn shape(&self) -> (usize, usize) {
            (self.oh, self.ow)
        }

        crate::accessors! {
            /// 通道数。
            channels -> usize = c,
            /// 取样方式。
            sampler -> crate::conv::Sampler = sampler,
        }
    }
);

// ============================ 归一化 ============================
//
// 归一化层与上面的窗口族共用同一个文件：它们都不碰窗口，只碰通道与组的整除关系，而形状口径
// （末轴是通道）与下界口径（问算子语义表）完全一致。层只声明形状与算子归属，执行仍在
// `nn::norm`。

type OpBatch = crate::norm::BatchNorm2d<f32>;
type OpGroup = crate::norm::GroupNorm<f32>;
type OpInst = crate::norm::InstanceNorm<f32>;
type OpNorm = crate::norm::LayerNorm<f32>;

/// 校验 `eps` 为正且有限：所有归一化层共用。
fn check_eps(eps: f32, layer: &str) -> Result<()> {
    if !(eps.is_finite() && eps > 0.0) {
        return Err(
            Error::new(Code::Invalid, format!("{layer}: `eps` 必须为正且有限")).with(layer),
        );
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

/// 四个归一化层只差三处：公开签名、构造前校验、算子参数袋；结构体、规格装配、那两条访问器
/// 与 `Contract` 委托四层逐字相同。收进一个宏体写一次，每层就只剩一行声明——签名、校验、
/// 参数袋并排，差异一眼可见。
macro_rules! norm_layer {
    (
        $(#[$d:meta])* $t:ident, $o:ident, $p:literal, $a:ident;
        ($($arg:ident: $aty:ty),*);
        $chk:block;
        ($($f:ident),*)
    ) => {
        $(#[$d])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $t {
            spec: ContractSpec,
            norm: $o,
            $a: usize,
        }
        impl $t {
            /// 构造。
            ///
            /// # Errors
            ///
            /// 轴长为零、`eps ≤ 0`，或该层独有的参数（`momentum` 越界、`channels` 不能被
            /// `groups` 整除）不合法时返回说明。
            pub fn new(name: impl Into<String>, $($arg: $aty),*) -> Result<Self> {
                $chk
                let norm = $o { $($f),* };
                let spec = Spec::new(name, crate::Shape::new([$a]), [$p])?;
                Ok(Self { spec, norm, $a })
            }

            #[doc = concat!("归一化所沿那条轴的尺寸（`", stringify!($a), "`）。")]
            pub fn $a(&self) -> usize {
                self.$a
            }

            /// 归一化算子的参数袋：`eps`、以及该层独有的 `momentum` / `groups`。
            pub fn norm(&self) -> &$o {
                &self.norm
            }
        }

        impl Contract for $t {
            spec_accessors!();

            fn bounds(&self) -> BoundsResult {
                // 层不写第二份流量公式：下界向算子语义表要。
                Bounds::of_geometry($p, LayerGeometry::axis(self.$a)).map(Some)
            }
        }
    };
}

norm_layer!(
    /// 批归一化层（NHWC，最后维 `channels`）。
    BatchNorm2d, OpBatch, "batch_norm_f32", channels;
    (channels: usize, eps: f32, momentum: f32);
    {
        let layer = "nn::BatchNorm2d::new";
        nonzero(channels, "channels", layer)?;
        check_eps(eps, layer)?;
        if !(0.0..=1.0).contains(&momentum) {
            return Err(Error::new(Code::Invalid, format!("{layer}: `momentum` 必须落在 [0, 1]")).with(layer));
        }
    };
    (channels, eps, momentum)
);

norm_layer!(
    /// 组归一化层：`(n, group)` 内按 `(C/G, spatial)` 统计。规格构造口径同 [`BatchNorm2d`]。
    GroupNorm, OpGroup, "group_norm_f32", channels;
    (groups: usize, channels: usize, eps: f32);
    { check_group_norm(groups, channels, "nn::GroupNorm::new")?; check_eps(eps, "nn::GroupNorm::new")?; };
    (groups, channels, eps)
);

norm_layer!(
    /// 实例归一化层：等价于 `GroupNorm(groups = channels)`。规格构造口径同 [`BatchNorm2d`]。
    InstanceNorm, OpInst, "instance_norm_f32", channels;
    (channels: usize, eps: f32);
    { nonzero(channels, "channels", "nn::InstanceNorm::new")?; check_eps(eps, "nn::InstanceNorm::new")?; };
    (channels, eps)
);

norm_layer!(
    /// 层归一化层：逐行在最后一维 `width` 上统计。规格构造口径同 [`BatchNorm2d`]。
    LayerNorm, OpNorm, "layer_norm_f32", width;
    (width: usize, eps: f32);
    { nonzero(width, "width", "nn::LayerNorm::new")?; check_eps(eps, "nn::LayerNorm::new")?; };
    (width, eps)
);

#[cfg(test)]
#[path = "../tests/classic_tests.rs"]
mod tests;
