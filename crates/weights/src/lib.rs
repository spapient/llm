//! 权重文件的结构：[`Entry`]、[`Header`] 与两条读取契约（[`Reader`] / [`Loader`]），以及
//! safetensors 的两种后端（[`safetensors::Archive`] 借用整份字节、[`safetensors::File`] 只读头部）。
//!
//! **拥有**：`Header`/`Entry`/`Reader`/`Loader` 三条契约，`safetensors` 的两种后端，自有的 `.u`
//! 容器 v2（`container`/`writer`/`archive`/`seg`/`codec`；CRC/LEB/Sink 已归 L1 `wire`）、量化方案
//! `quant`/`ptq`，外部格式只读适配（`torch` 含 ZIP/pickle 解码、`gguf` 含量化块反量化、`json`），
//! 以及“外部格式 → 中性段流”的桥接（`archive::import`）。
//! **不拥有**：dtype 闭集（`dtype`）、张量元数据（`tensor`）、物理布局（`storage`）。
//! **依赖**：`dtype`、`diagnostic`、`tensor`、`wire`（`storage` 经可选 `mmap` feature
//! 引入，只在 Unix 上提供映射源；`+dev`：`testkit`）。
//! **被依赖**：`train`、`nn`、`llm`（`+dev`：`fuzz`）。
//!
//! 两条读取契约的分工是所有权而不是风格：[`Reader`] 把字节**借**出来（内存切片、映射），
//! [`Loader`] 把字节**拷进**调用方给的缓冲区——只有后者能由文件实现，因为不引入 `unsafe` 就无法
//! 自己映射，也无法承诺"借出的引用在函数返回后仍然有效"。真实权重动辄十几 GB，靠的正是后者。
//!
//! 这一层先解决“权重文件能不能自证合法”：条目名唯一、字节区间不重叠、区间起点不早于
//! 数据段，以及 safetensors 的“声明区间必须等于形状推出的字节数”。这几条不成立时，任何
//! 后续读写都会在别处以更难查的方式失败。
//!
//! 解析器只填这三个类型，不新增契约：换个文件格式（GGUF）只需再写一个“字节 → `Header`”，
//! 上层（`nn` 的层清单、`train` 的检查点）不需要跟着改。
//!
//! # 例子
//!
//! ```
//! use tensor::{DType, Shape};
//! use llm_weights::{Entry, Header};
//!
//! let entry = Entry::new("blk.0.attn_q.weight", DType::F32, Shape::new([2, 3])).at(64);
//! let header = Header::new(vec![entry], 64).expect("区间合法");
//! assert_eq!(header.data_bytes(), 24);
//! assert!(header.get("blk.0.attn_q.weight").is_some());
//! ```

#![deny(unsafe_code)]

/// 全 crate 的“非法输入”构造点转发自 L1 [`wire`]：判据与上下文标注各只有一处实现。
pub(crate) use storage::bad;

// 权重格式的闭集枚举与字段编解码声明宏：新格式槽位只填声明，不重写读写骨架。
// —— 闭集枚举宏：全 crate 的"码位 ↔ 变体 ↔ 短名"单一生成点；文本作用域须先于各使用模块 ——
//
// 线上码位稳定是容器格式的一部分，因此每个变体的码位只写一次——枚举本身与它的
// 反向解析不再各维护一份 `match`，未知码统一显式拒绝、不静默回退。带短名的枚举
// 还把账本/文本里的字符串写法一并生成，短名与码位同源、不会漂移。
//
// 形态选择：
// - 默认（可带 `=> "短名"`）：枚举 + `from_code`（未知码 `Invalid`），带短名时另生成 `as_str`；
// - `@named`：只有枚举与 `as_str`，没有线上数值码的反向解析（账本列）；
// - `@coded` / `@coded_unsupported`：另生成 `code()` 访问器，供线上读写双端使用；
//   后者把未知码报成 `Unsupported`——读到本端不认识的格式特性，与"字节损坏"（`Invalid`）
//   不是一回事。

/// 生成一个带稳定判别值的闭集枚举，以及它的 `from_code`；带 `=> "短名"` 时再生成 `as_str`。
///
/// 每个变体带 `= 码位`；`from_code` 按判别值反向解析，未知码返回 [`crate::bad`] 的
/// `Invalid` 错误。变体上的文档注释原样保留。首 token 选 `@named` / `@coded` /
/// `@coded_unsupported` 可换形态，见上方形态选择注释。
macro_rules! closed_enum {
    // 带短名：枚举 + from_code + as_str。
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident = $code:literal => $sname:literal,
            )+
        }
    ) => {
        closed_enum!(@str_only
            $(#[$meta])*
            $vis enum $name {
                $( $(#[$variant_meta])* $variant = $code, )+
            }
            { $( $variant => $sname, )+ }
        );
        closed_enum!(@from_code $name { $( $code => $variant, )+ });
    };

    // 无短名：只有枚举与 from_code。
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident = $code:literal,
            )+
        }
    ) => {
        closed_enum!(@plain
            $(#[$meta])*
            $vis enum $name {
                $( $(#[$variant_meta])* $variant = $code, )+
            }
        );
    };

    // —— 内部臂：枚举定义 / 反向解析 / 字符串写法，各只写一份 ——
    (
        @def
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $( $(#[$variant_meta:meta])* $variant:ident = $code:literal, )+
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u8)]
        $vis enum $name {
            $( $(#[$variant_meta])* $variant = $code, )+
        }
    };

    (
        @from_code
        $name:ident {
            $( $code:literal => $variant:ident, )+
        }
    ) => {
        impl $name {
            /// 从码位解析；未知码显式报错，不静默回退。
            pub(crate) fn from_code(code: u8) -> diagnostic::Result<Self> {
                match code {
                    $( $code => Ok(Self::$variant), )+
                    other => Err(crate::bad(format!(
                        "未知 {} 码 {other}",
                        stringify!($name)
                    ))),
                }
            }
        }
    };

    (
        @as_str
        $name:ident {
            $( $variant:ident => $sname:literal, )+
        }
    ) => {
        impl $name {
            /// 账本/线上的字符串写法（闭集，码位稳定）。
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $( Self::$variant => $sname, )+
                }
            }
        }
    };

    // —— 内部臂：只带短名（枚举定义 + as_str），不带 from_code；供外层两个臂共用 ——
    (
        @str_only
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $( $(#[$variant_meta:meta])* $variant:ident = $code:literal, )+
        }
        { $( $vs:ident => $sname:literal, )+ }
    ) => {
        closed_enum!(@def
            $(#[$meta])*
            $vis enum $name {
                $( $(#[$variant_meta])* $variant = $code, )+
            }
        );
        closed_enum!(@as_str $name { $( $vs => $sname, )+ });
    };

    // 只带短名：没有线上数值码的反向解析，只有枚举定义与 `as_str`；调用首 token 写 `@named`。
    (
        @named
        $(#[$nm:meta])*
        $v:vis enum $e:ident {
            $( $(#[$vm:meta])* $var:ident = $c:literal => $sn:literal, )+
        }
    ) => {
        closed_enum!(@str_only
            $(#[$nm])*
            $v enum $e {
                $( $(#[$vm])* $var = $c, )+
            }
            { $( $var => $sn, )+ }
        );
    };

    // —— 读写双端形态：另生成 `code()`，写端不再 `as u8` 散写 ——

    // 模式决定反向解析的错误类别，以及是否另生成线上 code() 访问器。
    (
        @plain $name:ident { $( $code:literal => $variant:ident, )+ }
    ) => {
        closed_enum!(@from_code $name { $( $code => $variant, )+ });
    };
    (
        @coded $name:ident { $( $code:literal => $variant:ident, )+ }
    ) => {
        closed_enum!(@code $name);
        closed_enum!(@from_code $name { $( $code => $variant, )+ });
    };
    (
        @coded_unsupported $name:ident { $( $code:literal => $variant:ident, )+ }
    ) => {
        closed_enum!(@code $name);
        closed_enum!(@from_code_unsupported $name { $( $code => $variant, )+ });
    };

    (
        @$mode:ident
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident = $code:literal,
            )+
        }
    ) => {
        closed_enum!(@def
            $(#[$meta])*
            $vis enum $name {
                $( $(#[$variant_meta])* $variant = $code, )+
            }
        );
        closed_enum!(@$mode $name { $( $code => $variant, )+ });
    };

    (
        @code $name:ident
    ) => {
        impl $name {
            /// 线上码位：与判别值同源，不存在第二份映射表。
            pub(crate) const fn code(self) -> u8 {
                self as u8
            }
        }
    };

    (
        @from_code_unsupported
        $name:ident {
            $( $code:literal => $variant:ident, )+
        }
    ) => {
        impl $name {
            /// 从码位解析；未知码按"本端不支持"显式报错，不静默回退。
            pub(crate) fn from_code(code: u8) -> diagnostic::Result<Self> {
                match code {
                    $( $code => Ok(Self::$variant), )+
                    other => Err(diagnostic::Error::new(
                        diagnostic::Code::Unsupported,
                        format!("未知 {} 码 {other}", stringify!($name)),
                    )),
                }
            }
        }
    };
}

/// 由线上字段声明生成 `encode_fields` / `decode_fields`：字段顺序就是字节顺序，
/// 写端与读端不可能各维护一份字段列表。语义校验、`scheme` 分支与公开错误口径仍由调用方保留。
///
/// 字段类型是刻意收敛的小闭集：`leb_u32`、`bool`、`bool_flag<mask>`、`f64`、
/// `enum_u8<Type>`、`wire_type`，以及 `wire_type if 布尔字段` 的条件尾部。
/// 需要新类型时在这里加一组内部臂，而不是在格式模块里重新手写对称代码。
macro_rules! wire_fields {
    (
        $name:ident {
            $(
                $field:ident : $kind:ident $(<$arg:tt>)? $(if $guard:ident)?
            ),+ $(,)?
        }
    ) => {
        impl $name {
            /// 按声明顺序编码字段体；外层负责语义校验与前导 scheme。
            fn encode_fields(&self, out: &mut impl storage::sink::Sink) {
                $(
                    wire_fields!(@encode self, out, $field, $kind $(<$arg>)? $(if $guard)?);
                )+
            }
            /// 按声明顺序解码字段体；外层负责组装结构与语义校验。
            fn decode_fields(
                cur: &mut storage::leb::Cursor<'_>,
            ) -> diagnostic::Result<(
                $( wire_fields!(@field_type $kind $(<$arg>)? $(if $guard)?), )+
            )> {
                $(
                    wire_fields!(@decode $name, cur, $field, $kind $(<$arg>)? $(if $guard)?);
                )+
                Ok(($($field,)+))
            }
        }
    };
    (@field_type leb_u32) => { u32 };
    (@field_type bool) => { bool };
    (@field_type bool_flag<$mask:literal>) => { bool };
    (@field_type f64) => { f64 };
    (@field_type enum_u8<$ty:ident>) => { $ty };
    (@field_type wire_type) => { crate::Wire };
    (@field_type wire_type if $guard:ident) => { Option<crate::Wire> };
    (@encode $this:ident, $out:ident, $field:ident, leb_u32) => {
        storage::leb::write($out, $this.$field);
    };
    (@encode $this:ident, $out:ident, $field:ident, bool) => {
        $out.put(&[u8::from($this.$field)]);
    };
    (@encode $this:ident, $out:ident, $field:ident, bool_flag<$mask:literal>) => {
        $out.put(&[u8::from($this.$field)]);
    };
    (@encode $this:ident, $out:ident, $field:ident, f64) => {
        $out.put(&$this.$field.to_le_bytes());
    };
    (@encode $this:ident, $out:ident, $field:ident, enum_u8<$ty:ident>) => {
        $out.put(&[$this.$field as u8]);
    };
    (@encode $this:ident, $out:ident, $field:ident, wire_type) => {
        storage::leb::write($out, $this.$field.code());
    };
    (@encode $this:ident, $out:ident, $field:ident, wire_type if $guard:ident) => {
        if $this.$guard {
            if let Some(value) = $this.$field {
                storage::leb::write($out, value.code());
            }
        }
    };
    (@decode $name:ident, $cur:ident, $field:ident, leb_u32) => {
        let $field = $cur.read::<u32>()?;
    };
    (@decode $name:ident, $cur:ident, $field:ident, bool) => {
        let $field = $cur.read_u8()? != 0;
    };
    (@decode $name:ident, $cur:ident, $field:ident, bool_flag<$mask:literal>) => {
        let $field = {
            let flags = $cur.read_u8()?;
            if flags & !$mask != 0 {
                return Err(crate::bad(format!(
                    "{} flags 含非零预留位",
                    stringify!($name),
                )));
            }
            flags & $mask != 0
        };
    };
    (@decode $name:ident, $cur:ident, $field:ident, f64) => {
        let $field = {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice($cur.take(8)?);
            f64::from_le_bytes(bytes)
        };
    };
    (@decode $name:ident, $cur:ident, $field:ident, enum_u8<$ty:ident>) => {
        let $field = <$ty>::from_code($cur.read_u8()?)?;
    };
    (@decode $name:ident, $cur:ident, $field:ident, wire_type) => {
        let $field = crate::read_wire($cur)?;
    };
    (@decode $name:ident, $cur:ident, $field:ident, wire_type if $guard:ident) => {
        let $field = if $guard {
            Some(crate::read_wire($cur)?)
        } else {
            None
        };
    };
}

// —— 自有 `.u` 容器 v2：描述符、读/写、帧编码、增量补丁与双模式 ——
mod archive;
mod container;
mod seg;
mod writer;

// —— 平铺的子模块：原各父模块的单文件子目录，拍平到顶层 ——
pub(crate) mod codec;

// —— 量化：磁盘方案、成对推断与 2/3/4/8-bit 对称 PTQ ——
pub mod quant;

/// 对称训练后量化；实现与磁盘量化方案同居 `quant`，公开路径保持稳定。
pub mod ptq {
    pub use crate::quant::{
        Adaptive, AdaptiveCompensated, Compensated, I2, I3, I4, I8, Scheme, adaptive_quantize,
        compensate, compensate_adaptive, dequantize, quantize,
    };
}

// —— 谱隙证书链：f64 单边 Jacobi SVD、谱隙判定账本、五轴无损选码（.u v2 张量级压缩侧车）——
pub mod axes;
pub mod ledger;
/// 单边 Jacobi 奇异值分解与谱隙选秩；数学内核由 `ledger` 持有。
pub mod svd {
    pub use crate::ledger::{REL_GAP, Svd, compute_svd, signal_rank};
    pub(crate) use crate::ledger::{Spectral, factor_low, spectral_gap};
}

// —— 沉降账本：LLD §4.7 的逐张量体积 / 原因 / 无损判决，全部由落盘字节现算 ——
pub mod settle;

// —— L3 合成律：LLD §4.5 的逐段证书 → 网络级误差上界与适用域 ——
pub mod chain;

// —— 外部格式只读适配：torch `.bin`（ZIP_STORED + pickle，已平铺进单模块）、GGUF、受限 JSON ——
pub mod gguf;
pub mod json;
pub mod torch;

pub use archive::import;
pub mod safetensors;

// 纯字节码基础件归 L1 `wire`；这里只做薄转发，保持容器内部路径不变。
/// CRC-32（转发自 `storage::crc`）。
pub mod crc {
    pub use storage::crc::*;
}

/// 生命周期模式薄转发：实现见内部 `archive` 模块。
pub mod mode {
    pub use crate::archive::{MODE_NAME, Mode, is_opt, mode_of, release};
}

/// 增量补丁薄转发：实现见内部 `archive` 模块。
pub mod patch {
    pub use crate::archive::{diff, join};
}

#[cfg(test)]
#[path = "../tests/gguf_fixture.rs"]
pub(crate) mod gguf_fixture;

/// 词表段保留名（字节容器占位，不接 tokenizer）。
pub use crate::archive::VOCAB_NAME;

pub use archive::{Archive, E8_RATIO, Metrics, Proof};
pub use axes::{Axis, AxisSet};
pub use chain::{Bracket, Chain, Ruling, Rung};
pub use dtype::DType as Wire;

pub use ptq::*;
pub use quant::{Quant, Quantized};
pub use seg::Codec;
pub use seg::Segment;
pub use seg::Stored;
pub use settle::{Reason, Settled, Settlement, Thresholds};
pub use writer::{Config, Writer};

pub use container::{
    Entry, Header, Loader, Reader, decode_entry_f32, decode_f32, load_f32, read_f32,
};

/// 权重文件的检查点快照：字节格式归 L2 `storage::snapshot`，这里把它具体化到 [`Header`]。
///
/// 上层照旧写 `weights::Snapshot`，不必知道快照是泛型的；换一种权重容器只是换一个别名。
pub type Snapshot = storage::snapshot::Snapshot<Header>;
pub(crate) use container::{align_up, check_capacity, read_wire};

#[cfg(all(feature = "mmap", unix))]
pub use safetensors::MmapSource;

#[cfg(test)]
#[path = "../tests/lib.rs"]
mod tests;

#[cfg(test)]
mod testutil {
    //! 测试共享夹具：f32 段构造与归档写盘，mode/patch 等模块的测试共用同一份。
    use crate::Wire;
    use crate::seg::Segment;
    use crate::writer::Writer;

    /// 由 f32 值构造一个单维 F32 段。
    pub(crate) fn f32_segment(name: &str, vals: &[f32]) -> Segment {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        Segment::new(name, Wire::F32, vec![vals.len() as u64], None, bytes).expect("段")
    }

    /// 由 f32 值构造一个二维 F32 段：五轴选码只认二维稠密权重，形状因此必须给全。
    pub(crate) fn f32_matrix(name: &str, vals: &[f32], (m, n): (usize, usize)) -> Segment {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        Segment::new(name, Wire::F32, vec![m as u64, n as u64], None, bytes).expect("段")
    }

    /// 把段依次写进归档，返回对齐到 `2^6` 的字节。
    pub(crate) fn archive(segments: Vec<Segment>) -> Vec<u8> {
        let mut w = Writer::new();
        for seg in segments {
            w.push(seg).expect("push");
        }
        w.finish(6).expect("字节")
    }

    /// 读取本机 torch `.bin` 样本；样本缺失返回 `None`，测试跳过而不是伪造字节。
    pub(crate) fn real_bin_bytes() -> Option<Vec<u8>> {
        std::fs::read("/tmp/pt/model.bin")
            .ok()
            .or_else(|| testkit::skip_none("weights 样本缺失"))
    }
}
