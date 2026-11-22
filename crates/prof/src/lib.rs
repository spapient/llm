//! 每台设备的**全部数据**：能力档案（区间、存储层、执行单元）与片段文本（原语怎么写）。
//!
//! 一个 crate 装两样东西，理由不是"都跟设备有关"，而是**它们必须一起在构建期读**：能力档案声明
//! "这台设备能拼出 `dot`"，片段文本声明"`dot` 在 `.metal` 里这么写"，两边对不上就该在构建期失败，
//! 而不是等运行期 emit 出来编译不过。合并前它们被切成三个 crate（`prof` / `frag` / `prim`），
//! 而 cargo 不允许一个包当自己的构建依赖——`frag` 存在的唯一理由就是给 `prof` 的 `build.rs` 提供
//! 目录，于是同一件事有两个 crate。现在生成器就在 [`build.rs`](../build.rs) 里。
//!
//! # 四层，各归各位
//!
//! - `src/<vendor>/` 是**内容**：`<vendor>.toml`（存储层物理名、输出语言、工具链版本、执行单元、
//!   每条能力区间）与 8 段 `<prim>.<ext>` 片段、`epilogue.<ext>`、`witness.<ext>`；
//! - [`atom`] 是**唯一的词汇表**：16 个原语、类型/存储位集、参数化签名 `Need`/`Provide`、
//!   属性键闭集 [`atom::AttrKey`]；
//! - [`prof`] 是**唯一的 schema**：解析、闭集字段、见证与 `[units]` 一致性校验；
//! - [`device`] 是**设备行的形状**：这次构建带上的元数据、见证源码、可选镜像与 [`Status`]。
//!
//! # 生成，不解析
//!
//! [`build.rs`](../build.rs) 读 `src/<vendor>/<vendor>.toml`，一次生成两棵树：
//! `$OUT_DIR/embedded.rs`（档案 → `diagnostic::data::Table` 字面量，[`embedded`] / [`names`]，
//! ADR 0022）与 `$OUT_DIR/tables.rs`（片段 → 每台设备一个模块 + `BUILTIN` / `WITNESS` /
//! `DEVICES` / `ROOTS` 四张聚合表）。运行期只遍历这两棵树：不读目录、没有 TOML 解析器。
//!
//! `lib.rs` 里**没有 vendor 清单**：`src/` 下哪个目录有 `<vendor>/<vendor>.toml`，哪台设备就存在。
//! 加一台设备 = 加一个目录，机制代码零改动——这是"设备差异只有数据"的可执行版本。
//!
//! # 依赖方向
//!
//! `cutile` 只做机制（协商、降级、拼接、emit），因此本 crate 不反向依赖它：依赖方向
//! `cutile → prof`，而 `prof` 只依赖 `std` 与同属基座的 `diagnostic`（错误类别由产生错误的那一层
//! 给，见 `docs/ARCHITECTURE.md` 第三节）。
//!
//! # 两种模式：源码调试 / 设备镜像发布
//!
//! 默认（调试）只有源码：片段文本 + 见证源码，运行期照旧 emit 后编译。打开 `device-image`
//! feature（发布）时，能用本机工具链编出设备镜像的设备（如今是 `metal`）额外带上二进制；
//! [`Device::status`] 说明每台设备这次走到了哪一步，没有镜像也只是"这一步没走"，不是配置错误。
//!
//! # 例子
//!
//! ```
//! let cpu = llm_prof::embedded("cpu").expect("内置 cpu 档案");
//! assert!(llm_prof::names().contains(&"cpu"));
//! assert!(llm_prof::prof::Profile::parse(cpu).is_ok());
//! assert_eq!(llm_prof::DEVICES.len(), llm_prof::names().len());
//!
//! let set = llm_prof::builtin("cpu", "c").expect("内置 cpu 片段");
//! assert!(set.get(llm_prof::Prim::Load).is_some());
//!
//! let device = llm_prof::device("metal").expect("内置设备");
//! assert_eq!(device.ext, "metal", "扩展名由档案的 emit 语言决定");
//! ```

#![deny(unsafe_code)]

/// 能力词汇表定义在 [`spec::atom`]（`spec` 在 `prof` 之上，词汇是双方共用的契约）；
/// `prof` 侧再转出键、需求与跨度，`prof::atom::*` 的历史路径保持不变。
pub mod atom {
    pub use spec::atom::*;
    pub use spec::key::KernelKey;
    pub use spec::need::{Attr, Need, Provide, Witness};
    pub use spec::span::Span;
}

pub use spec::inherent_names;

/// 声明一个闭集枚举：变体、`ALL`、名字表、`Display` 与固有 `parse` 由同一张表展开，
/// 名字因此只有一份（与 `spec::atom::vocabulary!` 同构，多出「附加列」一臂）。
///
/// 变体顺序即 `ALL` 顺序、即位集下标（`idx = self as usize`）；`as_str` 按 `idx` 查同序名字表，
/// 展开的查找与渲染都是表驱动（ccn = 1）。带附加列的形式（`ext: Cuda => ("cuda", "cu")`）
/// 额外生成一个 `const fn ext(self) -> &'static str`，供 `Emit::ext` 这类"名字之外的列"用。
///
/// ```ignore
/// closed_enum! {
///     /// 输出语言闭集。
///     Emit {
///         ext: Cuda => ("cuda", "cu"), Metal => ("metal", "metal"),
///     }
///     Skeleton {
///         Elementwise => "elementwise", Reduce => "reduce",
///     }
/// }
/// ```
#[macro_export]
macro_rules! closed_enum {
    (
        $(#[$doc:meta])*
        $name:ident { $extra:ident: $($variant:ident => ($text:literal, $value:literal)),* $(,)? }
    ) => {
        $crate::closed_enum! { $(#[$doc])* $name { $($variant => $text),* } }
        impl $name {
            /// 附加列取值（按闭集顺序）。
            #[must_use]
            pub const fn $extra(self) -> &'static str {
                const VALUES: &[&str] = &[$($value),*];
                // `Closed::idx` 是 trait 方法，不能出现在 `const fn` 里；闭集是 fieldless enum，
                // 判别值就是 `ALL` 的下标，`as usize` 与 `idx()` 同值。
                VALUES[self as usize]
            }
        }
    };
    (
        $(#[$doc:meta])*
        $name:ident { $($variant:ident => $text:literal),* $(,)? }
    ) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        // 变体的文档就是它的跨层名字：名字表只有一份，文档不该是第二份手抄。
        pub enum $name { $(#[doc = $text] $variant,)* }

        impl $crate::atom::Closed for $name {
            const ALL: &'static [Self] = &[$(Self::$variant),*];
            fn as_str(self) -> &'static str {
                const NAMES: &[&str] = &[$($text),*];
                NAMES[self.idx()]
            }
            fn idx(self) -> usize {
                self as usize
            }
        }
        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(self.as_str())
            }
        }
        impl $name {
            /// 闭集全量（见 [`crate::atom::Closed::ALL`]）。
            pub const ALL: &'static [Self] = <Self as $crate::atom::Closed>::ALL;
            /// 跨层名字（见 [`crate::atom::Closed::as_str`]）。
            #[must_use]
            pub fn as_str(self) -> &'static str {
                <Self as $crate::atom::Closed>::as_str(self)
            }
            /// 按名字解析（见 [`crate::atom::Closed::parse`]）。
            #[must_use]
            pub fn parse(name: &str) -> Option<Self> {
                <Self as $crate::atom::Closed>::parse(name)
            }
        }
    };
}

// ---- 词汇层 ------------------------------------------------------------------
//
// `model` / `sig` / `sch` / `tpl` / `einsum` 这些词汇定义在 `spec`，但下游一座座 crate
// 只依赖 `prof`。从这里转出，「算子声明 + 词汇」就沿同一个 crate 名到达，消费者不必为
// 读一个 `Model` 再加一条 `spec` 依赖。词汇侧的布局规格走 `prof::sch::layout`。
pub use spec::{dropout, formula, init, parse, rnd};
pub use spec::{einsum, model, operator, quant, same_bytes, sample, sch, sig, tpl};

pub mod device;
pub mod prof;
pub mod sign;
/// 保留词的历史路径：三个常量住在 [`sign`]，`stems::*` 兼容下游既有引用。
pub mod stems {
    pub use crate::sign::{EPILOGUE_STEM, MARKER, WITNESS_STEM};
}

pub use atom::Prim;
pub use device::{
    Builtin, Device, Image, Status, builtin, builtin_witness, device, devices, dir_of,
};
pub use fragment::{Fragment, FragmentSet, Layout, SIGN_MARKER, WITNESS_STEM, layer_root, load};

/// 构建期生成的档案表（`build.rs` 读 `src/<vendor>/<vendor>.toml` 展开成 `Table` 字面量）。
#[allow(clippy::all, missing_docs)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/embedded.rs"));
}

// 构建期生成的片段表：每台设备一个模块 + 四张聚合表。
// `include!` 在 crate 根上、不套 `mod`：套一层就要在 `lib.rs` 里重导出每台设备，那份 vendor
// 清单会重新变成第二份真相——生成表的意义正是让它只存在于 `src/` 的目录名里。
include!(concat!(env!("OUT_DIR"), "/tables.rs"));

pub use generated::{embedded, names};

mod fragment;

mod eval;
mod schema;
mod witness;

pub mod decl;
#[cfg(test)]
#[path = "../tests/embedded.rs"]
mod tests;

/// 构建期生成的算子声明表（`build.rs` 读 `src/ops/*.toml` 展开成 `Table` 字面量）。
#[allow(clippy::all, missing_docs)]
mod declared_generated {
    include!(concat!(env!("OUT_DIR"), "/declared.rs"));
}

pub use declared_generated::declared;

/// 全量算子清单里「名字 → 模板 stem」的那一半：由 [`build.rs`](../build.rs) 与声明表同一次
/// 扫描生成（`ops/*.toml` 的 `[op].stem`），语料是唯一事实源——加算子 = 加两个文件。
///
/// 这里不含任何内核类型：本 crate 只回答"有哪些算子、各自用哪份模板"。谁实现它们由
/// `nn::registry` 回答——那张表按名字与本表逐条对齐，错位在编译期就失败。
#[macro_export]
macro_rules! for_each_template {
    ($m:ident) => {
        include!(concat!(env!("OUT_DIR"), "/template_list.rs"))
    };
}

/// 模板表：算子名 → `ops/<stem>.k` 的正文，`include_str!` 在编译期读入，与
/// [`for_each_template!`] 同一次扫描生成。
mod templates {
    include!(concat!(env!("OUT_DIR"), "/templates_table.rs"));
}

pub use templates::TEMPLATES;

/// 算子目录：`src/ops/*.toml` 的声明表，与 `<stem>.k` 的模板正文。
///
/// 与 [`builtin`] 的档案表并列但不同：档案表答"这台机器能跑什么"，目录答"有哪些算子"。
pub mod catalogue {
    pub use crate::declared_generated::names;
    pub use crate::templates::TEMPLATES;

    /// 按名字取模板正文；名字不在清单里时返回 `None`。表按名升序（生成序），二分查找。
    pub fn template(name: &str) -> Option<&'static str> {
        let index = TEMPLATES
            .binary_search_by_key(&name, |(entry, _)| entry)
            .ok()?;
        Some(TEMPLATES[index].1)
    }

    /// 一次注册：一个算子的三件事——名字、语义表、是否可微（模板问 [`template`]）。
    ///
    /// 这是**注册表的值类型**：`nn::REGISTRY` 按 [`for_each_template!`](crate::for_each_template)
    /// 的同一顺序逐条填它，于是"有哪些算子"只有一份，`cutile` 与下游只认这个结构，不认识任何
    /// 内核类型。
    #[derive(Debug, Clone, Copy)]
    pub struct Registration {
        /// 算子名（与 `src/ops/<name>.toml` 的文件名一致）。
        pub name: &'static str,
        /// 代价、样本与标量参考。
        pub model: spec::model::Model,
        /// 训练反向图里是否可微：`true` 的算子必须配套一条反向公式。
        pub differentiable: bool,
    }

    /// 一份算子声明：编译期生成的声明表与同名 `<stem>.k` 模板。
    #[derive(Debug, Clone, PartialEq)]
    pub struct Op {
        name: &'static str,
        declaration: &'static diagnostic::data::Table,
        template: &'static str,
    }

    impl Op {
        /// 由名字、声明表与模板构造；名字取 `<name>.toml` 的文件名。
        pub fn new(
            name: &'static str,
            declaration: &'static diagnostic::data::Table,
            template: &'static str,
        ) -> Self {
            Self {
                name,
                declaration,
                template,
            }
        }

        /// 算子名（与 `<name>.toml` 的文件名一致）。
        pub fn name(&self) -> &'static str {
            self.name
        }

        /// 声明表：`[op]` / `[axes]` / `[needs]` / `[layouts]` / `[[variants]]`。
        pub fn declaration(&self) -> &'static diagnostic::data::Table {
            self.declaration
        }

        /// 模板正文：C 风格文本 + `{{插值}}` + `@原语` 槽位。
        pub fn template(&self) -> &'static str {
            self.template
        }
    }

    /// 按名字取一条内嵌算子；名字不在清单里时返回 `None`。
    pub fn named(name: &str) -> Option<Op> {
        let (entry, template) = TEMPLATES
            .binary_search_by_key(&name, |(entry, _)| entry)
            .ok()
            .map(|index| TEMPLATES[index])?;
        Some(Op::new(entry, crate::declared(entry)?, template))
    }

    /// 全部内嵌算子，顺序与 [`TEMPLATES`] 一致。
    pub fn all() -> Vec<Op> {
        TEMPLATES
            .iter()
            .filter_map(|(name, template)| Some(Op::new(name, crate::declared(name)?, template)))
            .collect()
    }
}
