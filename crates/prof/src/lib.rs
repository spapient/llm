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
//! - [`device`](mod@device) 是**设备行的形状**：这次构建带上的元数据、见证源码、可选镜像与 [`Status`]。
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
// 主机原生内核（`native`）在 nightly 上走手写 SME/NEON 车道，stable 上走标量退路：与 nn 的
// `lib.rs` 同一套门控。`aarch64_unstable_target_feature` / `stdarch_aarch64_feature_detection`
// 只在 aarch64 上存在，x86_64 等目标直接声明就是 E0635，因此还要按架构门控。
#![cfg_attr(nightly, feature(portable_simd))]
#![cfg_attr(
    all(nightly, target_arch = "aarch64"),
    feature(aarch64_unstable_target_feature)
)]
#![cfg_attr(
    all(nightly, target_arch = "aarch64"),
    feature(stdarch_aarch64_feature_detection)
)]

mod annotate;

// ---- 词汇层：算子契约 ---------------------------------------------------------
//
// [`atom`] 是唯一的词汇表：原语闭集、存储位、数据类型、属性键、证据等级，以及需求/供给的
// 代数（[`atom::Need`] / [`atom::Provide`] / [`atom::Span`] / [`atom::KernelKey`]）。其余模块
// 是契约本身：[`sig`] 是 ABI 的唯一事实源，[`sch`] 是布局闭集，[`tpl`] 是模板词法，
// [`einsum`] 出代价表达式与求和语义，[`model`] 出语义模型，[`sample`] 出确定性样本，
// [`operator`] 把三者缝成一张算子契约，[`epilogue`] 是写回阶段的文法。
//
// 这些词汇与档案、片段同住一个 crate，理由不是"都跟设备有关"，而是**同一条文法**：档案声明
// "这台设备能拼出 `dot`"，片段声明"`dot` 在 `.metal` 里这么写"，模板与签名就是两边的共同语言。
// 拆成两个 crate 只会让这份语言多一层转发（`pub use spec::…`，一行代码也没少），
// 所以词汇随档案落地：`prof::atom` / `prof::sig` / `prof::model` 是它唯一的公开路径。
pub mod atom;
pub mod einsum;
pub mod epilogue;
pub mod model;
pub mod need;
pub mod operator;
pub mod sample;
pub mod sch;
pub mod sig;
pub mod tpl;

pub use model::quant;
pub use sample::{dropout, init, rnd};
pub use sch::layout;
pub use sig::same_bytes;
pub use tpl::{formula, parse};

#[cfg(test)]
#[path = "../tests/spec/support/lib_spec_tests.rs"]
mod spec_tests;

/// 主机原生微内核隔离子岛：f32 NEON/AVX2 8×8 块（`gemm8x8`）、i8 dotprod 4×4 `sdot` 块
/// （`i8`）、Apple SME 16×64 ZA 外积（`sme`）三条通道的瓦片本体、运行时探测与端口探针，
/// 以及编译期瓦片几何常量、面板暂存、线程数、对齐等设备无关公共件。`nn` 的 GEMM 骨架经
/// `prof::native::*` 借用瓦片尺寸、坐标与内核入口，`cutile` 的 tile DSL IR 经
/// `native::gemm_negotiate` 择优调用。与 `device` 同是「被上层消费、不反向依赖上层」的设备
/// 实现层；arch 门、特性探测与 `unsafe` 全部收在本岛，上层不写这些门。
pub mod native;

pub mod device;
pub mod prof;
pub mod sign;
/// 保留词的历史路径：三个常量住在 [`sign`]，`stems::*` 兼容下游既有引用。
pub mod stems {
    pub use crate::sign::{EPILOGUE_STEM, MARKER, WITNESS_STEM};
}

pub use atom::Prim;
pub use device::{
    Builtin, Device, Fragment, FragmentSet, Image, Layout, SIGN_MARKER, Status, WITNESS_STEM,
    builtin, builtin_witness, device, devices, dir_of, layer_root, load,
};

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

mod decode;
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

pub mod catalogue;
