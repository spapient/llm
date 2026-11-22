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

pub mod atom;
pub mod device;
pub mod prof;
pub mod sign;
pub mod stems;

pub use atom::Prim;
pub use device::{Builtin, Device, Image, Status};
pub use builtin::{
    BUILTIN, DEVICES, ROOTS, WITNESS, builtin, builtin_witness, device, devices, dir_of,
};
pub use fragment::{Fragment, FragmentSet, SIGN_MARKER};
pub use layout::{Layout, WITNESS_STEM, layer_root, load};

/// 构建期生成的档案表（`build.rs` 读 `src/<vendor>/<vendor>.toml` 展开成 `Table` 字面量）。
#[allow(clippy::all, missing_docs)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/embedded.rs"));
}

/// 构建期生成的片段表：每台设备一个模块 + 四张聚合表。
///
/// `include!` 在 crate 根上、不套 `mod`：套一层就要在 `lib.rs` 里重导出每台设备，那份 vendor
/// 清单会重新变成第二份真相——生成表的意义正是让它只存在于 `src/` 的目录名里。
include!(concat!(env!("OUT_DIR"), "/tables.rs"));

pub use generated::{embedded, names};

mod builtin;
mod fragment;
mod layout;

mod eval;
mod key;
mod need;
mod provide;
mod schema;
mod span;
mod witness;

#[cfg(test)]
#[path = "../tests/embedded.rs"]
mod tests;
