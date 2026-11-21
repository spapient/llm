//! 档案层：设备能力档案（[`devices/`](../devices)）与它的 schema（[`prof`]）。
//!
//! 本 crate 是"设备差异只有数据"的容身之处：
//!
//! - [`devices/`](../devices) 里的 `<name>.toml` 是**内容**：存储层物理名、输出语言、工具链版本、
//!   执行单元及其可访问的存储层、每条能力区间，全部在里面；
//! - [`prof`] 是**唯一的 schema**：解析、闭集字段、见证与 `[units]` 一致性校验；
//! - [`atom`] 是**唯一的能力词汇表**：16 个原语、类型/存储位集、参数化签名 `Need`/`Provide`
//!   与属性键闭集 [`atom::AttrKey`]；
//! - [`device`] 是**设备行的形状**：一台设备这次构建带上的元数据、见证源码与可选的镜像，
//!   由各设备 crate（`cuda` / `cpu` / `opencl` / `metal` / `musa` / `rocm`）的 `build.rs` 生成，
//!   由 `prim` 合成一张表。它在这里而不是在 `prim` 里，理由只有依赖方向：设备 crate 依赖本层，
//!   `prim` 依赖设备 crate。
//!
//! 档案语料在**构建期**由 [`build.rs`](../build.rs) 用构建依赖 `toml` 解析 `devices/*.toml`，
//! 展开成 `diagnostic::data::{Table, Value}` 字面量写进 `$OUT_DIR/embedded.rs`，再由 [`embedded`]
//! / [`names`] 暴露（ADR 0022）。运行期只遍历这棵树：不读档案目录，也没有任何 TOML 解析器。
//!
//! `cutile` 只做机制（协商、降级、拼接、emit），因此本 crate 不反向依赖它：依赖方向
//! `cutile → prof`，而 `prof` 只依赖 `std`、同属基座的 [`diagnostic`]，以及错误类型
//! `diagnostic`（类别由产生错误的那一层给，见 `docs/ARCHITECTURE.md` 第三节）。
//!
//! 档案里 `witness`（L2 证据）指向片段层（`crates/prim`）：路径相对**片段层根**，仓库里是
//! `crates/prim`。本 crate 只在装载期做"文件在不在"的校验，不解析片段内容，也就不至于把两层
//! 耦合成一个 crate。
//!
//! # 例子
//!
//! ```
//! let cpu = llm_prof::embedded("cpu").expect("内置 cpu 档案");
//! assert!(llm_prof::names().contains(&"cpu"));
//! assert!(llm_prof::prof::Profile::parse(cpu).is_ok());
//! ```

#![deny(unsafe_code)]

pub mod atom;
pub mod device;
pub mod prof;

pub use device::{Device, Image, Status};

/// 构建期生成的档案表（`build.rs` 读 `devices/*.toml` 展开成 `Table` 字面量）。
#[allow(clippy::all, missing_docs)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/embedded.rs"));
}

pub use generated::{embedded, names};

#[cfg(test)]
mod tests;
