//! `nvidia` 设备的片段：闭集里每条原语在这台设备上怎么写，以及它们共同的见证源码。
//!
//! **拥有**：本目录的 `<prim>.cu` 与 `witness.cu`；`build.rs` 在编译期扫描它们，生成
//! `FRAGMENTS` / `PRIMS` / `DEVICE` 三张表。**不拥有**：设备之间的合成（`prim`）、能力区间
//! （`prof`）、拼接与 emit（`cutile`）。
//!
//! 一台设备一个 crate，因为片段是**内容**：加一台设备 = 加一个这样的 crate，机制
//! （`prof::codegen`）一行不改。文件名即原语名，第一行的签名声明
//! `/* cutile-sign: <prim>/<impl> */` 在构建期与档案里的 `provides` 对齐。
//!
//! 输出语言是 CUDA C，工具链是`nvcc` 工具链。
//!
//! # 例子
//!
//! ```
//! let device = &llm_cuda::DEVICE;
//! assert_eq!(device.vendor, "nvidia");
//! assert_eq!(device.ext, "cu");
//! assert!(
//!     device.entry(prof::atom::Prim::Load).is_some(),
//!     "每台设备都要有载入的入口"
//! );
//! ```

#![deny(unsafe_code)]

/// 本 crate 的根目录：片段与见证就在这个目录下，`prim` 按它读磁盘。
pub const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src");

include!(concat!(env!("OUT_DIR"), "/tables.rs"));

/// 片段层与档案的对账：签名、见证入口、能力供给三者必须闭合。
#[cfg(test)]
#[path = "../tests/fragments.rs"]
mod tests;
