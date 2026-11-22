//! 片段层：某台设备上某个原语的具体写法，以及它的见证源码。
//!
//! **拥有**：[`Fragment`] / [`FragmentSet`]（装载后的片段）、[`Layout`]（片段层在哪）、
//! [`Builtin`]（编译期内嵌表）。**不拥有**：片段文本本身（`llm-frag` 的六个设备模块：
//! `cpu` / `cuda` / `metal` / `musa` / `opencl` / `rocm`）、`Device` / `Image` / `Status` 与设备能力区间
//! （`prof`）、拼接与 emit（`cutile`）。**依赖**：`frag`、`prof`（原语闭集与档案）、
//! `diagnostic`（错误类别由本层给，装载失败因此带类别而不是字符串）。**被依赖**：`cutile`。
//!
//! 这一层是**机制**：怎么找、怎么装载、怎么校验签名声明。一台设备的某条原语怎么写，全部是
//! `crates/frag/src/<设备>/` 里的 `<prim>.<ext>` 文本，本 crate 一行片段都没有。于是"加一台设备"是加
//! 一个设备目录与一行 `build.rs`，再加 `builtin` 里四行（四张表各一行）；机制代码不动，片段也不会与拼接器
//! 挤在同一个 crate。
//!
//! 这个文件只是**门面**：三个模块各承担一个问题，公开项在这里统一导出，路径与模块归属无关。
//!
//! | 问题 | 模块 | 公开项 |
//! |---|---|---|
//! | 片段是什么 | `fragment` | [`Fragment`]、[`FragmentSet`]、[`SIGN_MARKER`] |
//! | 片段在哪 | `layout` | [`Layout`]、[`load`]、[`layer_root`]、[`WITNESS_STEM`] |
//! | 内置了什么 | `builtin` | [`BUILTIN`]、[`WITNESS`]、[`DEVICES`]、[`ROOTS`]、[`Builtin`]、[`builtin`]、[`device`]、[`devices`]、[`dir_of`]、[`builtin_witness`] |
//!
//! # 签名声明
//!
//! 每段片段的第一行是**签名声明** `/* cutile-sign: <prim>/<impl> */`：同一原语可以有多条签名
//! （`dot` 的 `dp4a` 与标量 `fma`），片段只说"这段文本写的是哪一条"，"这台设备有没有这条签名"
//! 仍由档案回答。两边的对齐由 `cutile` 在装载期判定，所以"协商选中 `dp4a`、文本写的却是 `fma`"
//! 不会晚到构建期、更不会静默用错算法。解析规则只有一份，住在 [`prof::sign`]。
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
//! let set = llm_prim::builtin("cpu", "c").expect("内置 cpu 片段");
//! assert_eq!(set.vendor(), "cpu");
//! assert!(set.get(llm_prim::Prim::Load).is_some());
//!
//! let device = llm_prim::device("metal").expect("内置设备");
//! assert_eq!(device.ext, "metal", "扩展名由档案的 emit 语言决定");
//! assert!(device.entry(llm_prim::Prim::Dot).is_some(), "每个原语都有入口");
//!
//! let layout = llm_prim::Layout::builtin();
//! assert!(layout.witness("cpu/witness.c").is_some(), "内置见证在 frag 的设备目录里");
//! ```

#![deny(unsafe_code)]

mod builtin;
mod fragment;
mod layout;

pub use builtin::{
    BUILTIN, Builtin, DEVICES, ROOTS, WITNESS, builtin, builtin_witness, device, devices, dir_of,
};
pub use fragment::{Fragment, FragmentSet, SIGN_MARKER};
pub use layout::{Layout, WITNESS_STEM, layer_root, load};
pub use prof::atom::Prim;
pub use prof::{Device, Image, Status};

#[cfg(test)]
#[path = "../tests/prim_tests.rs"]
mod tests;
