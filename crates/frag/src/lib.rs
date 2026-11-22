//! 六台设备的片段：`src/<设备>/<prim>.<ext>` 与 `witness.<ext>`，一台设备一个模块。
//!
//! **拥有**：本目录的片段文本与见证；`build.rs` 在编译期扫描每个设备目录，为它生成
//! `ROOT` / `FRAGMENTS` / `PRIMS` / `EPILOGUE` / `DEVICE`。**不拥有**：设备之间的合成
//! （`prim`）、能力区间（`prof`）、拼接与 emit（`cutile`）。
//!
//! 六台设备同在一个 crate，因为它们共用同一份生成器与同一套判据：加一台设备 = 加一个目录 +
//! `build.rs` 里一行。片段是**内容**，文件名即原语名：第一行的签名声明
//! `/* cutile-sign: <prim>/<impl> */` 在构建期与档案里的 `provides` 对齐。
//!
//! | 模块 | vendor | 输出语言 | 工具链 |
//! |---|---|---|---|
//! | [`cpu`] | `cpu` | C | `cc` |
//! | [`cuda`] | `nvidia` | CUDA C | `nvcc` |
//! | [`metal`] | `metal` | Metal | `xcrun metal` |
//! | [`musa`] | `musa` | CUDA C | `mcc` |
//! | [`opencl`] | `intel` | OpenCL | `clang` |
//! | [`rocm`] | `ascend` | AscendC | `ccec` |
//!
//! # 例子
//!
//! ```
//! let device = &llm_frag::cuda::DEVICE;
//! assert_eq!(device.vendor, "nvidia");
//! assert_eq!(device.ext, "cu");
//! assert!(
//!     device.entry(prof::atom::Prim::Load).is_some(),
//!     "每台设备都要有载入的入口"
//! );
//! ```

#![deny(unsafe_code)]

include!(concat!(env!("OUT_DIR"), "/tables.rs"));
