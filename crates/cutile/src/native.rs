//! `cutile::native`：主机原生微内核的转发面（nn → cutile → prof）。
//!
//! 真正的微内核体、`target_feature` 门、特性探测与协商表全在 [`prof::native`]：一个设备/通道
//! 一个专门文件（`gemm8x8` 是 NEON/AVX2/scalar 的 8×8 寄存器块，`i8` 是 aarch64 dotprod
//! 4×4 `sdot` 块，`sme` 是 Apple SME 16×64 ZA 外积）。本模块只做一层**零成本转发** `pub use`，
//! 把「cutile 调用 prof」这条调用关系记录在 crate 图里，不复制任何实现：
//!
//! ```text
//! nn ── kernel 骨架 ──> cutile::native ──(pub use)──> prof::native
//! ```
//!
//! 为什么不做「配置化再组合」：这些微内核是**运行时按形状择优**的（[`gemm_negotiate`] 的
//! [`GEMM_KERNELS`] 表），不是编译期拼 DSL 片段；后者是 [`crate::cap`] 的职责（TOML 档案
//! 片段选择），两者机制不同、各自单点，转发面里不混。

// 协商面与公共方法（设备无关）：调用方只问「这块谁能算」，不写任何 `cfg(target_arch)` / 特性探测。
pub use prof::native::{align_up, gemm_negotiate, threads_override, with_panels};

// 编译期配置表与内核签名：协商表的条目类型（`Gemm` / `Native` / `GEMM_KERNELS`）随协商面一并转发。
pub use prof::native::{GEMM_KERNELS, Gemm, Native};

// 瓦片几何单一事实源（f32 8×8 / i8 4×4 / SME 16×64）与微内核体。
pub use prof::native::{f32, gemm8x8, i8, i8_geo, sme_geo};

// Apple SME 16×64 微内核：nightly + aarch64 才存在（`target_feature(enable = "sme")` 仍不稳定）。
#[cfg(all(nightly, target_arch = "aarch64"))]
pub use prof::native::sme;
