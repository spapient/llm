//! 稠密计算核：矩阵乘族（`gemm` / `gemv` / `matmul`）、激活（`embed`/`rmsnorm`/`rope`/`softmax`）、归一化（`norm`）、
//! 反量化（`dequant`）、逐元素（`elementwise` / `unary`）、归约（`reduce`）与 SIMD/SME 通道（`fast`）。

// 不稳定特性一律挂在 `cfg(nightly)` 上（由本 crate 的 `build.rs` 按 `rustc --version` 判定）：
// nightly 上走手写车道，stable 上走 `gemv::simd` 的标量退路，两边的能力差异只在
// 「向量化没向量化」，不在语义。这样 `rust-toolchain.toml` 不必把整个工作区钉在某个 nightly。
//
// 本 crate 只用到 `portable_simd`（`std::simd`，用在 `gemv::simd` 的解包点积）。它**不按目标门控**：
// 架构无关的 16 车道源码，aarch64 上落成 NEON、x86_64 上落成 SSE2 基线，两个目标编译同一份算术。
// SME 与 i8 dotprod 通道本体已迁入 `prof::native`，各自的 `sme` / `dotprod` `target_feature` 与
// 运行时探测门在 `prof` 的 `lib.rs` 按 nightly + aarch64 声明；本 crate 经 `cutile::native`
// 转发调用（nn → cutile → prof），不再持有这些门，也不直接 import `prof` 的内核层。

#[path = "kernel_dequant.rs"]
pub mod dequant;
#[path = "kernel_elementwise.rs"]
pub mod elementwise;
#[path = "kernel_embed.rs"]
pub mod embed;
#[path = "kernel_epilogue.rs"]
pub mod epilogue;
#[path = "kernel_fast.rs"]
pub mod fast;
#[path = "kernel_gemm.rs"]
pub mod gemm;
#[path = "kernel_gemv.rs"]
pub mod gemv;
#[path = "kernel_matmul.rs"]
pub mod matmul;
#[path = "kernel_norm.rs"]
pub mod norm;
#[path = "kernel_reduce.rs"]
pub mod reduce;
#[path = "kernel_rmsnorm.rs"]
pub mod rmsnorm;
#[path = "kernel_rope.rs"]
pub mod rope;
#[path = "kernel_scalar.rs"]
pub mod scalar;
#[path = "kernel_softmax.rs"]
pub mod softmax;
#[path = "kernel_unary.rs"]
pub mod unary;

// 一元 / 二元 / 归约 / 反量化 / GEMV / GEMM 骨架内的声明式子模块，re-export 到 crate 根
// 保持 `<crate>::<name>::MODEL` / `forward` 路径。
pub use dequant::{dequant_i2, dequant_i4, i8 as dequant_i8};
pub use elementwise::{
    add_f32, div_f32, max_f32, min_f32, mul_f32, remainder_f32, residual_add_f32, sub_f32,
};
pub use epilogue::apply as apply_epilogue;
pub use gemm::{f32 as gemm_f32, i2 as gemm_i2, i4 as gemm_i4, i8 as gemm_i8};
pub use gemv::{f32 as gemv_f32, i2 as gemv_i2, i4 as gemv_i4, i8 as gemv_i8};
pub use matmul::i8 as matmul_i8;
pub use reduce::{reduce_max_f32, reduce_prod_f32, reduce_sum_f32};
pub use scalar::Scalar;
pub use unary::{
    abs_f32, cos_f32, exp_f32, gelu_f32, hard_sigmoid_f32, leaky_relu_f32, log_f32, neg_f32,
    recip_f32, relu_f32, scale_f32, sigmoid_f32, sign_f32, sin_f32, silu_f32, sqrt_f32,
    square_f32, tanh_f32,
};
pub use {embed as embed_f32, rmsnorm as rmsnorm_f32, rope as rope_f32, softmax as softmax_f32};

// 契约层在 `prof`：内核与本 crate 的骨架宏都按 crate 根解析契约路径，这里把它们搬回来，
// 于是内核源码里写的是 `crate::model::…` / `crate::operator_via!`，而不是到处写依赖 crate 名。
pub use prof::same_bytes;
// `$crate::formula_cell!` / `$crate::operator_via!` 在骨架宏里按 crate 根解析，这里把 spec 导出的
// 宏搬回本 crate 根：内核里几十处 `#[model]` 展开都不必改路径。
pub use model::quant;
pub use prof::tpl::{formula, parse};
pub use prof::{einsum, model, operator, sample, sch, sig, tpl};
pub use prof::{formula_cell, operator_here, operator_via};
pub use sch::layout;

#[cfg(test)]
#[path = "../tests/kernel_prop_numeric.rs"]
mod prop_numeric;
