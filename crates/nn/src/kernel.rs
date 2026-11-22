//! 稠密计算核：矩阵乘族（`gemm` / `gemv` / `matmul`）、激活（`activation`）、归一化（`norm`）、
//! 反量化（`dequant`）、逐元素（`elementwise` / `unary`）、归约（`reduce`）与 SIMD/SME 通道（`fast`）。

// 不稳定特性一律挂在 `cfg(nightly)` 上（由本 crate 的 `build.rs` 按 `rustc --version` 判定）：
// nightly 上走手写车道，stable 上走 `gemv/simd/fallback.rs` 的等价退路，两边的能力差异只在
// 「向量化没向量化」，不在语义。这样 `rust-toolchain.toml` 不必把整个工作区钉在某个 nightly。
//
// 三个特性各自还有额外的门：
// - `aarch64_unstable_target_feature`：放行 `#[target_feature(enable = "sme")]`。
// - `stdarch_aarch64_feature_detection`：放行运行时探测 `is_aarch64_feature_detected!("sme")`。
//   上面这两个特性名**只在 aarch64 上存在**：x86_64 等其它目标根本不认识，直接声明会 E0635
//   （unknown feature）。特性门不是惰性的，名字在编译目标上不认识就是硬错误，因此还要按架构门控。
// - `portable_simd`：`std::simd`，用在 `fast::sme::pack_a` 的分块转置（只存在于 aarch64）与
//   `gemv::simd` 的解包点积。后者**不按目标门控**，因为它是架构无关的 16 车道源码：aarch64 上
//   落成 NEON、x86_64 上落成 SSE2 基线，两个目标编译的是同一份算术。

pub mod activation;
pub mod dequant;
pub mod elementwise;
pub mod epilogue;
pub mod fast;
pub mod gemm;
pub mod gemv;
pub mod matmul;
pub mod norm;
pub mod reduce;
pub mod scalar;
pub mod unary;

// 一元 / 二元 / 归约 / 反量化 / GEMV / GEMM 骨架内的声明式子模块，re-export 到 crate 根
// 保持 `<crate>::<name>::MODEL` / `forward` 路径。
pub use activation::{
    embed as embed_f32, rmsnorm as rmsnorm_f32, rope as rope_f32, softmax as softmax_f32,
};
pub use dequant::{dequant_i2, dequant_i4, i8 as dequant_i8};
pub use elementwise::{add_f32, div_f32, max_f32, min_f32, mul_f32, residual_add_f32, sub_f32};
pub use epilogue::apply as apply_epilogue;
pub use gemm::{f32 as gemm_f32, i2 as gemm_i2, i4 as gemm_i4, i8 as gemm_i8};
pub use gemv::{f32 as gemv_f32, i2 as gemv_i2, i4 as gemv_i4, i8 as gemv_i8};
pub use matmul::i8 as matmul_i8;
pub use reduce::{reduce_max_f32, reduce_sum_f32};
pub use scalar::Scalar;
pub use unary::{
    exp_f32, gelu_f32, hard_sigmoid_f32, leaky_relu_f32, log_f32, relu_f32, scale_f32, sigmoid_f32,
    silu_f32, tanh_f32,
};

// 契约层在 `spec`：内核与本 crate 的骨架宏都按 crate 根解析契约路径，这里把它们搬回来，
// 于是内核源码里写的是 `crate::model::…` / `crate::operator_via!`，而不是到处写依赖 crate 名。
pub use llm_spec::same_bytes;
// `$crate::formula_cell!` / `$crate::operator_via!` 在骨架宏里按 crate 根解析，这里把 spec 导出的
// 宏搬回本 crate 根：内核里几十处 `#[model]` 展开都不必改路径。
pub(crate) use llm_spec::tpl::guard;
pub use llm_spec::tpl::{formula, parse};
pub use llm_spec::{einsum, model, operator, sample, sch, sig, tpl};
pub use llm_spec::{formula_cell, operator_here, operator_via};
pub use model::quant;
pub use sch::layout;

#[cfg(test)]
#[path = "../tests/prop_numeric.rs"]
mod prop_numeric;
