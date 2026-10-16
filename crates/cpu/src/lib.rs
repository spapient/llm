#![forbid(unsafe_code)]
//! CPU execution backend and numerical kernels for the LLM runtime.
//!
//! `cpu` is the first Layer 1 crate. It implements the frozen execution
//! contracts owned by [`api`] on the host CPU and adds no contract of
//! its own beyond backend policy.
//!
//! # Composition
//!
//! Three narrow pieces compose here:
//!
//! * [`cpubackend`] declares the backend identity, device, and capabilities
//!   required by the `api::backend` contract. It holds no execution logic.
//! * The kernels are concrete, stateless types, each implementing one frozen
//!   `api` kernel contract: [`cpurelu`], [`cpusilu`], [`cpugelutanh`], and
//!   [`cpusoftmax`] are unary (`api::kernel`); [`cpumatmul`], [`cpuadd`], and
//!   [`cpumul`] are binary (`api::binarykernel`); [`cpurmsnorm`] is weighted
//!   (`api::weightedkernel`); [`cpuembedding`] gathers (`api::gatherkernel`);
//!   [`cpurope`] and [`cpuattention`] carry per-call attributes.
//! * [`cpuexecutor`] owns a [`cpubackend`], enforces its declared element, byte,
//!   and rank budget, dispatches to a kernel selected by [`cpuop`], and is the
//!   crate's [`api::compute`] provider.
//!
//! Dispatch is static: the executor resolves [`cpuop`] to a concrete kernel
//! type with a `match`. No registry, trait object, or global mutable state is
//! introduced.
//!
//! # Scope
//!
//! The kernels are f32-only and cover the operator set the differentiation and
//! neural-network layers compose. Broadcast is not performed: binary operands
//! must have equal shapes, and batched matrix multiplication requires equal
//! leading batch dimensions. `attention` is the naive
//! O(seq^2 * heads * headdim) algorithm, not a fused or tiled kernel. `f16` and
//! `bf16` are unsupported; the backend declares only `f32` in its capabilities.
//!
//! # Errors
//!
//! Kernel failures use [`backenderror`](api::backenderror), the type fixed by
//! the kernel contracts. Backend-policy failures that
//! [`backenderror`](api::backenderror) does not model, such as a rejected
//! element, byte, or rank budget, are reported by [`cpuerror`].
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum, and
//! variant names are lowercase concatenated and constants are uppercase
//! concatenated. Module file names differ from the public types they export.

mod activation;
mod attention;
mod backend;
mod elementwise;
mod embedding;
mod error;
mod executor;
mod matmul;
mod normalization;
mod rotary;
mod scan;
mod softmax;
mod submission;
mod support;

pub use activation::{cpugelutanh, cpurelu, cpusilu};
pub use attention::cpuattention;
pub use backend::{CPUMAXRANK, cpubackend};
pub use elementwise::{cpuadd, cpumul};
pub use embedding::cpuembedding;
pub use error::cpuerror;
pub use executor::{cpuexecutor, cpuop};
pub use matmul::cpumatmul;
pub use normalization::cpurmsnorm;
pub use rotary::cpurope;
pub use scan::cpuscan;
pub use softmax::cpusoftmax;
pub use submission::cpusubmission;
