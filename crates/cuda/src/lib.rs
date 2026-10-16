#![deny(unsafe_code)]
//! NVIDIA CUDA GPU execution backend for the LLM runtime.
//!
//! `cuda` is a Layer 1 backend: it implements the frozen [`api::backend`] and
//! [`api::compute`] contracts for `f32` operands on GPU device 0. It owns no
//! request state and holds no model weights. An executor owns the retained
//! device context and the kernels compiled from the CUDA sources under
//! `kernels/`, while every operator
//! call takes, uses, and releases its own execution stream (docs/design.md
//! §21.6), so no operand or result survives a call.
//!
//! # Platform
//!
//! The crate compiles to an empty library on every non-Linux target so that
//! `cargo check --workspace --all-targets` stays green everywhere. All items
//! are `#[cfg(target_os = "linux")]`.
//!
//! # Three mechanisms
//!
//! The operator set is served by three CUDA-specific mechanisms, chosen per
//! operator rather than per convenience (docs/design.md §21.2 decision 11):
//!
//! * **cuBLAS** serves `matmul` — the CUDA-specific fast path.
//! * **cuDNN** serves `softmax`, whose last axis maps onto cuDNN's channel axis.
//! * **In-repo CUDA C++** compiled at run time through NVRTC serves everything
//!   else: `add`, `mul`, `silu`, `rmsnorm`, `rope`, `embedding`, and a fused
//!   `attention`. The NVRTC matmul kernel remains available through
//!   `cudaexecutor::newnvrtc` as the independent reference the cuBLAS path is
//!   validated against.
//!
//! `silu` deliberately does *not* use cuDNN's `CUDNN_ACTIVATION_SWISH`: that
//! entry point is a second `unsafe` construct for an elementwise operation the
//! NVRTC kernel family already covers at no additional exemption cost.
//!
//! # Unsafe
//!
//! This crate is one of the three carriers of the workspace's restricted
//! `unsafe` exemption (see docs/design.md §21.8). The crate root uses
//! `#![deny(unsafe_code)]`; the single module `ffi` declares the crate's one
//! allowance and confines every platform call there, each preceded by a
//! `// SAFETY:` comment. The registered sites are three, one per class of
//! platform interaction: the NVRTC kernel launch, the cuBLAS `gemm`, and the
//! cuDNN softmax. `scripts/checkunsafe.sh` enforces the exact site count and
//! the SAFETY comments mechanically.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated. Module file names differ from the public types they
//! export because a module and a re-exported type share one name namespace at
//! the crate root.

#[cfg(target_os = "linux")]
mod backend;
#[cfg(target_os = "linux")]
mod error;
#[cfg(target_os = "linux")]
mod executor;
#[cfg(target_os = "linux")]
mod ffi;
#[cfg(target_os = "linux")]
mod source;

#[cfg(target_os = "linux")]
pub use backend::{CUDAMAXHEADDIM, CUDAMAXRANK, cudabackend};
#[cfg(target_os = "linux")]
pub use executor::{cudaexecutor, cudasubmission, matmulpath};
