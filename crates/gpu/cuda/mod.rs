//! NVIDIA CUDA GPU execution backend.
//!
//! `cuda` is a Layer 1 backend: it implements the frozen [`api::backend`] and
//! [`api::compute`] contracts for `f32` operands on GPU device 0. It owns no
//! request state and holds no model weights. An executor owns the retained
//! device context and the kernels compiled from the sources under `kernels/`,
//! while every operator call takes, uses, and releases its own execution stream
//! (docs/design.md §21.6), so no operand or result survives a call.
//!
//! # Platform
//!
//! The module is compiled only on Linux; on every other target the crate
//! exposes no `cuda` namespace and `cargo check --workspace --all-targets`
//! stays green.
//!
//! # Two mechanisms
//!
//! The operator set is served by two CUDA-specific mechanisms, chosen per
//! operator rather than per convenience (docs/design.md §21.2 decision 11):
//!
//! * **cuBLAS** serves `matmul` — the CUDA-specific fast path.
//! * **In-repo CUDA C++** compiled at run time through NVRTC serves everything
//!   else: `softmax`, `add`, `mul`, `silu`, `rmsnorm`, `rope`, `embedding`, and
//!   a fused `attention`. The NVRTC matmul kernel remains available through
//!   `cudaexecutor::newnvrtc` as the independent reference the cuBLAS path is
//!   validated against.
//!
//! `softmax` uses the in-repo max-subtracting kernel rather than cuDNN: the
//! kernel reuses the NVRTC launch site (no exemption growth), implements the
//! same formulation as `cpu`/`metal`, and avoids cuDNN's per-call handle
//! creation. Removing the cuDNN path reduced the backend from three `unsafe`
//! sites to two.
//!
//! # Unsafe
//!
//! This backend is one of the carriers of the workspace's restricted `unsafe`
//! exemption (see docs/design.md §21.8). The crate root uses
//! `#![deny(unsafe_code)]`; `ffi` declares this backend's one allowance and
//! confines every platform call there, each preceded by a `// SAFETY:` comment.
//! The registered sites are two, one per class of platform interaction: the
//! NVRTC kernel launch and the cuBLAS `gemm`. `scripts/checkunsafe.sh` enforces
//! the exact site count and the SAFETY comments mechanically.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated. Module file names differ from the public types they
//! export because a module and a re-exported type share one name namespace
//! inside the backend namespace.

mod backend;
mod error;
mod executor;
mod ffi;
mod source;

pub use backend::{CUDAMAXHEADDIM, CUDAMAXRANK, cudabackend};
pub use executor::{cudaexecutor, cudasubmission, matmulpath};
