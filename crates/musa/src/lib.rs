#![deny(unsafe_code)]
//! Moore Threads MUSA GPU execution backend for the LLM runtime.
//!
//! `musa` is a Layer 1 backend: it implements the frozen [`api::backend`] and
//! [`api::compute`] contracts for `f32` operands on a Moore Threads MUSA device.
//! It owns no request state and holds no model weights.
//!
//! # Platform
//!
//! The crate compiles to an empty library on every non-Linux target so that
//! `cargo check --workspace --all-targets` stays green everywhere. All items
//! are `#[cfg(target_os = "linux")]`.
//!
//! # Architecture (docs/musa-backend-plan.md §5)
//!
//! The MUSA backend converges on the workspace's unified GPU-backend terminal
//! state (docs/gpu-backend-unification.md): **build-time compilation, zero
//! runtime JIT, and the minimal unsafe surface**. Kernels are compiled at build
//! time by `mcc` into a fatbin that is embedded with `include_bytes!` and
//! loaded at run time through the `mu*` driver API — no runtime compiler, no
//! `muBLAS`/`muDNN` (matmul is served by a WMMA kernel on the Tensor Core,
//! softmax by the max-subtracting kernel shared with `cpu`/`metal`), so the
//! whole platform FFI collapses to a single driver-handoff function. The
//! registered unsafe site count is `musa = 2`: the `unsafe extern "C"` block
//! (mandatory under edition 2024) and the driver-handoff `unsafe` block
//! (docs/design.md §21.8).
//!
//! # Unsafe
//!
//! This crate is the third carrier of the workspace's restricted `unsafe`
//! exemption (docs/design.md §21.8): the crate root uses `#![deny(unsafe_code)]`;
//! the single module `ffi` (created in stage P1b) declares the crate's one
//! allowance and confines every platform call there, each preceded by a
//! `// SAFETY:` comment. `scripts/checkunsafe.sh` enforces the exact site count
//! and the SAFETY comments mechanically.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated.

#[cfg(target_os = "linux")]
mod backend;
#[cfg(target_os = "linux")]
mod error;

#[cfg(target_os = "linux")]
pub use backend::{MUSAMAXHEADDIM, MUSAMAXRANK, musabackend};
#[cfg(target_os = "linux")]
pub use error::musaerror;
