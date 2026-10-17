//! Moore Threads MUSA GPU execution backend.
//!
//! `musa` is a Layer 1 backend: it implements the frozen [`api::backend`] and
//! [`api::compute`] contracts for `f32` operands on a Moore Threads MUSA device.
//! It owns no request state and holds no model weights.
//!
//! # Platform
//!
//! The module is compiled only on Linux with the MUSA toolkit present (the
//! `musa_toolkit` cfg emitted by the crate's build script). On every other
//! host the crate exposes no `musa` namespace and
//! `cargo check --workspace --all-targets` stays green.
//!
//! # Architecture (docs/musa-backend-plan.md §5)
//!
//! The MUSA backend converges on the workspace's unified GPU-backend terminal
//! state (docs/gpu-backend-unification.md): **build-time compilation, zero
//! runtime JIT, and the minimal unsafe surface**. Kernels are compiled at build
//! time by `mcc` into ELF fatbins embedded with `include_bytes!` (see
//! [`registry`]) and loaded at run time through the `mu*` driver API — no
//! runtime compiler, no `muBLAS`/`muDNN` (matmul is served by an in-repo
//! Tensor-Core (TF32) kernel with the scalar FMA kernel kept as the reference
//! path, softmax by the max-subtracting kernel shared with `cpu`/`metal`),
//! so the whole platform FFI collapses to one driver-handoff function. The
//! registered unsafe site count is `musa = 2`: the `unsafe extern "C"` block
//! (mandatory under edition 2024) and the driver-handoff `unsafe` block
//! (docs/design.md §21.8). The handoff block is lazy by design: device open,
//! module load, function resolution, transfers, launch, and synchronisation
//! all live in the single `dispatch` block, so no second handoff site exists.
//!
//! # Unsafe
//!
//! This backend is one of the carriers of the workspace's restricted `unsafe`
//! exemption (docs/design.md §21.8): the crate root uses
//! `#![deny(unsafe_code)]`; `ffi` declares this backend's one allowance and
//! confines every platform call there, each preceded by a `// SAFETY:` comment.
//! `scripts/checkunsafe.sh` enforces the exact site count and the SAFETY
//! comments mechanically.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated.

mod backend;
mod error;
mod executor;
mod ffi;
mod registry;
mod source;

pub use backend::{MUSAMAXHEADDIM, MUSAMAXRANK, musabackend};
pub use error::musaerror;
pub use executor::{matmulpath, musaexecutor, musasession, musasubmission};
