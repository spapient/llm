#![deny(unsafe_code)]
//! Native GPU execution backends: Metal, CUDA, and MUSA.
//!
//! `gpu` is a Layer 1 crate that owns the three platform backends as modules
//! and exposes each one under its own namespace — `gpu::metal`, `gpu::cuda`,
//! `gpu::musa`. Every module implements the frozen [`api::backend`] and
//! [`api::compute`] contracts for `f32` operands and holds no request state and
//! no model weights.
//!
//! Each backend keeps its own device layer, kernel sources, build pipeline, and
//! `unsafe` island. The operator *orchestration* is shared: [`engine`] holds the
//! single [`api::compute`] implementation and a `platform` trait of device
//! primitives, so a backend writes only the launch calls its device needs.
//! Where a backend has a genuinely better native form — MUSA's fused
//! `rmsnormmatmul`, `scan_cached`, and cached attention — it overrides that
//! primitive instead of following the shared path. Kernel sources, tensor
//! construction, and the shape predicates that the backends must classify
//! identically are shared alongside it.
//!
//! # Platform
//!
//! The whole crate compiles to an empty library on targets that own no backend,
//! so `cargo check --workspace --all-targets` stays green everywhere.
//!
//! # Unsafe
//!
//! This crate is one of the carriers of the workspace's restricted `unsafe`
//! exemption (docs/design.md §21.8). The crate root uses
//! `#![deny(unsafe_code)]`; each backend's `ffi` module declares the crate's
//! three allowances — one per backend — and confines every platform call there,
//! each preceded by a `// SAFETY:` comment. The registered site counts are
//! metal = 3, cuda = 2, musa = 2. `scripts/checkunsafe.sh` enforces the exact
//! counts and the SAFETY comments mechanically.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated.

#[cfg(target_os = "linux")]
pub mod cuda;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(all(target_os = "linux", musa_toolkit))]
pub mod musa;

#[doc(hidden)]
pub mod engine;

mod catalog;
mod ffi;
