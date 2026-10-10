#![deny(unsafe_code)]
//! Apple Metal GPU execution backend for the LLM runtime.
//!
//! `metal` is a Layer 1 backend: it implements the frozen [`api::backend`] and
//! [`api::compute`] contracts for `f32` operands on the default Metal device.
//! It owns no request state and holds no model weights; every operator call
//! creates, uses, and releases its own device resources.
//!
//! # Platform
//!
//! The crate compiles to an empty library on every non-macOS target so that
//! `cargo check --workspace --all-targets` stays green everywhere. All items
//! are `#[cfg(target_os = "macos")]`.
//!
//! # Unsafe
//!
//! This crate is one of the two carriers of the workspace's restricted
//! `unsafe` exemption (see docs/design.md §21.8). The crate root uses
//! `#![deny(unsafe_code)]`; the single module `ffi` declares
//! `#![allow(unsafe_code)]` and confines every platform call there, each
//! preceded by a `// SAFETY:` comment. `scripts/checkunsafe.sh` enforces the
//! bounded site count and the SAFETY comments mechanically.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated. Module file names differ from the public types they
//! export because a module and a re-exported type share one name namespace at
//! the crate root.

#[cfg(target_os = "macos")]
mod backend;
#[cfg(target_os = "macos")]
mod error;
#[cfg(target_os = "macos")]
mod executor;
#[cfg(target_os = "macos")]
mod ffi;
#[cfg(target_os = "macos")]
mod geometry;
#[cfg(target_os = "macos")]
mod source;

#[cfg(target_os = "macos")]
pub use backend::{METALMAXHEADDIM, METALMAXRANK, metalbackend};
#[cfg(target_os = "macos")]
pub use executor::metalexecutor;
