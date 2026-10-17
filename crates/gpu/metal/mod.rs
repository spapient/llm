//! Apple Metal GPU execution backend.
//!
//! `metal` is a Layer 1 backend: it implements the frozen [`api::backend`] and
//! [`api::compute`] contracts for `f32` operands on the default Metal device.
//! It owns no request state and holds no model weights; every operator call
//! creates, uses, and releases its own device resources.
//!
//! # Platform
//!
//! The module is compiled only on macOS; on every other target the crate
//! exposes no `metal` namespace and `cargo check --workspace --all-targets`
//! stays green.
//!
//! # Unsafe
//!
//! This backend is one of the carriers of the workspace's restricted `unsafe`
//! exemption (see docs/design.md §21.8). The crate root uses
//! `#![deny(unsafe_code)]`; `ffi` declares this backend's one allowance and
//! confines every platform call there, each preceded by a `// SAFETY:` comment.
//! `scripts/checkunsafe.sh` enforces the bounded site count and the SAFETY
//! comments mechanically.
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

pub use backend::{METALMAXHEADDIM, METALMAXRANK, metalbackend};
pub use executor::{metalexecutor, metalsubmission};
