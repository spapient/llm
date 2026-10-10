#![forbid(unsafe_code)]
//! Define-by-run reverse-mode differentiation for the LLM runtime.
//!
//! `autograd` is a Layer 1 crate. It records a forward computation as a tape of
//! nodes and then walks that tape in reverse to accumulate gradients. It owns
//! no backend of its own: every forward operator runs through an
//! [`api::compute`] provider, so the same graph code works with any backend and
//! this crate never depends on `cpu`.
//!
//! # Model
//!
//! A [`graph`] is the tape and the sole owner of every [`variable`]. A
//! [`variable`] holds its forward value, its accumulated gradient (initially
//! absent), whether it requires a gradient, and how it was produced. The
//! producer is a static [`op`] enum that names the input [`variableid`] handles,
//! so the tape has no dynamic dispatch and no captured heap state.
//!
//! Forward operators are inherent methods on [`graph`] (for example
//! [`graph::matmul`] and [`graph::softmax`]). Each forwards to the provider,
//! pushes one node, and returns its handle. [`graph::backward`] then seeds the
//! output gradient and walks the tape once in reverse creation order,
//! accumulating each operator's vector-Jacobian product into its inputs.
//!
//! # Recording control
//!
//! [`graph::nograd`] returns an RAII [`nogradscope`] that suppresses tape
//! recording for its lifetime. Outputs produced while suppressed are detached:
//! they keep their value but carry no producer, so a later reverse pass ignores
//! them. The scope restores the previous mode on drop, including on unwind.
//!
//! # Scope
//!
//! Operators are `f32` and cover matrix multiplication (rank two), elementwise
//! add and multiply, RMS normalization, softmax over the last axis, and the
//! `relu`, `silu`, and `gelutanh` activations. Batched matrix multiplication,
//! reduction, convolution, and half precision are unsupported.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum, and
//! variant names are lowercase concatenated and constants are uppercase
//! concatenated. Module file names differ from the public types they export
//! (for example `arena` exports [`graph`]) because a module and a re-exported
//! type share one name namespace at the crate root.

mod arena;
mod backward;
mod error;
mod gradient;
mod node;
mod nograd;
mod ops;
mod variables;

pub use arena::graph;
pub use error::autoggraderror;
pub use node::{op, variableid};
pub use nograd::nogradscope;
pub use variables::variable;
