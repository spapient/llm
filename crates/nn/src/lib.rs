#![forbid(unsafe_code)]
//! Neural-network parameter containers and composed layers for the LLM runtime.
//!
//! `nn` is a Layer 2 crate. It owns two things: the containers that name and
//! carry parameters ([`parameter`], [`statedict`], [`module`]), and the
//! inference layers built from the [`api::compute`] operator set
//! ([`linear`], [`embedding`], [`rmsnorm`], [`rotary`], [`attention`], [`mlp`]).
//!
//! # Composition
//!
//! Every layer is generic over [`api::compute`], so a layer never depends on a
//! concrete backend and the same layer code runs on any provider. A forward
//! pass composes only the provider's primitive operators; where the primitive
//! set deliberately omits a behavior (bias broadcasting, for example), the
//! layer performs it explicitly rather than assuming a hidden rule. Layers
//! never introduce an operator that the provider does not own.
//!
//! # Parameters
//!
//! A [`parameter`] binds a leaf name to an owned tensor. A [`module`] collects
//! its parameters into a [`statedict`] under a hierarchical prefix, so a
//! composed layer such as [`mlp`] yields names like `gate.weight` without a
//! separate naming scheme per layer. Collection is the single place names and
//! values meet; the dictionary orders entries by name for deterministic
//! iteration.
//!
//! # Initialization
//!
//! Initialization is deterministic and driven by a [`lib::seed`]. Functions such
//! as [`initxavier`] and [`fillnormal`] each build their own SplitMix64 stream
//! with no global state, so two runs from the same
//! seed and the same call order produce identical parameters.
//!
//! # Scope
//!
//! Layers are `f32`-only and cover the Llama-family inference path: RMS
//! normalization, rotary position embeddings, grouped-query attention, and the
//! SwiGLU feed-forward network. Forward passes are the deliverable of this
//! stage; the differentiation path is owned by `autograd` and is not wired into
//! these layers yet.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum, and
//! variant names are lowercase concatenated and constants are uppercase
//! concatenated. Module file names differ from the public types they export.

mod dictionary;
mod error;
mod init;
mod layers;
mod modules;
mod parameters;

pub use dictionary::{joinname, statedict};
pub use error::nnerror;
pub use init::{fillnormal, fillone, filluniform, fillvalue, fillzero, initxavier};
pub use layers::{DEFAULTBASE, DEFAULTEPS};
pub use layers::{attention, embedding, linear, mlp, rmsnorm, rotary, ssm};
pub use modules::module;
pub use parameters::parameter;
