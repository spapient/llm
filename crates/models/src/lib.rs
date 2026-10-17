#![forbid(unsafe_code)]
//! Concrete model architectures for the LLM runtime.
//!
//! `models` is a Layer 3 crate. It owns the causal-language-model contract
//! ([`causallm`]), the key/value cache contract an inference layer implements
//! ([`kvcache`]), the architecture registry ([`architecture`],
//! [`getarchitecture`]), and the Llama-family implementation
//! ([`llamaconfig`], [`llamaweights`], [`llamablock`], [`llamamodel`]).
//!
//! # Composition
//!
//! A model is assembled from the `nn` layers and reaches the backend only
//! through [`api::compute`]. The model owns its weights immutably and holds no
//! request-scoped state, so one model value serves independent requests without
//! sharing mutable state (OW2).
//!
//! # Weight ownership
//!
//! [`llamaweights`] owns every parameter as an `nn` layer, and the model borrows
//! that value. Weights are validated against the configuration at construction;
//! a shape that disagrees is reported rather than discovered during a forward
//! pass.
//!
//! # KV cache
//!
//! [`kvcache`] is defined here because `models` owns the cache abstraction it
//! will consume, while the request-scoped buffer is implemented by the
//! inference layer. This stage defines the contract only; the deterministic
//! forward pass recomputes the full sequence.
//!
//! # Scope
//!
//! `f32`-only, single architecture (Llama: RMSNorm, RoPE, grouped-query
//! attention, SwiGLU), no KV cache, no sampling. The registry maps names to
//! constructors; parsing a serialized architecture name is owned by the format
//! layer and is not wired here yet.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum, and
//! variant names are lowercase concatenated and constants are uppercase
//! concatenated. Module file names differ from the public types they export.

mod block;
mod causal;
mod config;
mod error;
mod model;
mod registry;
mod weights;

pub use block::llamablock;
pub use causal::{causallm, devicekvcache, kvcache};
pub use config::llamaconfig;
pub use error::modelerror;
pub use model::{cachedsubmission, llamamodel};
pub use registry::{architecture, getarchitecture};
pub use weights::{llamalayerweights, llamaweights};
