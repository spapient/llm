#![forbid(unsafe_code)]
//! Autoregressive generation for the LLM runtime.
//!
//! `infer` is a Layer 3 crate. It owns the generation workflow: the
//! [`generationengine`] that orchestrates prefill and decode over an immutable
//! [`causallm`](models::causallm) model, the request-scoped mutable state
//! ([`generationrequest`]) with an explicit owner and lifecycle, the sampling
//! contract ([`samplingparams`], [`sampler`]), stop conditions
//! ([`stopcondition`], [`stopreason`]), the streaming contract
//! ([`streamsink`]), and a request-owned key/value cache ([`kvstate`]) that
//! implements the cache abstraction owned by `models`.
//!
//! # Ownership and isolation
//!
//! Model weights are immutable and hold no request state. All generation state
//! lives in a [`generationrequest`]: the prompt and accumulated context, the
//! sampler position, the stop criteria, and the [`kvstate`]. The
//! [`generationengine`] borrows the model and the compute provider and owns no
//! per-request state, so two requests can run concurrently against one model
//! without sharing mutable state (OW2). Cancellation is observed through a
//! caller-owned [`executioncontext`](context::executioncontext) handle and is
//! checked at every decode step.
//!
//! # Correctness first
//!
//! The primary tested path recomputes the full context on every step through
//! [`causallm::forward`](models::causallm::forward). The [`kvstate`] cache is a
//! request-owned structure implementing the `models` cache contract; it is not
//! yet wired into the forward pass, and no speedup is claimed for it. Wiring the
//! cache requires a cache-aware forward entry point and a measured baseline
//! (`rule.md` PF1/PF5); see `docs/decision-records/inference.md`.
//!
//! # Dependencies
//!
//! `infer` depends on `models` (the model and cache contracts), `api` (the
//! `compute` operator boundary), `tensor` (logits), `context` (cancellation and
//! the step checkpoint), and `lib` (the reproducible seed). It does not depend
//! on `train`, `cli`, `serve`, or a concrete backend.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated. Module file names differ from the public types they
//! export because a module and a re-exported type share one name namespace.

mod decode;
mod engine;
mod error;
mod generation;
mod kvcache;
mod prefill;
mod request;
mod sampling;
mod stopping;
mod stream;

pub use engine::generationengine;
pub use error::infererror;
pub use generation::generationresult;
pub use kvcache::kvstate;
pub use request::{generationparams, generationrequest};
pub use sampling::{sampler, samplingparams};
pub use stopping::{stopcondition, stopreason};
pub use stream::{collectingsink, streamsink};
