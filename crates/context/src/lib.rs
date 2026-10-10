#![forbid(unsafe_code)]
//! Execution controls for the LLM runtime.
//!
//! `context` owns the explicitly defined, request-scoped controls an execution
//! needs: an immutable [`executionconfig`], a [`cancellation`] handle with
//! explicit checkpoints, a [`memorybudget`] with byte-granular RAII accounting,
//! a bounded [`workergroup`] coordination primitive, and a narrow [`metrics`]
//! interface. It owns no model architecture, no sampling algorithm, and no
//! universal application-state container, and it introduces no global mutable
//! state.
//!
//! # Dependencies
//!
//! `context` depends on `lib` for [`requestid`](lib::requestid), the
//! per-execution identity. It uses only the Rust standard library for
//! synchronization.
//!
//! # Concurrency
//!
//! Every shared field is an atomic or a mutex with documented ordering. There is
//! no global lock and no static mutable state. Cancellation propagates through a
//! `Release`/`Acquire` atomic pair; memory accounting uses compare-and-update
//! semantics so concurrent reservations cannot exceed the configured capacity.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, method, and function names are lowercase concatenated, and constants
//! are uppercase concatenated. Three module file names differ from the public
//! types they export — `cancel` exports [`cancellation`], `budget` exports
//! [`memorybudget`], and `metric` exports [`metrics`] — because a module and a
//! re-exported type share one name namespace and cannot both use the same name.

mod budget;
mod cancel;
mod error;
mod execution;
mod metric;
mod worker;

pub use budget::{memorybudget, memoryreservation};
pub use cancel::cancellation;
pub use error::contexterror;
pub use execution::{executionconfig, executioncontext};
pub use metric::{collectingmetrics, metrics, noopmetrics};
pub use worker::{workererror, workergroup};
