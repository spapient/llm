#![forbid(unsafe_code)]
//! Foundational shared types for the LLM runtime.
//!
//! `lib` owns identifiers, precision descriptors, reproducible seeds,
//! resource limits, and the shared limit-validation error type. It depends only
//! on the Rust standard library and `thiserror` and MUST NOT depend on any other
//! workspace crate.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated.
//!
//! # Scope
//!
//! This crate intentionally provides no default resource limits. Every bound is
//! supplied explicitly by the caller so that no undocumented limit is invented.

// The private module identifiers `prec`, `seeds`, and `limit` intentionally
// differ from the public types `precision`, `seed`, and `limits` they contain:
// a module and a re-exported type share the type namespace and cannot both use
// the same name at the crate root.
mod error;
mod limit;
mod prec;
mod request;
mod seeds;
mod token;

pub use error::limiterror;
pub use limit::limits;
pub use prec::precision;
pub use request::requestid;
pub use seeds::seed;
pub use token::tokenid;
