#![forbid(unsafe_code)]
//! Shared test fixtures, assertions, and reference implementations.
//!
//! `testkit` is development and test infrastructure. It MUST NOT become a
//! production dependency and it MUST NOT enter the production dependency
//! closure (see `AGENTS.md` section 3.1 and `docs/architecture.md`).
//!
//! It provides four narrow pieces:
//!
//! * [`fixtures`] builds deterministic tensor inputs so tests are reproducible.
//! * [`assertions`] compares numerical results within a documented tolerance.
//! * [`mod@reference`] holds independent scalar implementations used to
//!   cross-check optimized kernels.
//! * [`probe`] probes GPU availability with the per-backend environment gate.
//! * [`tinymodel`] is reserved for a tiny deterministic model fixture.
//!
//! The library depends only on `tensor`; the cross-crate integration tests in
//! `tests/` bring in `lib`, `api`, and `cpu` to exercise the full stack through
//! public interfaces.

pub mod assertions;
pub mod fixtures;
pub mod probe;
pub mod reference;
pub mod tinymodel;
