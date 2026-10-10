#![forbid(unsafe_code)]
//! The optional HTTP application for the LLM runtime.
//!
//! `serve` is a Layer 4 application crate. It is a thin adapter around the
//! reusable generation workflow owned by [`infer`]: it parses an HTTP request,
//! maps it to a [`generationrequest`](infer::generationrequest), runs the
//! [`generationengine`](infer::generationengine) over an immutable model, and
//! maps the result back to an HTTP response.
//!
//! # Separation of transport and model errors
//!
//! The crate owns a single error enum, [`serveerror`]. Transport/protocol
//! failures (a malformed request line, oversized headers or body, an
//! unsupported media type or method, I/O) are distinct variants, and a model
//! execution failure is carried transparently as [`infererror`](infer::infererror).
//! A transport failure is therefore never reported as a model failure, and a
//! model failure is never collapsed into a transport variant (ER2). The mapping
//! to HTTP status lives only in [`httpresponse`], keeping protocol concerns at
//! the application boundary.
//!
//! # Request lifecycle and cancellation
//!
//! One HTTP request becomes one [`inferenceservice::handlewith`] call. The
//! service validates the transport input, applies admission control before any
//! expensive work, creates a request-scoped
//! [`executioncontext`](context::executioncontext), and hands its
//! [`cancellation`](context::cancellation) handle to the transport through the
//! `onstart` callback. The [`httpserver`] uses that seam to map a transport
//! deadline to a cancellation: it runs generation on a scoped worker and, when
//! the deadline elapses, cancels the request; the engine observes the
//! cancellation at its next checkpoint and returns
//! [`infererror::context`](infer::infererror::context), which maps to HTTP 408.
//! A cancelled or failed request never mutates shared weights; all mutable state
//! is request-scoped and released on drop (OW2, OW3).
//!
//! # Isolation and admission
//!
//! The service borrows an immutable model and compute provider and owns only
//! immutable configuration plus an atomic [admission](admissions) counter.
//! Request state lives in the per-request context, request, and sink, so
//! concurrent requests over one model cannot share mutable generation state.
//! Admission rejects a request with HTTP 429 before it reaches the model when
//! the configured concurrency ceiling is reached (IN4).
//!
//! # HTTP stack decision
//!
//! The transport is a minimal, standard-library-only HTTP/1.1 adapter over
//! [`std::net::TcpListener`]; no async runtime or HTTP framework is used. The
//! routing/handler/application layer is transport-neutral and independently
//! testable without opening a socket. The decision and the rejected `tokio`
//! alternative are recorded in `docs/design.md`.
//!
//! # Dependencies
//!
//! `serve` depends on `infer` (the generation workflow), `models` (the
//! [`causallm`](models::causallm) contract), `api` (the
//! [`compute`](api::compute) provider contract), `context` (execution controls),
//! and `lib` (identifiers and seeds). It does not depend on a concrete backend;
//! `cpu` is a dev-dependency used only by the integration tests.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated. Module file names differ from the public types they
//! export, because a module and a re-exported type share one name namespace.

mod admit;
mod error;
mod request;
mod response;
mod routes;
mod server;

pub use admit::{admission, admissions};
pub use error::serveerror;
pub use request::{httpmethod, httprequest, transportlimits};
pub use response::httpresponse;
pub use routes::{inferenceservice, serviceconfig};
pub use server::httpserver;
