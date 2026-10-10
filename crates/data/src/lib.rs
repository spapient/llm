#![forbid(unsafe_code)]
//! Dataset, record, packing, batching, sampling, and data-loader abstractions.
//!
//! `data` owns the preparation pipeline that sits between a token source and a
//! training or inference consumer: it defines the [`dataset`] contract, the
//! [`record`] holder, an in-memory [`textdataset`], fixed-length [`packing`],
//! rectangular [`batch`] construction, a reproducible index [`sampler`], and a
//! lifecycle-explicit [`loader`] together with its [`loaderconfig`].
//!
//! The crate depends only on `lib` for [`tokenid`](lib::tokenid),
//! [`seed`](lib::seed), and [`limits`](lib::limits). It deliberately does not
//! depend on `tokenizer`: converting raw text to token identifiers is a
//! composition owned by a higher layer, so a [`textdataset`] is built from
//! already-tokenized documents. This keeps the dependency direction one-way and
//! avoids a low-level reverse dependency (see `docs/crate-boundaries.md`
//! section 5.9).
//!
//! # Ownership
//!
//! A [`loader`] borrows its dataset immutably and owns all epoch and batch
//! cursor state privately, so two loaders over the same dataset never share
//! request-scoped state. All buffers are released when the loader is dropped.
//!
//! # Resource limits
//!
//! Every input whose size or buffering can grow without bound is validated
//! against an explicit [`limits`](lib::limits) budget: dataset construction
//! bounds the record and token counts, packing bounds the total token count,
//! batch construction bounds the batch, sequence, and element counts, and the
//! loader bounds its per-batch buffers. Sampling bounds the number of indices.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated. Module file names are collision-free containers: a
//! module and a type re-exported from it cannot share one name, so the module
//! file is plural (`records`) while the exported type is singular (`record`),
//! matching the existing `lib` and `tensor` crates.

mod batches;
mod datasets;
mod error;
mod loaders;
mod packings;
mod records;
mod samplers;
mod textdatasets;

pub use batches::batch;
pub use datasets::dataset;
pub use error::dataerror;
pub use loaders::{loader, loaderconfig};
pub use packings::packing;
pub use records::record;
pub use samplers::sampler;
pub use textdatasets::textdataset;
