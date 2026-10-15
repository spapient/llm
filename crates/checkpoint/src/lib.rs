#![forbid(unsafe_code)]
//! Model and training-state persistence for the LLM runtime.
//!
//! `checkpoint` owns the checkpoint artifact: its [`manifest`], the in-memory
//! [`checkpoint`] container with load and save workflows, best-effort
//! atomic persistence ([`writeatomic`]), and compatibility validation
//! ([`checkcompat`], [`checkversion`]). It depends on `format` for the storage
//! dtype token mapping, `tensor` for the shape and dtype taxonomy, and `lib` for
//! resource limits. It MUST NOT own the training loop, optimizer update rules,
//! or the inference scheduler.
//!
//! # Artifact Layout
//!
//! A checkpoint is a single file: the 8-byte magic `LLMCKPT1`, a little-endian
//! `u64` manifest length, the UTF-8 JSON manifest, and the contiguous data
//! section. The manifest names every tensor with its dtype, shape, and byte
//! range, so the layout is self-describing and deterministic.
//!
//! # Failure Model
//!
//! [`checkpointerror`] keeps recoverable I/O failure separate from structural
//! incompatibility and from a malformed artifact; [`checkpointerror::isrecoverable`]
//! and [`checkpointerror::isincompatibility`] expose that separation without
//! variant matching. Writes stage to a same-directory temporary file and rename
//! onto the target, so an incomplete write never appears as a valid checkpoint.
//! All externally supplied bytes are validated with checked arithmetic before
//! use, and no runtime path uses `panic`, `unwrap`, or `expect`.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated. Module file names differ from the public types they
//! export because a module and a re-exported type share one name namespace at
//! the crate root.

mod atomicwrite;
mod compatibility;
mod crc32c;
mod error;
mod load;
mod manifests;
mod save;
mod training;

/// The checkpoint artifact version this build writes and accepts.
pub const CHECKPOINTVERSION: u32 = 1;

pub use atomicwrite::writeatomic;
pub use compatibility::{checkcompat, checkversion, tensorspec};
pub use crc32c::crc32c;
pub use error::checkpointerror;
pub use load::loadcheckpoint;
pub use manifests::{checkpoint, manifest, manifestentry};
pub use save::savecheckpoint;
pub use training::trainingstate;
