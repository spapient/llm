//! The shared execution engine: one `compute` implementation for all backends.
//!
//! Every backend performs the same orchestration — validate the budget, derive
//! the launch geometry, call the device primitive, rebuild the result tensor —
//! so it lives here once. A backend implements [`platform`] with only its
//! device primitives. Where an operator has a genuinely better native form,
//! the backend overrides that primitive: MUSA's fused `rmsnormmatmul` and
//! `scan_cached`, and its native `attention_cached` / `rope_at` kernels, stay
//! in `musa` and are selected through the same trait.
//!
//! Recorded execution follows the same split: [`recordplatform`] adds the
//! per-device recording primitives and [`submission`] implements
//! [`api::submission`] once. MUSA keeps its own submission because its
//! pinned-input and persistent-session state is a different resource model.

pub mod backend;
pub mod error;
/// The shared execution engine, split by responsibility:
/// [`executor`](executor) (platform + compute), [`record`](record)
/// (recordplatform + submission), [`session`](session), [`backend`](backend)
/// (MAXRANK + devicespec + gpubackend), [`error`](error) (gpuerror + wrap),
/// and [`shape`](shape) (shared geometry helpers).
#[path = "executor.rs"]
pub mod exec;
pub mod record;
#[path = "session.rs"]
pub mod sess;
pub mod shape;

pub(crate) use backend::{MAXRANK, devicespec, gpubackend};
pub(crate) use error::{gpuerror, wrap};
pub(crate) use exec::{executor, platform, rmsnormmatmulshape};
pub(crate) use record::{recordplatform, submission};
pub(crate) use sess::session;
pub(crate) use shape::{concatdata, concatshape, embeddingoutshape, scanshape, scanstable};
