#![forbid(unsafe_code)]
//! Optimization algorithms for the LLM runtime.
//!
//! `optim` is a Layer 2 crate. It owns the optimizer contract and its concrete
//! update rules ([`adamw`], [`sgd`]), gradient clipping ([`clipconfig`],
//! [`clipgradients`]), learning-rate schedules ([`lrschedule`]), and the
//! optimizer's own mutable state ([`optimizerstate`], [`statetensor`]). It
//! depends only on `tensor` and `lib` and MUST NOT own model architecture,
//! dataset loading, checkpoint file encoding, or training-loop orchestration.
//!
//! # Interface
//!
//! The [`optimizer`] trait operates on two parallel borrowed slices: the model's
//! parameters (`&mut [tensor<f32>]`) and their gradients (`&mut [tensor<f32>]`),
//! in a caller-defined and stable order. This deliberately decouples `optim`
//! from `autograd`: a gradient is any `tensor<f32>`, whether it came from a
//! reverse pass or from another source, so no graph representation is required
//! to apply an update.
//!
//! Optimizer state is keyed by the position of a parameter in that slice and
//! lives entirely inside the optimizer. It is distinct from the model
//! architecture, and [`optimizer::getstate`] / [`optimizer::restore`] map it
//! onto named [`statetensor`] values so a higher layer can persist it through
//! the checkpoint tensor contract.
//!
//! # Update rules
//!
//! [`adamw`] documents its exact bias-corrected, decoupled-weight-decay
//! equations; [`sgd`] documents its momentum recurrence and coupled weight
//! decay. [`clipgradients`] applies global-norm or elementwise value clipping
//! before an update when a [`clipconfig`] is attached to an optimizer.
//! [`lrschedule`] is a pure step-to-rate function that a caller applies through
//! [`optimizer::setlr`].
//!
//! # Scope
//!
//! Optimizers are `f32`-only. Nesterov momentum, dampening, and adaptive methods
//! other than AdamW are unsupported in this stage.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum, and
//! variant names are lowercase concatenated and constants are uppercase
//! concatenated. Module file names differ from the public types they export
//! because a module and a re-exported type share one name namespace at the
//! crate root.

mod adamws;
mod error;
mod gradientclip;
mod optimizers;
mod scheduler;
mod sgds;
mod state;

pub use adamws::{adamw, adamwconfig};
pub use error::opterror;
pub use gradientclip::{clipconfig, clipgradients};
pub use optimizers::optimizer;
pub use scheduler::lrschedule;
pub use sgds::{sgd, sgdconfig};
pub use state::{optimizerstate, statetensor};
