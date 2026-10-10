//! Training orchestration: configuration, loss, gradient accumulation, stepping,
//! evaluation, and resume.
//!
//! `train` is a coordination layer. It owns the *workflow* of a training run — a
//! validated [`trainconfig`], a [`trainer`] that drives a loss/step loop over
//! explicit parameter and gradient tensors, gradient accumulation with correct
//! averaging, cross-entropy [`crossentropy`], an evaluation routine, and
//! checkpoint [`saveresume`]/[`loadresume`] orchestration — while the layers
//! below own the mechanisms: `tensor` owns shapes and buffers, `optim` owns the
//! update rules and their state, `autograd` owns differentiation, and
//! `checkpoint` owns persistence.
//!
//! ```
//! use lib::seed;
//! use optim::{sgd, sgdconfig};
//! use ::tensor::{shape, tensor};
//! use train::{trainconfig, trainer};
//!
//! // Minimize `w^2` with plain SGD; the gradient is supplied by the closure.
//! let params = vec![tensor::new(shape::new(vec![1]).unwrap(), vec![4.0_f32]).unwrap()];
//! let optimizer = sgd::new(sgdconfig::new(0.1, 0.0, 0.0).unwrap());
//! let config = trainconfig::single(0, seed::new(0)).unwrap();
//! let mut trainer = trainer::new(params, optimizer, config).unwrap();
//! let result = trainer
//!     .step(|params| {
//!         let value = params[0].getdata()[0];
//!         let grad = tensor::new(shape::new(vec![1]).unwrap(), vec![2.0 * value]).unwrap();
//!         Ok((value * value, vec![grad]))
//!     })
//!     .unwrap();
//! assert_eq!(result.getstep(), 1);
//! ```
//!
//! # How a step is driven
//!
//! [`trainer::step`] takes a loss function `FnMut(&[tensor<f32>]) -> Result<(f32,
//! Vec<tensor<f32>>), trainerror>`: it receives the current parameters and
//! returns one scalar loss per micro-batch together with one gradient per
//! parameter, in the optimizer's stable order. The trainer sums the gradients of
//! `accumulationsteps` micro-batches, averages them, applies the scheduled
//! learning rate, and calls [`optim::optimizer::update`] exactly once. The returned
//! [`stepresult`] reports the mean loss, the completed step count, and the rate
//! applied. This keeps `train` independent of any particular architecture: a
//! hand-written gradient and an `autograd`-computed one drive the identical
//! loop.
//!
//! # Failure and cancellation
//!
//! State advances only when an optimizer update completes. If the loss function
//! errors, returns a non-finite value, or violates the gradient contract, the
//! buffered window is discarded and neither the step counter nor the optimizer
//! counter moves. If a [`context::cancellation`] token attached through
//! [`trainer::withcancellation`] is set, the run stops at the next micro-batch
//! boundary with [`trainerror::cancelled`] and no state advance. The optimizer
//! itself validates before mutating, so a rejected update leaves parameters and
//! state unchanged.
//!
//! # Declared limitation: no backward path through `nn`/`models`
//!
//! `nn` and `models` are currently **forward-only**: they build and run a
//! forward pass but expose no autograd backward path, and `autograd` only has
//! vector-Jacobian products for `matmul`, `add`, `mul`, `rmsnorm`, `softmax`,
//! `relu`, `silu`, and `gelutanh`. Consequently this crate does **not** and
//! cannot train a full `models` Llama end to end: there is no differentiable
//! graph from a Llama forward pass back to its weights. `train` therefore does
//! not fabricate one. What it provides instead is the real orchestration that
//! such a path would plug into, and it demonstrates genuine loss decrease where
//! the autograd operators actually support it (a small composite of the ops
//! above, or a convex least-squares problem with a known optimum).
//!
//! What remains to make end-to-end Llama training work is a backward path
//! through the `nn`/`models` layers — autograd operators and VJPs for the
//! forward-only layers (normalization, attention, rotary, embedding, and the
//! model assembly) so that a forward pass produces a graph that can be reversed
//! to per-parameter gradients. Until that exists, the limitation is structural,
//! not a configuration of this crate.
//!
//! # Identifier style
//!
//! This crate follows the workspace's no-separator naming rule: type, trait,
//! enum, and function identifiers are all-lowercase concatenations
//! (`trainconfig`, `crossentropy`, `loadresume`), and constants are uppercase
//! concatenations (`CHECKPOINTVERSION`, `PARAMPREFIX`).

#![forbid(unsafe_code)]

mod error;
mod evaluation;
mod gradientaccumulation;
mod loss;
mod resume;
mod step;
mod trainconfigs;
mod trainers;

pub use error::trainerror;
pub use evaluation::{evalresult, evalsample, evaluate};
pub use gradientaccumulation::gradientaccumulator;
pub use loss::{crossentropy, lossresult};
pub use resume::{loadresume, resumestate, saveresume};
pub use step::stepresult;
pub use trainconfigs::trainconfig;
pub use trainers::trainer;
