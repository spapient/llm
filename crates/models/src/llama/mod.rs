//! The Llama-family architecture.
//!
//! The architecture is split by responsibility: [`llamaconfig`] validates the
//! dimensions, [`llamaweights`] owns and validates the parameters,
//! [`llamablock`] holds the configuration-derived operators of one transformer
//! block, and [`llamamodel`] composes them into a [`causallm`](crate::causallm)
//! implementation.

mod block;
mod config;
mod model;
mod weights;

pub use block::llamablock;
pub use config::llamaconfig;
pub use model::{cachedsubmission, llamamodel};
pub use weights::{llamalayerweights, llamaweights};
