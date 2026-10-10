//! Neural-network layers.
//!
//! Each layer is a stateless-in-execution value that owns its parameters,
//! implements [`module`](crate::module) for name and parameter export, and
//! exposes an inherent `forward` whose arity matches its operation. Layers
//! compose only through the [`api::compute`](api::compute) contract, so they
//! never depend on a concrete backend.
//!
//! Module file names (`projection`, `lookup`, `normalize`, `rotate`,
//! `multihead`, `gated`) differ from the layer types they export (`linear`,
//! `embedding`, `rmsnorm`, `rotary`, `attention`, `mlp`) because a module and a
//! re-exported type share one name namespace at the parent.

mod gated;
mod lookup;
mod multihead;
mod normalize;
mod projection;
mod rotate;

pub use gated::mlp;
pub use lookup::embedding;
pub use multihead::attention;
pub use normalize::{DEFAULTEPS, rmsnorm};
pub use projection::linear;
pub use rotate::{DEFAULTBASE, rotary};

use ::tensor::{shape, tensor};

use crate::error::nnerror;

/// Returns the product of `dims`, treating an empty slice as one row.
///
/// # Errors
///
/// Returns a [`nnerror::tensor`] error when the product overflows `usize`.
pub(crate) fn leadingcount(dims: &[usize]) -> Result<usize, nnerror> {
    if dims.is_empty() {
        return Ok(1);
    }
    Ok(shape::new(dims.to_vec())?.getelementcount()?)
}

/// Builds a tensor from owned data and an explicit dimension list.
///
/// # Errors
///
/// Returns a [`nnerror::tensor`] error when the shape is empty, its element
/// count overflows, or the data length disagrees with the shape.
pub(crate) fn reshape(data: Vec<f32>, dims: Vec<usize>) -> Result<tensor<f32>, nnerror> {
    Ok(tensor::new(shape::new(dims)?, data)?)
}

/// Tiles a length-`c` vector into `rows` consecutive copies as `[rows, c]`.
///
/// This is the explicit broadcast step `api::compute` deliberately omits: the
/// layer expands the bias rather than relying on a hidden broadcasting rule.
///
/// # Errors
///
/// Returns a [`nnerror::tensor`] error when `rows * c` overflows or the output
/// shape is invalid.
pub(crate) fn expandrows(values: &[f32], rows: usize) -> Result<tensor<f32>, nnerror> {
    let c = values.len();
    let total = rows
        .checked_mul(c)
        .ok_or_else(|| nnerror::config("expandrows", "element count overflow"))?;
    let mut data = Vec::with_capacity(total);
    for _ in 0..rows {
        data.extend_from_slice(values);
    }
    reshape(data, vec![rows, c])
}
