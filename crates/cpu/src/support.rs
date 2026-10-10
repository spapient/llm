use ::tensor::{shape, tensor};
use api::backenderror;

/// Builds an owned output tensor from a shape and its computed data.
///
/// The output of every kernel is derived from an existing input view of the
/// same shape, so `data.len()` equals the shape's element count and the count
/// cannot overflow. A construction failure therefore indicates a broken
/// internal invariant; it is reported as a backend execution failure rather
/// than silently ignored or discarded.
pub(crate) fn finish(s: &shape, data: Vec<f32>) -> Result<tensor<f32>, backenderror> {
    tensor::new(s.clone(), data).map_err(|_| backenderror::executionfailed)
}
