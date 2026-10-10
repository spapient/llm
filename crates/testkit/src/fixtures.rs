//! Deterministic tensor fixtures.
//!
//! Every fixture fixes both its shape and its filler values so that a test run
//! is reproducible byte for byte. Constructors return [`tensorerror`] rather
//! than panicking when a shape is rejected by `tensor`.

use ::tensor::{shape, tensor, tensorerror};

/// Builds a `rows` by `cols` tensor whose values are the sequence
/// `0, 1, 2, ...` in row-major order.
///
/// # Errors
///
/// Returns a [`tensorerror`] if the shape is rejected by `tensor`, for example
/// when the element count overflows.
pub fn sequential(rows: usize, cols: usize) -> Result<tensor<f32>, tensorerror> {
    let s = shape::new(vec![rows, cols])?;
    let n = s.getelementcount()?;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    tensor::new(s, data)
}

/// Builds a rank-1 tensor of `n` values formed by the ramp `0, 1, 2, ...`.
///
/// # Errors
///
/// Returns a [`tensorerror`] if the shape is rejected by `tensor`.
pub fn ramp(n: usize) -> Result<tensor<f32>, tensorerror> {
    let s = shape::new(vec![n])?;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    tensor::new(s, data)
}

/// Builds a rank-1 tensor holding `n` copies of `value`.
///
/// # Errors
///
/// Returns a [`tensorerror`] if the shape is rejected by `tensor`.
pub fn constant(n: usize, value: f32) -> Result<tensor<f32>, tensorerror> {
    tensor::new(shape::new(vec![n])?, vec![value; n])
}

/// Builds a `rows` by `cols` tensor filled with `value`.
///
/// # Errors
///
/// Returns a [`tensorerror`] if the shape is rejected by `tensor`.
pub fn matrix(rows: usize, cols: usize, value: f32) -> Result<tensor<f32>, tensorerror> {
    let s = shape::new(vec![rows, cols])?;
    let n = s.getelementcount()?;
    tensor::new(s, vec![value; n])
}
