//! The hardware-independent half of a backend's resource policy.
//!
//! Every backend declares the same bounds through [`capabilities`] and MUST
//! reject the same inputs for the same reason, so the rank bound, the element
//! budget, and the byte budget are decided once here instead of being restated
//! in each executor. Two executors that both call these predicates cannot drift
//! from one another; a third that computes the bounds itself can, which is why
//! the repository's cross-backend contract suite checks the observed error of
//! every backend against one shared vector list.
//!
//! Nothing here allocates, touches a device, or interprets an operator: the
//! predicates only decide whether a shape, or a gather output size, lies within
//! the bounds a [`capabilities`] value declares.

use ::tensor::shape;

use crate::{backenderror, capabilities};

/// Validates `s` against the rank, element, and byte bounds of `caps`.
///
/// The checks run in the order the error contract documents: rank first, then
/// the element count, then the byte size. `elementsize` is the storage width of
/// the element type being validated, in bytes, so an `f32` caller passes
/// `size_of::<f32>()`.
///
/// # Errors
///
/// Returns [`backenderror::resourceexhausted`] when the rank of `s` exceeds
/// [`capabilities::getmaxrank`], or when its element count or byte size is
/// strictly greater than the corresponding bound of
/// [`capabilities::getlimits`]; returns [`backenderror::sizeoverflow`] when the
/// element count or the byte size does not fit in `usize`.
pub fn checkbudget(caps: &capabilities, s: &shape, elementsize: usize) -> Result<(), backenderror> {
    if s.getrank() > caps.getmaxrank() {
        return Err(backenderror::resourceexhausted);
    }
    let elements = s
        .getelementcount()
        .map_err(|_| backenderror::sizeoverflow)?;
    checkelements(caps, elements, elementsize)
}

/// Validates a gather output of `rows * width` elements against `caps`.
///
/// A gather's output size comes from the index list rather than from any
/// operand shape, so it needs its own entry point rather than a shape.
///
/// # Errors
///
/// As [`checkbudget`], with the element count taken from `rows * width`.
pub fn checkgatherbudget(
    caps: &capabilities,
    rows: usize,
    width: usize,
    elementsize: usize,
) -> Result<(), backenderror> {
    let elements = rows.checked_mul(width).ok_or(backenderror::sizeoverflow)?;
    checkelements(caps, elements, elementsize)
}

/// Validates an already computed element count and its byte size.
fn checkelements(
    caps: &capabilities,
    elements: usize,
    elementsize: usize,
) -> Result<(), backenderror> {
    let limits = caps.getlimits();
    limits
        .checkelements(elements)
        .map_err(|_| backenderror::resourceexhausted)?;
    let bytes = elements
        .checked_mul(elementsize)
        .ok_or(backenderror::sizeoverflow)?;
    limits
        .checkbytes(bytes)
        .map_err(|_| backenderror::resourceexhausted)?;
    Ok(())
}
