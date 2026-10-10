use crate::dtype;
use thiserror::Error;

/// Failures produced while constructing or validating tensor metadata.
///
/// Every variant carries the values needed to report the exact rejection, so a
/// caller can react without re-deriving them. The type is owned by the tensor
/// crate and describes only tensor metadata invariants.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum tensorerror {
    /// A shape was created from an empty dimension list.
    #[error("a shape must have at least one dimension")]
    emptyshape,

    /// The product of the shape dimensions does not fit in `usize`.
    #[error("element count overflows usize for dimensions {dims:?}")]
    elementcountoverflow {
        /// The dimensions whose product overflowed.
        dims: Vec<usize>,
    },

    /// The element count exceeds the configured element budget.
    #[error("requested element count {requested} exceeds limit {limit}")]
    elementcountlimitexceeded {
        /// The requested element count.
        requested: usize,
        /// The configured maximum element count.
        limit: usize,
    },

    /// The supplied element data length differs from the shape element count.
    #[error("data length {actual} does not match expected element count {expected}")]
    lengthmismatch {
        /// The expected element count.
        expected: usize,
        /// The actual element data length.
        actual: usize,
    },

    /// A layout stride is invalid for its dimension.
    #[error("invalid stride {stride} for dimension {dim}")]
    invalidstride {
        /// The zero-based dimension index.
        dim: usize,
        /// The offending stride.
        stride: usize,
    },

    /// An element type does not match the required storage type.
    #[error("dtype mismatch: expected {dtype:?}")]
    dtypemismatch {
        /// The storage type that was expected.
        dtype: dtype,
    },
}
