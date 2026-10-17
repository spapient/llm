use crate::dtype;

/// Failures produced while constructing or validating tensor metadata.
///
/// Every variant carries the values needed to report the exact rejection, so a
/// caller can react without re-deriving them. The type is owned by the tensor
/// crate and describes only tensor metadata invariants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum tensorerror {
    /// A shape was created from an empty dimension list.
    emptyshape,

    /// The product of the shape dimensions does not fit in `usize`.
    elementcountoverflow {
        /// The dimensions whose product overflowed.
        dims: Vec<usize>,
    },

    /// The element count exceeds the configured element budget.
    elementcountlimitexceeded {
        /// The requested element count.
        requested: usize,
        /// The configured maximum element count.
        limit: usize,
    },

    /// The supplied element data length differs from the shape element count.
    lengthmismatch {
        /// The expected element count.
        expected: usize,
        /// The actual element data length.
        actual: usize,
    },

    /// A layout stride is invalid for its dimension.
    invalidstride {
        /// The zero-based dimension index.
        dim: usize,
        /// The offending stride.
        stride: usize,
    },

    /// An element type does not match the required storage type.
    dtypemismatch {
        /// The storage type that was expected.
        dtype: dtype,
    },

    /// A byte view is not aligned to the element type's alignment.
    viewmisaligned {
        /// The required alignment, in bytes.
        align: usize,
    },

    /// A supplied layout is not valid for the supplied shape.
    invalidlayout,
}

impl std::fmt::Display for tensorerror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::emptyshape => f.write_str("a shape must have at least one dimension"),
            Self::elementcountoverflow { dims } => {
                write!(f, "element count overflows usize for dimensions {dims:?}")
            }
            Self::elementcountlimitexceeded { requested, limit } => write!(
                f,
                "requested element count {requested} exceeds limit {limit}"
            ),
            Self::lengthmismatch { expected, actual } => write!(
                f,
                "data length {actual} does not match expected element count {expected}"
            ),
            Self::invalidstride { dim, stride } => {
                write!(f, "invalid stride {stride} for dimension {dim}")
            }
            Self::dtypemismatch { dtype } => write!(f, "dtype mismatch: expected {dtype:?}"),
            Self::viewmisaligned { align } => {
                write!(f, "byte view is not aligned to {align} bytes")
            }
            Self::invalidlayout => f.write_str("layout is not valid for the shape"),
        }
    }
}

impl std::error::Error for tensorerror {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
