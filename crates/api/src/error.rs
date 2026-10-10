use ::tensor::dtype;
use thiserror::Error;

/// Failures produced while dispatching or executing a backend kernel.
///
/// Each variant distinguishes a cause that a caller may act on: dtype support,
/// operation support, a size computation that overflowed, and a backend-owned
/// execution failure.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum backenderror {
    /// The backend does not support the requested element dtype.
    #[error("unsupported dtype: {dtype:?}")]
    unsupporteddtype {
        /// The unsupported dtype.
        dtype: dtype,
    },

    /// The backend does not implement the requested operation.
    #[error("unsupported operation: {op}")]
    unsupportedoperation {
        /// The name of the unsupported operation.
        op: &'static str,
    },

    /// A size computation exceeded the addressable range.
    #[error("size overflow")]
    sizeoverflow,

    /// Execution failed for a reason owned by the backend.
    #[error("execution failed")]
    executionfailed,

    /// Two operands have shapes that are incompatible for the requested
    /// operation.
    #[error("incompatible operand shapes for {op}")]
    shapemismatch {
        /// The stable name of the operation whose operands were rejected.
        op: &'static str,
    },

    /// A row index operand is outside the table's valid range.
    #[error("index {index} out of range for length {len}")]
    indexoutofrange {
        /// The rejected index.
        index: usize,
        /// The number of rows the index was validated against.
        len: usize,
    },

    /// A configured resource budget rejected the operation before execution.
    #[error("resource limit exceeded")]
    resourceexhausted,
}
