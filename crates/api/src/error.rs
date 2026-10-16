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

    /// No device is available for this backend.
    ///
    /// Raised when the backend cannot open a usable device (for example, no
    /// Metal device on the host, or no CUDA-capable GPU), so the caller can
    /// distinguish "the environment has no GPU" from "the kernel failed".
    #[error("no device is available for this backend")]
    devicenotavailable,

    /// Kernel compilation or module loading failed for this backend.
    ///
    /// Carries the backend name, the kernel entry point, and the driver-provided
    /// diagnostic so a compilation failure is reachable and actionable, instead
    /// of collapsing into the generic [`backenderror::executionfailed`].
    #[error("kernel compilation failed on {backend} ({kernel}): {message}")]
    kernelcompilefailed {
        /// The backend whose kernel failed to compile.
        backend: &'static str,
        /// The kernel entry point (the `(source, entry)` pair's entry half)
        /// that failed to compile or load.
        kernel: String,
        /// The driver-provided diagnostic.
        message: String,
    },

    /// Execution failed for a reason owned by the backend.
    ///
    /// Carries the backend name, the failing stage or kernel entry point, and
    /// the backend-provided diagnostic so a runtime failure is reachable and
    /// actionable — a caller can tell *where* it failed (`launch` vs
    /// `transfer` vs `construct`) and *why*, instead of receiving an opaque
    /// marker.
    #[error("execution failed on {backend} ({kernel}): {message}")]
    executionfailed {
        /// The backend whose execution failed.
        backend: &'static str,
        /// The failing stage or kernel entry point.
        kernel: String,
        /// The backend-provided diagnostic.
        message: String,
    },

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

    /// An attribute value lies outside the domain the operation accepts.
    #[error("invalid parameter for {op}")]
    invalidparameter {
        /// The stable name of the operation whose parameter was rejected.
        op: &'static str,
    },

    /// A configured resource budget rejected the operation before execution.
    #[error("resource limit exceeded")]
    resourceexhausted,
}
