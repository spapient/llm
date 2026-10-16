use thiserror::Error;

/// Failures owned by the Metal backend.
///
/// The executor maps these onto the [`api::backenderror`] at the
/// `api::compute` boundary: shape and size failures keep their category; device
/// absence becomes `devicenotavailable`; compilation and pipeline failures
/// become `kernelcompilefailed` carrying the driver diagnostic; allocation and
/// runtime execution failures collapse into `executionfailed`. The richer
/// diagnostic stays here so it can be inspected without widening the public
/// `api` contract.
#[derive(Debug, Error)]
pub enum metalerror {
    /// No Metal device is available on this system.
    #[error("no Metal device is available")]
    nodevice,

    /// A Metal buffer, or the host buffer that mirrors it, could not be
    /// allocated.
    #[error("failed to allocate memory for a Metal buffer")]
    allocation,

    /// Metal Shading Language compilation failed.
    #[error("Metal Shading Language compilation failed: {message}")]
    compile {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// Compute pipeline creation failed.
    #[error("Metal compute pipeline creation failed: {message}")]
    pipeline {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// Kernel execution failed.
    #[error("Metal kernel execution failed: {message}")]
    execution {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// A size computation exceeded the addressable range.
    #[error("size overflow")]
    sizeoverflow,
}
