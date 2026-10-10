use thiserror::Error;

/// Failures owned by the Metal backend.
///
/// The executor maps these onto the frozen [`api::backenderror`] at the
/// `api::compute` boundary: shape and size failures keep their category, while
/// device, compilation, pipeline, and execution failures collapse into
/// `backenderror::executionfailed`. The richer diagnostic stays here so it can
/// be inspected without widening the public `api` contract.
#[derive(Debug, Error)]
pub enum metalerror {
    /// No Metal device is available on this system.
    #[error("no Metal device is available")]
    nodevice,

    /// A Metal buffer could not be allocated.
    #[error("failed to allocate a Metal buffer")]
    bufferallocation,

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
