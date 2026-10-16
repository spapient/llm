use thiserror::Error;

/// Failures owned by the CUDA backend.
///
/// The executor maps these onto the [`api::backenderror`] at the
/// `api::compute` boundary: size overflow keeps its category; device absence
/// becomes `devicenotavailable`; compilation and module-loading failures become
/// `kernelcompilefailed` carrying the driver diagnostic; transfer, launch,
/// cuBLAS, and cuDNN runtime failures collapse into `executionfailed`. The
/// richer diagnostic stays here so it can be inspected without widening the
/// public `api` contract.
#[derive(Debug, Error)]
pub enum cudaerror {
    /// No usable CUDA device could be opened.
    #[error("no usable CUDA device: {message}")]
    device {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// NVRTC compilation of the kernel source failed.
    #[error("NVRTC compilation failed: {message}")]
    compile {
        /// The NVRTC-provided diagnostic.
        message: String,
    },

    /// Loading the compiled module or resolving a kernel failed.
    #[error("CUDA module load failed: {message}")]
    module {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// Copying data between host and device failed.
    #[error("CUDA host/device transfer failed: {message}")]
    transfer {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// Kernel launch failed.
    #[error("CUDA kernel launch failed: {message}")]
    launch {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// The cuBLAS call failed.
    #[error("cuBLAS call failed: {message}")]
    blas {
        /// The cuBLAS-provided diagnostic.
        message: String,
    },

    /// The cuDNN call failed.
    #[error("cuDNN call failed: {message}")]
    cudnn {
        /// The cuDNN-provided diagnostic.
        ///
        /// `CudnnError` carries only a status code, so the message is the
        /// debug rendering of that code; the code itself is the stable part.
        message: String,
    },

    /// A size computation exceeded the addressable range.
    #[error("size overflow")]
    sizeoverflow,
}
