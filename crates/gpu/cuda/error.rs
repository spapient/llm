use thiserror::Error;

/// Failures owned by the CUDA backend.
///
/// The executor maps these onto the [`api::backenderror`] at the
/// `api::compute` boundary: size overflow keeps its category; device absence
/// becomes `devicenotavailable`; compilation and module-loading failures become
/// `kernelcompilefailed` carrying the driver diagnostic; transfer, launch,
/// and cuBLAS runtime failures collapse into `executionfailed`. The
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
    #[error("NVRTC compilation failed ({kernel}): {message}")]
    compile {
        /// The kernel entry point whose source failed to compile.
        kernel: String,
        /// The NVRTC-provided diagnostic.
        message: String,
    },

    /// Loading the compiled module or resolving a kernel failed.
    #[error("CUDA module load failed ({kernel}): {message}")]
    module {
        /// The kernel entry point that could not be loaded or resolved.
        kernel: String,
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

    /// A size computation exceeded the addressable range.
    #[error("size overflow")]
    sizeoverflow,
}
