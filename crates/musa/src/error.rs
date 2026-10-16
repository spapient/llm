use thiserror::Error;

/// Failures owned by the MUSA backend.
///
/// The executor maps these onto the [`api::backenderror`] at the
/// `api::compute` boundary: size overflow keeps its category; device absence
/// becomes `devicenotavailable`; module-loading failures become
/// `kernelcompilefailed` carrying the driver diagnostic; transfer and launch
/// runtime failures collapse into `executionfailed`. The richer diagnostic
/// stays here so it can be inspected without widening the public `api`
/// contract.
///
/// Compared with `cudaerror` (docs/design.md §21.2 decision 11), the
/// compilation, cuBLAS, and cuDNN classes are absent by design
/// (docs/musa-backend-plan.md §5): kernels are compiled at build time by `mcc`
/// (a build failure is a build failure, not a run-time error), and the backend
/// deliberately uses no vendor acceleration libraries — matmul runs a WMMA
/// kernel on the Tensor Core and softmax the max-subtracting kernel, so the
/// only run-time interaction classes left are device, module, transfer,
/// launch, and size overflow.
#[derive(Debug, Error)]
pub enum musaerror {
    /// No usable MUSA device could be opened.
    #[error("no usable MUSA device: {message}")]
    device {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// Loading the embedded fatbin module or resolving a kernel failed.
    #[error("MUSA module load failed ({kernel}): {message}")]
    module {
        /// The kernel entry point that could not be loaded or resolved.
        kernel: String,
        /// The driver-provided diagnostic.
        message: String,
    },

    /// Copying data between host and device failed.
    #[error("MUSA host/device transfer failed: {message}")]
    transfer {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// Kernel launch failed.
    #[error("MUSA kernel launch failed: {message}")]
    launch {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// A size computation exceeded the addressable range.
    #[error("size overflow")]
    sizeoverflow,
}
