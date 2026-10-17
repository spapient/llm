use ::tensor::{shape, tensor};
use api::backenderror;

use std::fmt;

/// Errors from a GPU backend, each carrying the backend name in diagnostics.
///
/// Each variant is constructed only by the backend that can produce it, so a
/// given target sees the other variants as unused.
#[allow(dead_code)]
#[derive(Debug)]
pub enum gpuerror {
    /// No usable GPU device could be opened.
    device {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// No device is available on this system.
    nodevice,

    /// A device allocation failed.
    allocation,

    /// Kernel compilation failed.
    compile {
        /// The kernel entry point whose source failed to compile.
        kernel: String,
        /// The compiler-provided diagnostic.
        message: String,
    },

    /// Loading the compiled module or resolving a kernel failed.
    module {
        /// The kernel entry point that could not be loaded or resolved.
        kernel: String,
        /// The driver-provided diagnostic.
        message: String,
    },

    /// Compute pipeline creation failed.
    pipeline {
        /// The kernel entry point whose pipeline could not be created.
        kernel: String,
        /// The driver-provided diagnostic.
        message: String,
    },

    /// Copying data between host and device failed.
    transfer {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// Kernel launch failed.
    launch {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// A vendor BLAS call failed.
    blas {
        /// The library-provided diagnostic.
        message: String,
    },

    /// Kernel execution failed.
    execution {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// A size computation exceeded the addressable range.
    sizeoverflow,
}

impl gpuerror {
    /// Maps this failure onto the frozen `api::backenderror`.
    ///
    /// `backend` names the backend in the diagnostic, so the shared mapping
    /// still identifies the device that failed.
    pub(crate) fn maptobackend(self, backend: &'static str) -> api::backenderror {
        use api::backenderror;
        match self {
            gpuerror::sizeoverflow => backenderror::sizeoverflow,
            gpuerror::device { .. } | gpuerror::nodevice => backenderror::devicenotavailable,
            gpuerror::compile { kernel, message }
            | gpuerror::module { kernel, message }
            | gpuerror::pipeline { kernel, message } => backenderror::kernelcompilefailed {
                backend,
                kernel,
                message,
            },
            gpuerror::transfer { message } => backenderror::executionfailed {
                backend,
                kernel: "transfer".into(),
                message,
            },
            gpuerror::launch { message } => backenderror::executionfailed {
                backend,
                kernel: "launch".into(),
                message,
            },
            gpuerror::blas { message } => backenderror::executionfailed {
                backend,
                kernel: "blas".into(),
                message,
            },
            gpuerror::allocation => backenderror::executionfailed {
                backend,
                kernel: "alloc".into(),
                message: "device allocation failed".into(),
            },
            gpuerror::execution { message } => backenderror::executionfailed {
                backend,
                kernel: "run".into(),
                message,
            },
        }
    }
}

impl fmt::Display for gpuerror {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::device { message } => write!(f, "no usable GPU device: {message}"),
            Self::nodevice => f.write_str("no GPU device is available"),
            Self::allocation => f.write_str("failed to allocate device memory"),
            Self::compile { kernel, message } => {
                write!(f, "kernel compilation failed ({kernel}): {message}")
            }
            Self::module { kernel, message } => {
                write!(f, "module load failed ({kernel}): {message}")
            }
            Self::pipeline { kernel, message } => {
                write!(f, "compute pipeline creation failed ({kernel}): {message}")
            }
            Self::transfer { message } => write!(f, "host/device transfer failed: {message}"),
            Self::launch { message } => write!(f, "kernel launch failed: {message}"),
            Self::blas { message } => write!(f, "BLAS call failed: {message}"),
            Self::execution { message } => write!(f, "kernel execution failed: {message}"),
            Self::sizeoverflow => f.write_str("size overflow"),
        }
    }
}

impl std::error::Error for gpuerror {}

/// Rebuilds a kernel result into a tensor of `outshape`.
///
/// `backend` names the backend in the failure diagnostic.
pub(crate) fn wrap(
    backend: &'static str,
    outshape: shape,
    data: Vec<f32>,
) -> Result<tensor<f32>, backenderror> {
    tensor::new(outshape, data).map_err(|e| backenderror::executionfailed {
        backend,
        kernel: "construct".into(),
        message: e.to_string(),
    })
}
