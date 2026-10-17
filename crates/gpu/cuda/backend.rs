use ::tensor::dtype;
use api::{backend, capabilities, device};
use lib::limits;

/// The maximum tensor rank the CUDA backend accepts.
///
/// The value bounds the number of dimensions the executor processes. It is
/// declared here and enforced by [`cudaexecutor`](crate::cuda::cudaexecutor), not
/// validated by this declaration.
pub const CUDAMAXRANK: usize = 8;

/// The maximum head dimension the attention kernel accepts.
///
/// The value is a device limit rather than a contract limit: the kernel keeps
/// one accumulator slot per head dimension in shared memory, so its request is
/// `(2 * 256 + headdim) * 4` bytes. At this bound that is 34 KiB, which stays
/// under the 48 KiB of dynamic shared memory a threadgroup may use without the
/// opt-in that `cudarc` does not currently expose. Basis (GOV7): 34 KiB <
/// 48 KiB leaves ~14 KiB of headroom for driver accounting.
///
/// A call beyond this bound is rejected as resource exhaustion by
/// [`cudaexecutor`](crate::cuda::cudaexecutor).
pub const CUDAMAXHEADDIM: usize = 8192;

/// The CUDA execution backend.
///
/// `cudabackend` is the identity and capability declaration required by the
/// [`backend`] contract. It owns no execution logic and holds no request state;
/// the executor enforces the budget and drives the kernels.
///
/// The constructor takes explicit [`limits`]: this crate invents no default
/// resource bounds.
#[derive(Clone, Debug)]
pub struct cudabackend {
    caps: capabilities,
}

impl cudabackend {
    /// Creates a CUDA backend that declares `f32` support and `lim`.
    #[must_use]
    pub fn new(lim: limits) -> Self {
        Self {
            caps: capabilities {
                dtypes: vec![dtype::f32],
                maxrank: CUDAMAXRANK,
                limits: lim,
            },
        }
    }
}

impl backend for cudabackend {
    fn getname(&self) -> &'static str {
        "cuda"
    }

    fn getdevice(&self) -> device {
        device::cuda()
    }

    fn getcapabilities(&self) -> &capabilities {
        &self.caps
    }
}
