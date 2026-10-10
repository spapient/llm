use ::tensor::dtype;
use api::{backend, capabilities, device};
use lib::limits;

/// The maximum tensor rank the CPU backend accepts.
///
/// The value bounds the number of dimensions the elementwise and last-axis
/// kernels process. It is declared here and enforced by
/// [`cpuexecutor`](crate::cpuexecutor), not validated by this declaration.
pub const CPUMAXRANK: usize = 8;

/// The CPU execution backend.
///
/// `cpubackend` is the identity and capability declaration required by the
/// [`backend`] contract. It owns no execution logic and holds no request state;
/// kernels are implemented separately and composed by
/// [`cpuexecutor`](crate::cpuexecutor).
///
/// The constructor takes explicit [`limits`]: this crate invents no default
/// resource bounds.
#[derive(Clone, Debug)]
pub struct cpubackend {
    caps: capabilities,
}

impl cpubackend {
    /// Creates a CPU backend that declares `f32` support and the given resource
    /// limits.
    #[must_use]
    pub fn new(lim: limits) -> Self {
        Self {
            caps: capabilities {
                dtypes: vec![dtype::f32],
                maxrank: CPUMAXRANK,
                limits: lim,
            },
        }
    }
}

impl backend for cpubackend {
    fn getname(&self) -> &'static str {
        "cpu"
    }

    fn getdevice(&self) -> device {
        device::cpu()
    }

    fn getcapabilities(&self) -> &capabilities {
        &self.caps
    }
}
