use ::tensor::dtype;
use api::{backend, capabilities, device};
use lib::limits;

/// The maximum tensor rank the MUSA backend accepts.
///
/// The value bounds the number of dimensions the executor processes. It is
/// declared here and enforced by the executor (stage P1c), not validated by
/// this declaration.
pub const MUSAMAXRANK: usize = 8;

/// The maximum head dimension the attention kernel accepts.
///
/// The value is a device limit rather than a contract limit: the attention
/// kernel keeps one accumulator slot per head dimension in shared memory, so
/// its request is `(2 * lanes + headdim) * 4` bytes. The kernel is ported from
/// the CUDA backend unchanged (docs/musa-backend-plan.md §5.3), so the shared-
/// memory budget and this bound are identical to the CUDA derivation
/// (docs/design.md §21.1, GOV7).
///
/// A call beyond this bound is rejected as resource exhaustion by the
/// executor.
pub const MUSAMAXHEADDIM: usize = 8192;

/// The Moore Threads MUSA execution backend.
///
/// `musabackend` is the identity and capability declaration required by the
/// [`backend`] contract. It owns no execution logic and holds no request
/// state; the executor enforces the budget and drives the kernels.
///
/// The constructor takes explicit [`limits`]: this crate invents no default
/// resource bounds.
#[derive(Clone, Debug)]
pub struct musabackend {
    caps: capabilities,
}

impl musabackend {
    /// Creates a MUSA backend that declares `f32` support and `lim`.
    #[must_use]
    pub fn new(lim: limits) -> Self {
        Self {
            caps: capabilities {
                dtypes: vec![dtype::f32],
                maxrank: MUSAMAXRANK,
                limits: lim,
            },
        }
    }
}

impl backend for musabackend {
    fn getname(&self) -> &'static str {
        "musa"
    }

    fn getdevice(&self) -> device {
        device::musa()
    }

    fn getcapabilities(&self) -> &capabilities {
        &self.caps
    }
}
