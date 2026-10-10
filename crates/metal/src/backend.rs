use ::tensor::dtype;
use api::{backend, capabilities, device};
use lib::limits;

/// The maximum tensor rank the Metal backend accepts.
///
/// The value bounds the number of dimensions the executor processes. It is
/// declared here and enforced by
/// [`metalexecutor`](crate::metalexecutor), not validated by this declaration.
pub const METALMAXRANK: usize = 8;

/// The maximum head dimension the attention kernel accepts.
///
/// The value is a device limit rather than a contract limit: the kernel keeps
/// one accumulator slot per head dimension in threadgroup memory, so its
/// request is `(2 * ATTENTIONLANES + METALMAXHEADDIM) * 4` bytes. Apple GPUs
/// expose 32 KiB of threadgroup memory per threadgroup, so at this bound the
/// request is 18 432 B, leaving about 14 KiB of headroom. Basis (GOV7):
/// 18 432 B < 32 768 B. The bound MUST stay equal to the `accum` array in the
/// attention kernel of `crates/metal/src/source.rs`.
///
/// A call beyond this bound is rejected as resource exhaustion by
/// [`metalexecutor`](crate::metalexecutor).
pub const METALMAXHEADDIM: usize = 4096;

/// The Metal execution backend.
///
/// `metalbackend` is the identity and capability declaration required by the
/// [`backend`] contract. It owns no execution logic and holds no request state;
/// the executor enforces the budget and drives the kernels.
///
/// The constructor takes explicit [`limits`]: this crate invents no default
/// resource bounds.
#[derive(Clone, Debug)]
pub struct metalbackend {
    caps: capabilities,
}

impl metalbackend {
    /// Creates a Metal backend that declares `f32` support and `lim`.
    #[must_use]
    pub fn new(lim: limits) -> Self {
        Self {
            caps: capabilities {
                dtypes: vec![dtype::f32],
                maxrank: METALMAXRANK,
                limits: lim,
            },
        }
    }
}

impl backend for metalbackend {
    fn getname(&self) -> &'static str {
        "metal"
    }

    fn getdevice(&self) -> device {
        device::metal()
    }

    fn getcapabilities(&self) -> &capabilities {
        &self.caps
    }
}
