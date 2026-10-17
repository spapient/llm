use ::tensor::dtype;
use api::{backend, capabilities, device};
use lib::limits;

/// The maximum tensor rank every GPU backend accepts.
pub const MAXRANK: usize = 8;

/// The device-specific half of a backend declaration.
#[doc(hidden)]
pub trait devicespec {
    /// The backend name used in diagnostics.
    const NAME: &'static str;

    /// The attention head-dimension bound this device admits.
    const MAXHEADDIM: usize;

    /// The device variant this backend reports.
    fn device() -> device;
}

/// The identity and capability declaration required by the [`backend`] contract.
///
/// It owns no execution logic and holds no request state; the executor enforces
/// the budget and drives the kernels.
pub struct gpubackend<D> {
    caps: capabilities,
    // `fn() -> D` keeps the device marker out of the value's auto traits.
    marker: core::marker::PhantomData<fn() -> D>,
}

impl<D> Clone for gpubackend<D> {
    fn clone(&self) -> Self {
        Self {
            caps: self.caps.clone(),
            marker: core::marker::PhantomData,
        }
    }
}

impl<D> core::fmt::Debug for gpubackend<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("gpubackend")
            .field("caps", &self.caps)
            .finish()
    }
}

impl<D: devicespec> gpubackend<D> {
    /// Creates a backend that declares `f32` support and `lim`.
    #[must_use]
    pub fn new(lim: limits) -> Self {
        Self {
            caps: capabilities {
                dtypes: vec![dtype::f32],
                maxrank: MAXRANK,
                limits: lim,
            },
            marker: core::marker::PhantomData,
        }
    }
}

impl<D: devicespec> backend for gpubackend<D> {
    fn getname(&self) -> &'static str {
        D::NAME
    }

    fn getdevice(&self) -> device {
        D::device()
    }

    fn getcapabilities(&self) -> &capabilities {
        &self.caps
    }
}
