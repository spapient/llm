use ::tensor::element;
use ::tensor::tensor;
use ::tensor::tensorview;

use crate::backenderror;
use crate::capabilities;
use crate::device;

/// A narrow execution-backend identity contract.
///
/// A `backend` exposes only its name, its device, and its declared
/// capabilities. Execution, memory, and synchronization responsibilities are
/// intentionally not merged into this trait; they belong to
/// [`kernel`](crate::kernel) implementations.
pub trait backend {
    /// Returns the stable backend name.
    fn getname(&self) -> &'static str;

    /// Returns the device this backend runs on.
    fn getdevice(&self) -> device;

    /// Returns the capabilities this backend declares.
    fn getcapabilities(&self) -> &capabilities;
}

/// A generic, statically dispatched kernel call contract.
///
/// `kernel` is parameterized by the element type `T` it operates on and is
/// implemented per concrete kernel in a later layer. It is deliberately narrow:
/// it reports a name and applies itself to a borrowed input view, returning a
/// newly owned tensor or a [`backenderror`](crate::backenderror).
pub trait kernel<T: element> {
    /// Returns the stable kernel name.
    fn getname(&self) -> &'static str;

    /// Applies the kernel to `input`.
    ///
    /// # Errors
    ///
    /// Returns a [`backenderror`](crate::backenderror) when the element dtype is
    /// unsupported, the operation is unsupported, an element count overflows, or
    /// execution fails.
    fn apply(&self, input: &tensorview<'_, T>) -> Result<tensor<T>, backenderror>;
}
