use ::tensor::dtype;
use lib::limits;

/// The capabilities a backend declares to the runtime.
///
/// The fields are public so that a backend owner can assemble its own
/// declaration directly. Capabilities are a statement about what the backend
/// accepts; they do not perform validation or allocation.
#[derive(Clone, Debug)]
pub struct capabilities {
    /// The element dtypes the backend can execute.
    pub dtypes: Vec<dtype>,
    /// The maximum tensor rank the backend accepts.
    pub maxrank: usize,
    /// The resource limits applied to backend work.
    pub limits: limits,
}

impl capabilities {
    /// Reports whether the backend supports `dtype`.
    #[must_use]
    pub fn supports(&self, dtype: dtype) -> bool {
        self.dtypes.contains(&dtype)
    }

    /// Returns the dtypes the backend supports.
    #[must_use]
    pub fn getdtypes(&self) -> &[dtype] {
        &self.dtypes
    }

    /// Returns the maximum tensor rank the backend accepts.
    #[must_use]
    pub fn getmaxrank(&self) -> usize {
        self.maxrank
    }

    /// Returns the resource limits applied to backend work.
    #[must_use]
    pub fn getlimits(&self) -> limits {
        self.limits
    }
}
