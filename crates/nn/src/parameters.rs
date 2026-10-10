use ::tensor::{shape, tensor};

/// A single named, owned tensor that a layer learns.
///
/// A `parameter` binds a stable name to the tensor it holds. The tensor is
/// owned and immutable through the public interface; layers read it for forward
/// passes and collect it into a `statedict` for persistence. The name is the
/// leaf name only; a layer composes a hierarchical name by prefixing it during
/// collection, so the same parameter type works at any nesting depth.
#[derive(Clone, Debug)]
pub struct parameter {
    name: String,
    value: tensor<f32>,
}

/// Compares two tensors by shape and element data.
///
/// `tensor` deliberately does not implement `PartialEq` (element equality is
/// not a tensor-metadata concern). Layer containers define equality in terms of
/// shape and data so tests can assert round-trips without reaching into
/// private fields.
pub(crate) fn tensoreq(a: &tensor<f32>, b: &tensor<f32>) -> bool {
    a.getshape().getdims() == b.getshape().getdims() && a.getdata() == b.getdata()
}

impl PartialEq for parameter {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && tensoreq(&self.value, &other.value)
    }
}

impl parameter {
    /// Creates a parameter from its leaf name and value.
    #[must_use]
    pub fn new(name: impl Into<String>, value: tensor<f32>) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }

    /// Returns the leaf name.
    #[must_use]
    pub fn getname(&self) -> &str {
        &self.name
    }

    /// Returns the value tensor.
    #[must_use]
    pub fn getvalue(&self) -> &tensor<f32> {
        &self.value
    }

    /// Returns the value shape.
    #[must_use]
    pub fn getshape(&self) -> &shape {
        self.value.getshape()
    }

    /// Returns the number of elements.
    #[must_use]
    pub fn getlen(&self) -> usize {
        self.value.getlen()
    }
}
