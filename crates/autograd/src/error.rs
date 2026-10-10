use ::tensor::tensorerror;
use api::backenderror;
use thiserror::Error;

/// Failures produced while building or differentiating an autograd graph.
///
/// The enum separates the distinctions a caller of `autograd` must act on: a
/// backend operator failure, a handle that does not belong to the graph, an
/// operand shape the operator does not support, and a size computation that
/// overflowed. Backend failures keep the [`backenderror`] carried by the
/// `api` contracts.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum autoggraderror {
    /// A backend operator failed during the forward or reverse pass.
    #[error(transparent)]
    backend(#[from] backenderror),

    /// Tensor metadata construction failed while building a gradient.
    #[error(transparent)]
    tensor(#[from] tensorerror),

    /// A variable handle does not refer to a node in this graph.
    #[error("variable id {id} is not present in the graph")]
    unknownvariable {
        /// The rejected handle index.
        id: usize,
    },

    /// The requested operation is not defined for the operand shape.
    #[error("operation {op} does not support shape {dims:?}")]
    unsupportedshape {
        /// The stable operation name.
        op: &'static str,
        /// The rejected operand shape.
        dims: Vec<usize>,
    },

    /// A size computation overflowed while planning a gradient.
    #[error("size overflow while planning autograd operation {op}")]
    sizeoverflow {
        /// The stable operation name.
        op: &'static str,
    },
}
