use ::tensor::tensorerror;
use api::backenderror;

/// Failures produced while building or differentiating an autograd graph.
///
/// The enum separates the distinctions a caller of `autograd` must act on: a
/// backend operator failure, a handle that does not belong to the graph, an
/// operand shape the operator does not support, and a size computation that
/// overflowed. Backend failures keep the [`backenderror`] carried by the
/// `api` contracts.
#[derive(Debug, PartialEq, Eq)]
pub enum autoggraderror {
    /// A backend operator failed during the forward or reverse pass.
    backend(backenderror),

    /// Tensor metadata construction failed while building a gradient.
    tensor(tensorerror),

    /// A variable handle does not refer to a node in this graph.
    unknownvariable {
        /// The rejected handle index.
        id: usize,
    },

    /// The requested operation is not defined for the operand shape.
    unsupportedshape {
        /// The stable operation name.
        op: &'static str,
        /// The rejected operand shape.
        dims: Vec<usize>,
    },

    /// A size computation overflowed while planning a gradient.
    sizeoverflow {
        /// The stable operation name.
        op: &'static str,
    },
}

impl std::fmt::Display for autoggraderror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::backend(f0) => std::fmt::Display::fmt(f0, f),
            Self::tensor(f0) => std::fmt::Display::fmt(f0, f),
            Self::unknownvariable { id } => {
                write!(f, "variable id {id} is not present in the graph")
            }
            Self::unsupportedshape { op, dims } => {
                write!(f, "operation {op} does not support shape {dims:?}")
            }
            Self::sizeoverflow { op } => {
                write!(f, "size overflow while planning autograd operation {op}")
            }
        }
    }
}

impl std::error::Error for autoggraderror {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::backend(f0) => Some(f0),
            Self::tensor(f0) => Some(f0),
            _ => None,
        }
    }
}

impl From<backenderror> for autoggraderror {
    fn from(v: backenderror) -> Self {
        Self::backend(v)
    }
}

impl From<tensorerror> for autoggraderror {
    fn from(v: tensorerror) -> Self {
        Self::tensor(v)
    }
}
