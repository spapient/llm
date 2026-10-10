/// A handle to a [`variable`](crate::variable) stored in an autograd
/// [`graph`](crate::graph).
///
/// The handle is a graph-local index; it is copyable and carries no borrow, so
/// operators can read and write the graph through shared ownership instead of
/// holding a reference to a node. A handle only ever comes from the graph that
/// created it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct variableid(usize);

impl variableid {
    /// Creates a handle for the variable at `index`.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the graph-local index.
    #[must_use]
    pub const fn getindex(self) -> usize {
        self.0
    }
}

/// How a [`variable`](crate::variable) was produced.
///
/// The operator is a static enum rather than a boxed closure, so the tape holds
/// no dynamic dispatch and no captured heap state. Every variant names the input
/// handles it consumes, which is exactly the information the reverse pass
/// needs to route gradients.
#[derive(Clone, Copy, Debug)]
pub enum op {
    /// A leaf input with no producer.
    leaf,
    /// Matrix multiplication of `lhs` and `rhs`.
    matmul {
        /// The left operand `[m, k]`.
        lhs: variableid,
        /// The right operand `[k, n]`.
        rhs: variableid,
    },
    /// Elementwise addition of `lhs` and `rhs`.
    add {
        /// The left operand.
        lhs: variableid,
        /// The right operand.
        rhs: variableid,
    },
    /// Elementwise multiplication of `lhs` and `rhs`.
    mul {
        /// The left operand.
        lhs: variableid,
        /// The right operand.
        rhs: variableid,
    },
    /// RMS normalization of `input` by `weight` with `eps`.
    rmsnorm {
        /// The normalized input.
        input: variableid,
        /// The per-column scale weight.
        weight: variableid,
        /// The numerical-stability epsilon.
        eps: f32,
    },
    /// Softmax over the last axis of `input`.
    softmax {
        /// The input.
        input: variableid,
    },
    /// Rectified linear unit of `input`.
    relu {
        /// The input.
        input: variableid,
    },
    /// Sigmoid linear unit of `input`.
    silu {
        /// The input.
        input: variableid,
    },
    /// Tanh-approximation GELU of `input`.
    gelutanh {
        /// The input.
        input: variableid,
    },
}

impl op {
    /// Returns the stable operation name.
    #[must_use]
    pub const fn getname(self) -> &'static str {
        match self {
            Self::leaf => "leaf",
            Self::matmul { .. } => "matmul",
            Self::add { .. } => "add",
            Self::mul { .. } => "mul",
            Self::rmsnorm { .. } => "rmsnorm",
            Self::softmax { .. } => "softmax",
            Self::relu { .. } => "relu",
            Self::silu { .. } => "silu",
            Self::gelutanh { .. } => "gelutanh",
        }
    }

    /// Returns the input handles this operator consumes, in argument order.
    #[must_use]
    pub fn getinputs(self) -> Vec<variableid> {
        match self {
            Self::leaf => Vec::new(),
            Self::matmul { lhs, rhs } | Self::add { lhs, rhs } | Self::mul { lhs, rhs } => {
                vec![lhs, rhs]
            }
            Self::rmsnorm { input, weight, .. } => vec![input, weight],
            Self::softmax { input }
            | Self::relu { input }
            | Self::silu { input }
            | Self::gelutanh { input } => vec![input],
        }
    }

    /// Returns `true` when the operator has no producer.
    #[must_use]
    pub const fn isleaf(self) -> bool {
        matches!(self, Self::leaf)
    }
}
