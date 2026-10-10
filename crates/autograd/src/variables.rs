use ::tensor::{element, tensor};

use crate::node::{op, variableid};

/// One node of the autograd tape: a forward value, its accumulated gradient,
/// its producer, and whether it requires a gradient.
///
/// The [`graph`](crate::graph) owns every variable and is the only place they
/// are stored; a caller holds a [`variableid`] handle instead of a `variable`.
/// The gradient is absent until a reverse pass writes it. `requiresgrad` is
/// false for constants and for any value produced while recording is
/// suppressed, so such a value never participates in a reverse pass.
#[derive(Clone, Debug)]
pub struct variable<T: element> {
    id: variableid,
    value: tensor<T>,
    grad: Option<tensor<T>>,
    requiresgrad: bool,
    producer: op,
}

impl<T: element> variable<T> {
    pub(crate) const fn new(
        id: variableid,
        value: tensor<T>,
        requiresgrad: bool,
        producer: op,
    ) -> Self {
        Self {
            id,
            value,
            grad: None,
            requiresgrad,
            producer,
        }
    }

    /// Returns this node's handle.
    #[must_use]
    pub const fn getid(&self) -> variableid {
        self.id
    }

    /// Returns the forward value.
    #[must_use]
    pub const fn getvalue(&self) -> &tensor<T> {
        &self.value
    }

    /// Returns the accumulated gradient, if a reverse pass has written one.
    #[must_use]
    pub const fn getgrad(&self) -> Option<&tensor<T>> {
        self.grad.as_ref()
    }

    /// Returns whether a reverse pass should accumulate into this node.
    #[must_use]
    pub const fn requiresgrad(&self) -> bool {
        self.requiresgrad
    }

    /// Returns how this node was produced.
    #[must_use]
    pub const fn getproducer(&self) -> op {
        self.producer
    }

    /// Replaces the accumulated gradient.
    pub(crate) fn setgrad(&mut self, grad: tensor<T>) {
        self.grad = Some(grad);
    }

    /// Clears the accumulated gradient.
    pub(crate) fn cleargrad(&mut self) {
        self.grad = None;
    }
}
