use ::tensor::{element, tensor};

use crate::error::autoggraderror;
use crate::node::{op, variableid};
use crate::nograd::nogradscope;
use crate::variables::variable;

/// The define-by-run tape and the sole owner of every
/// [`variable`] it records.
///
/// A `graph` owns a `Vec` of variables in creation order. Leaving the tape in
/// the vector keeps its ownership explicit and safe: no pointer aliasing, no
/// interior mutability, and no reference cycle. Operators append one node per
/// call; the reverse pass walks the vector once in reverse order.
#[derive(Clone, Debug)]
pub struct graph<T: element> {
    pub(crate) vars: Vec<variable<T>>,
    pub(crate) recording: bool,
}

impl<T: element> graph<T> {
    /// Creates an empty graph with recording enabled.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vars: Vec::new(),
            recording: true,
        }
    }

    /// Returns the number of variables on the tape.
    #[must_use]
    pub fn getlen(&self) -> usize {
        self.vars.len()
    }

    /// Returns `true` when the tape holds no variables.
    #[must_use]
    pub fn isempty(&self) -> bool {
        self.vars.is_empty()
    }

    /// Returns the variable named by `id`, or `None` when it is not on the tape.
    #[must_use]
    pub fn get(&self, id: variableid) -> Option<&variable<T>> {
        self.vars.get(id.getindex())
    }

    /// Returns the forward value named by `id`, or `None`.
    #[must_use]
    pub fn getvalue(&self, id: variableid) -> Option<&tensor<T>> {
        self.get(id).map(variable::getvalue)
    }

    /// Returns the accumulated gradient named by `id`, or `None`.
    #[must_use]
    pub fn getgrad(&self, id: variableid) -> Option<&tensor<T>> {
        self.get(id).and_then(variable::getgrad)
    }

    /// Returns whether the variable named by `id` requires a gradient.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] when `id` is not on the tape.
    pub fn requiresgrad(&self, id: variableid) -> Result<bool, autoggraderror> {
        self.check(id).map(variable::requiresgrad)
    }

    /// Returns whether recording is currently enabled.
    #[must_use]
    pub const fn recording(&self) -> bool {
        self.recording
    }

    /// Adds a leaf value that optionally requires a gradient.
    ///
    /// The returned handle names a node with no producer, so the reverse pass
    /// stops there and leaves the accumulated gradient for the caller to read.
    pub fn leaf(&mut self, value: tensor<T>, requiresgrad: bool) -> variableid {
        self.push(value, requiresgrad, op::leaf)
    }

    /// Adds a constant leaf that never requires a gradient.
    pub fn constant(&mut self, value: tensor<T>) -> variableid {
        self.push(value, false, op::leaf)
    }

    /// Adds a detached copy of `id`: same value, no producer, no gradient.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] when `id` is not on the tape.
    pub fn detach(&mut self, id: variableid) -> Result<variableid, autoggraderror> {
        let value = self.valueof(id)?.clone();
        Ok(self.constant(value))
    }

    /// Opens an RAII scope that suppresses recording until it is dropped.
    ///
    /// Values produced inside the scope keep their value but carry no producer
    /// and never require a gradient. The previous mode is restored on drop.
    #[must_use]
    pub fn nograd(&mut self) -> nogradscope<'_, T> {
        nogradscope::begin(self)
    }

    /// Returns the variable named by `id`.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] when `id` is not on the tape.
    pub(crate) fn check(&self, id: variableid) -> Result<&variable<T>, autoggraderror> {
        self.get(id)
            .ok_or(autoggraderror::unknownvariable { id: id.getindex() })
    }

    /// Returns the forward value named by `id`.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] when `id` is not on the tape.
    pub(crate) fn valueof(&self, id: variableid) -> Result<&tensor<T>, autoggraderror> {
        self.check(id).map(variable::getvalue)
    }

    /// Appends one variable and returns its handle.
    pub(crate) fn push(
        &mut self,
        value: tensor<T>,
        requiresgrad: bool,
        producer: op,
    ) -> variableid {
        let id = variableid::new(self.vars.len());
        self.vars
            .push(variable::new(id, value, requiresgrad, producer));
        id
    }

    /// Returns whether an operation over `inputs` should record a gradient.
    ///
    /// Recording must be enabled and at least one input must require a gradient.
    /// A missing input is treated as not requiring a gradient; operators read
    /// their operands with [`Self::valueof`] first, so a missing input is
    /// reported there.
    pub(crate) fn record(&self, inputs: &[variableid]) -> bool {
        self.recording
            && inputs
                .iter()
                .any(|&id| self.get(id).is_some_and(variable::requiresgrad))
    }

    /// Replaces the accumulated gradient of `id`.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] when `id` is not on the tape.
    pub(crate) fn setgrad(
        &mut self,
        id: variableid,
        grad: tensor<T>,
    ) -> Result<(), autoggraderror> {
        let node = self
            .vars
            .get_mut(id.getindex())
            .ok_or(autoggraderror::unknownvariable { id: id.getindex() })?;
        node.setgrad(grad);
        Ok(())
    }
}

impl<T: element> Default for graph<T> {
    fn default() -> Self {
        Self::new()
    }
}
