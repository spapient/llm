use ::tensor::element;

use crate::arena::graph;
use crate::error::autoggraderror;
use crate::node::variableid;

/// Gradient storage helpers on the tape.
///
/// Reading a gradient is served by [`graph::getgrad`]; this module adds the
/// mutation side: clearing one node's gradient and clearing the whole tape
/// before a fresh reverse pass.
impl<T: element> graph<T> {
    /// Clears the accumulated gradient of `id`.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] when `id` is not on the tape.
    pub fn zerograd(&mut self, id: variableid) -> Result<(), autoggraderror> {
        let node = self
            .vars
            .get_mut(id.getindex())
            .ok_or(autoggraderror::unknownvariable { id: id.getindex() })?;
        node.cleargrad();
        Ok(())
    }

    /// Clears every accumulated gradient on the tape.
    ///
    /// The tape itself is unchanged; only the gradients are reset. Call this
    /// before re-running [`graph::backward`] on the same forward graph.
    pub fn cleargradients(&mut self) {
        for node in &mut self.vars {
            node.cleargrad();
        }
    }
}
