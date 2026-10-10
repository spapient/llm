use core::ops::{Deref, DerefMut};

use ::tensor::element;

use crate::arena::graph;

/// An RAII scope that suppresses autograd recording.
///
/// A `nogradscope` borrows the graph mutably and clears its recording flag on
/// creation. While the scope is alive, operators still run and produce values,
/// but every output is detached: it carries no producer and never requires a
/// gradient. The flag is restored when the scope is dropped, including during
/// unwinding, so recording never stays disabled by accident.
///
/// The scope derefs to [`graph`], so operators are called on it directly:
///
/// ```ignore
/// let mut guard = g.nograd();
/// let y = guard.matmul(&engine, a, b)?;
/// drop(guard);
/// ```
pub struct nogradscope<'a, T: element> {
    graph: &'a mut graph<T>,
    previous: bool,
}

impl<'a, T: element> nogradscope<'a, T> {
    /// Opens a suppression scope over `g`, remembering its previous mode.
    pub(crate) fn begin(g: &'a mut graph<T>) -> Self {
        let previous = g.recording;
        g.recording = false;
        Self { graph: g, previous }
    }
}

impl<T: element> Deref for nogradscope<'_, T> {
    type Target = graph<T>;

    fn deref(&self) -> &graph<T> {
        self.graph
    }
}

impl<T: element> DerefMut for nogradscope<'_, T> {
    fn deref_mut(&mut self) -> &mut graph<T> {
        self.graph
    }
}

impl<T: element> Drop for nogradscope<'_, T> {
    fn drop(&mut self) {
        self.graph.recording = self.previous;
    }
}
