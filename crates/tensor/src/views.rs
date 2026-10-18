use crate::{element, layout, shape, tensor, tensorerror};

/// A borrowed, read-only view over a tensor's shape, layout, and data.
///
/// The view holds only references to the owning tensor's parts, so it never
/// allocates and never shares ownership through `Arc` or dynamic dispatch. A
/// view can be copied freely; its lifetime cannot outlive the borrowed data.
#[derive(Clone, Copy, Debug)]
pub struct tensorview<'a, T: element> {
    shape: &'a shape,
    layout: &'a layout,
    data: &'a [T],
}

impl<'a, T: element> tensorview<'a, T> {
    /// Borrows the components of a tensor into a view.
    pub(crate) fn new(s: &'a shape, l: &'a layout, data: &'a [T]) -> Self {
        Self {
            shape: s,
            layout: l,
            data,
        }
    }

    /// Returns the viewed shape.
    #[must_use]
    pub fn getshape(&self) -> &'a shape {
        self.shape
    }

    /// Returns the viewed layout.
    #[must_use]
    pub fn getlayout(&self) -> &'a layout {
        self.layout
    }

    /// Returns the number of elements visible through the view.
    #[must_use]
    pub fn getlen(&self) -> usize {
        self.data.len()
    }

    /// Returns the visible element data.
    #[must_use]
    pub fn getdata(&self) -> &'a [T] {
        self.data
    }

    /// Copies the viewed data into a new owned tensor.
    ///
    /// # Errors
    ///
    /// Returns the error from [`tensor::new`] if the copied element count is
    /// inconsistent with the shape.
    pub fn toowned(&self) -> Result<tensor<T>, tensorerror> {
        tensor::new(self.shape.clone(), self.data.to_vec())
    }
}

// NOTE: a zero-copy `tryfrombytes` (borrowing a byte slice as a typed view)
// previously existed here via `bytemuck::try_cast_slice`. The workspace now
// forbids `unsafe` outside the registered islands (AGENTS.md §6), and a
// byte-slice reinterpretation cannot be expressed in safe Rust, so the method
// was removed (it had no callers). Byte construction paths use the copying
// [`tensor::tryfrombytes`] constructor instead.
