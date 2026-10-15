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

impl<'a, T: element + bytemuck::Pod> tensorview<'a, T> {
    /// Borrows a byte slice as a typed read-only view without copying.
    ///
    /// The byte length must equal the shape's element count multiplied by the
    /// element width, `l` must be a layout valid for `s` (the canonical choice
    /// is [`layout::contiguous`]), and `bytes` must be aligned to the element
    /// type's alignment. When the source is not aligned the method returns
    /// [`tensorerror::viewmisaligned`] and the caller may fall back to the
    /// copying constructor [`tensor::tryfrombytes`].
    ///
    /// # Errors
    ///
    /// Returns the error from [`shape::getelementcount`],
    /// [`tensorerror::lengthmismatch`] when the byte length is wrong,
    /// [`tensorerror::invalidlayout`] when `l` does not describe `s`, or
    /// [`tensorerror::viewmisaligned`] when the source is not aligned.
    pub fn tryfrombytes(s: &'a shape, l: &'a layout, bytes: &'a [u8]) -> Result<Self, tensorerror> {
        let count = s.getelementcount()?;
        let expected = count
            .checked_mul(core::mem::size_of::<T>())
            .ok_or_else(|| tensorerror::elementcountoverflow {
                dims: s.getdims().to_vec(),
            })?;
        if bytes.len() != expected {
            return Err(tensorerror::lengthmismatch {
                expected,
                actual: bytes.len(),
            });
        }
        if !l.isvalidfor(s) {
            return Err(tensorerror::invalidlayout);
        }
        let data = if bytes.is_empty() {
            &[]
        } else {
            bytemuck::try_cast_slice(bytes).map_err(|_| tensorerror::viewmisaligned {
                align: core::mem::align_of::<T>(),
            })?
        };
        Ok(Self::new(s, l, data))
    }
}
