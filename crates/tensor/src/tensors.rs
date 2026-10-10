use crate::{dtype, element, layout, shape, tensorerror, tensorview};

/// An owned, dense tensor with an explicit shape and contiguous layout.
///
/// The element buffer length always equals the shape's element count, and the
/// layout is the canonical contiguous layout for that shape. The tensor owns
/// only its data; it holds no request-scoped mutable state.
#[derive(Clone, Debug)]
pub struct tensor<T: element> {
    shape: shape,
    layout: layout,
    data: Vec<T>,
}

impl<T: element> tensor<T> {
    /// Creates a tensor from a shape and its element data.
    ///
    /// # Errors
    ///
    /// Returns the error from [`shape::getelementcount`], or
    /// [`tensorerror::lengthmismatch`] when `data.len()` differs from the
    /// shape's element count.
    pub fn new(s: shape, data: Vec<T>) -> Result<Self, tensorerror> {
        let n = s.getelementcount()?;
        if data.len() != n {
            return Err(tensorerror::lengthmismatch {
                expected: n,
                actual: data.len(),
            });
        }
        let l = layout::contiguous(&s);
        Ok(Self {
            shape: s,
            layout: l,
            data,
        })
    }

    /// Returns the shape.
    #[must_use]
    pub fn getshape(&self) -> &shape {
        &self.shape
    }

    /// Returns the layout.
    #[must_use]
    pub fn getlayout(&self) -> &layout {
        &self.layout
    }

    /// Returns the storage data type of the elements.
    #[must_use]
    pub fn getdtype(&self) -> dtype {
        T::DTYPE
    }

    /// Returns the number of elements.
    #[must_use]
    pub fn getlen(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when the tensor holds no elements.
    #[must_use]
    pub fn isempty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the element data.
    #[must_use]
    pub fn getdata(&self) -> &[T] {
        &self.data
    }

    /// Returns the element data as a mutable slice.
    #[must_use]
    pub fn getdatamut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns a read-only view borrowing this tensor's shape, layout, and data.
    #[must_use]
    pub fn getview(&self) -> tensorview<'_, T> {
        tensorview::new(&self.shape, &self.layout, &self.data)
    }
}
