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

    /// Creates a tensor from a shape, an explicit layout, and element data.
    ///
    /// # Errors
    ///
    /// Returns [`tensorerror::invalidlayout`] when the layout is not valid for
    /// the shape, or [`tensorerror::lengthmismatch`] when `data.len()` differs
    /// from the shape's element count.
    pub fn newwithlayout(s: shape, l: layout, data: Vec<T>) -> Result<Self, tensorerror> {
        let n = s.getelementcount()?;
        if data.len() != n {
            return Err(tensorerror::lengthmismatch {
                expected: n,
                actual: data.len(),
            });
        }
        if !l.isvalidfor(&s) {
            return Err(tensorerror::invalidlayout);
        }
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

impl<T: element + bytemuck::Pod> tensor<T> {
    /// Builds an owned tensor by copying a byte slice into typed elements.
    ///
    /// Unlike [`tensorview::tryfrombytes`], the source bytes may have any
    /// alignment; each element is read with an unaligned, checked read. The
    /// byte length must equal the shape's element count multiplied by the
    /// element width.
    ///
    /// # Errors
    ///
    /// Returns the error from [`shape::getelementcount`],
    /// [`tensorerror::lengthmismatch`] when the byte length is wrong, or the
    /// error from [`tensor::new`] (which cannot occur for a validated length).
    pub fn tryfrombytes(s: shape, bytes: &[u8]) -> Result<Self, tensorerror> {
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
        let data = bytes
            .chunks_exact(core::mem::size_of::<T>())
            .map(|c| bytemuck::pod_read_unaligned(c))
            .collect::<Vec<T>>();
        Self::new(s, data)
    }

    /// Builds an owned tensor from a byte slice and an explicit layout.
    ///
    /// Like [`tensor::tryfrombytes`], each element is read with an unaligned,
    /// checked read; the layout is adopted verbatim after validation, so the
    /// returned tensor can describe a non-contiguous (e.g. transposed) mapping
    /// onto the byte buffer.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`tensor::newwithlayout`] or
    /// [`tensor::tryfrombytes`].
    pub fn tryfrombyteswithlayout(s: shape, l: layout, bytes: &[u8]) -> Result<Self, tensorerror> {
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
        let data = bytes
            .chunks_exact(core::mem::size_of::<T>())
            .map(|c| bytemuck::pod_read_unaligned(c))
            .collect::<Vec<T>>();
        Self::newwithlayout(s, l, data)
    }

    /// Returns the element data as raw bytes in storage order.
    ///
    /// The returned slice's length equals the shape's element count multiplied
    /// by the element width, and its byte representation matches the storage
    /// dtype (IEEE-754 bits for floating-point types).
    #[must_use]
    pub fn getbytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }
}
