use crate::shape;

/// Describes how a logical tensor maps onto a linear element buffer.
///
/// A layout is one stride per dimension plus a base offset, measured in
/// elements. [`layout::contiguous`] builds the canonical row-major (C-order)
/// layout for a shape: the last stride is one and each earlier stride is the
/// product of the dimensions that follow it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct layout {
    strides: Vec<usize>,
    offset: usize,
}

impl layout {
    /// Builds the canonical contiguous (row-major) layout for `s`.
    ///
    /// Stride computation saturates to `usize::MAX` on overflow. Callers that
    /// require exact strides MUST validate the element count first, for example
    /// with [`shape::getelementcount`] or [`shape::checklimits`].
    #[must_use]
    pub fn contiguous(s: &shape) -> Self {
        let dims = s.getdims();
        let mut strides = vec![0usize; dims.len()];
        let mut acc: usize = 1;
        for (i, &d) in dims.iter().enumerate().rev() {
            strides[i] = acc;
            acc = acc.saturating_mul(d);
        }
        Self { strides, offset: 0 }
    }

    /// Returns the strides, one per dimension.
    #[must_use]
    pub fn getstrides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the base offset, in elements.
    #[must_use]
    pub fn getoffset(&self) -> usize {
        self.offset
    }

    /// Returns `true` when the strides describe a row-major layout.
    ///
    /// Because a layout does not carry its shape, contiguity is judged from the
    /// strides alone: the offset must be zero, the last stride must be one, and
    /// every earlier stride must be an exact multiple of the next one, with a
    /// zero inner stride requiring all outer strides to be zero as well.
    #[must_use]
    pub fn iscontiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        let Some(&last) = self.strides.last() else {
            return true;
        };
        if last != 1 {
            return false;
        }
        for w in self.strides.windows(2) {
            let outer = w[0];
            let inner = w[1];
            if inner == 0 {
                if outer != 0 {
                    return false;
                }
            } else if outer % inner != 0 {
                return false;
            }
        }
        true
    }

    /// Returns `true` when this layout can describe `s`.
    ///
    /// A layout is valid for a shape when the rank matches and every element of
    /// the shape maps to a linear index inside the shape's element count,
    /// computed with checked arithmetic. A shape with zero elements accepts any
    /// layout of matching rank.
    #[must_use]
    pub fn isvalidfor(&self, s: &shape) -> bool {
        if self.strides.len() != s.getrank() {
            return false;
        }
        let n = match s.getelementcount() {
            Ok(n) => n,
            Err(_) => return false,
        };
        if n == 0 {
            return true;
        }
        let mut hi = self.offset;
        for (&d, &st) in s.getdims().iter().zip(self.strides.iter()) {
            let span = match (d - 1).checked_mul(st) {
                Some(v) => v,
                None => return false,
            };
            hi = match hi.checked_add(span) {
                Some(v) => v,
                None => return false,
            };
        }
        hi < n
    }
}
