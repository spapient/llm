use crate::tensorerror;
use lib::limits;

/// The logical dimensions of a tensor.
///
/// A shape is an ordered, non-empty list of dimensions ordered from most
/// significant to least significant. The rank is the number of dimensions and
/// the element count is the checked product of the dimensions. An empty
/// dimension list is rejected, so every shape has rank at least one.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct shape {
    dims: Vec<usize>,
}

impl shape {
    /// Creates a shape from its dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`tensorerror::emptyshape`] when `dims` is empty.
    pub fn new(dims: Vec<usize>) -> Result<Self, tensorerror> {
        if dims.is_empty() {
            return Err(tensorerror::emptyshape);
        }
        Ok(Self { dims })
    }

    /// Returns the rank: the number of dimensions.
    #[must_use]
    pub fn getrank(&self) -> usize {
        self.dims.len()
    }

    /// Returns the dimensions as a slice, most significant first.
    #[must_use]
    pub fn getdims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns `true` when the shape contains no elements.
    ///
    /// A shape is empty when any dimension is zero. A shape such as `[1]` holds
    /// a single element and is not empty.
    #[must_use]
    pub fn isempty(&self) -> bool {
        self.dims.contains(&0)
    }

    /// Returns the total element count.
    ///
    /// # Errors
    ///
    /// Returns [`tensorerror::elementcountoverflow`] when the product of the
    /// dimensions does not fit in `usize`.
    pub fn getelementcount(&self) -> Result<usize, tensorerror> {
        let mut n: usize = 1;
        for &d in &self.dims {
            n = n
                .checked_mul(d)
                .ok_or_else(|| tensorerror::elementcountoverflow {
                    dims: self.dims.clone(),
                })?;
        }
        Ok(n)
    }

    /// Validates the element count against an element budget.
    ///
    /// Returns the validated element count on success.
    ///
    /// # Errors
    ///
    /// Returns the error from [`Self::getelementcount`], or
    /// [`tensorerror::elementcountlimitexceeded`] when the count is strictly
    /// greater than [`limits::getmaxelements`].
    pub fn checklimits(&self, lim: &limits) -> Result<usize, tensorerror> {
        let n = self.getelementcount()?;
        lim.checkelements(n)
            .map_err(|_| tensorerror::elementcountlimitexceeded {
                requested: n,
                limit: lim.getmaxelements(),
            })?;
        Ok(n)
    }
}
