use crate::error::limiterror;

/// An immutable resource budget applied before allocation or execution.
///
/// A `limits` value holds explicit upper bounds for element counts, byte sizes,
/// sequence lengths, and batch sizes. It defines no defaults: every bound is
/// supplied by the caller so that no undocumented limit is invented.
///
/// All limits are inclusive upper bounds. A requested value equal to the
/// corresponding bound is accepted; only a strictly greater value is rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct limits {
    maxelements: usize,
    maxbytes: usize,
    maxsequencelen: usize,
    maxbatch: usize,
}

impl limits {
    /// Creates a limit set from explicit inclusive upper bounds.
    #[must_use]
    pub const fn new(
        maxelements: usize,
        maxbytes: usize,
        maxsequencelen: usize,
        maxbatch: usize,
    ) -> Self {
        Self {
            maxelements,
            maxbytes,
            maxsequencelen,
            maxbatch,
        }
    }

    /// Returns the maximum element count.
    #[must_use]
    pub const fn getmaxelements(self) -> usize {
        self.maxelements
    }

    /// Returns the maximum byte size.
    #[must_use]
    pub const fn getmaxbytes(self) -> usize {
        self.maxbytes
    }

    /// Returns the maximum sequence length.
    #[must_use]
    pub const fn getmaxsequencelen(self) -> usize {
        self.maxsequencelen
    }

    /// Returns the maximum batch size.
    #[must_use]
    pub const fn getmaxbatch(self) -> usize {
        self.maxbatch
    }

    /// Checks a requested element count against the element bound.
    ///
    /// # Errors
    ///
    /// Returns [`limiterror::maxelementsexceeded`] when `requested` is strictly
    /// greater than [`Self::getmaxelements`].
    pub const fn checkelements(&self, requested: usize) -> Result<(), limiterror> {
        if requested > self.maxelements {
            return Err(limiterror::maxelementsexceeded {
                requested,
                limit: self.maxelements,
            });
        }
        Ok(())
    }

    /// Checks a requested byte size against the byte bound.
    ///
    /// # Errors
    ///
    /// Returns [`limiterror::maxbytesexceeded`] when `requested` is strictly
    /// greater than [`Self::getmaxbytes`].
    pub const fn checkbytes(&self, requested: usize) -> Result<(), limiterror> {
        if requested > self.maxbytes {
            return Err(limiterror::maxbytesexceeded {
                requested,
                limit: self.maxbytes,
            });
        }
        Ok(())
    }

    /// Checks a requested sequence length against the sequence bound.
    ///
    /// # Errors
    ///
    /// Returns [`limiterror::maxsequencelenexceeded`] when `requested` is
    /// strictly greater than [`Self::getmaxsequencelen`].
    pub const fn checksequencelen(&self, requested: usize) -> Result<(), limiterror> {
        if requested > self.maxsequencelen {
            return Err(limiterror::maxsequencelenexceeded {
                requested,
                limit: self.maxsequencelen,
            });
        }
        Ok(())
    }

    /// Checks a requested batch size against the batch bound.
    ///
    /// # Errors
    ///
    /// Returns [`limiterror::maxbatchexceeded`] when `requested` is strictly
    /// greater than [`Self::getmaxbatch`].
    pub const fn checkbatch(&self, requested: usize) -> Result<(), limiterror> {
        if requested > self.maxbatch {
            return Err(limiterror::maxbatchexceeded {
                requested,
                limit: self.maxbatch,
            });
        }
        Ok(())
    }
}
