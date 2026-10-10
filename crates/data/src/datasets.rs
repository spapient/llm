use crate::dataerror;
use crate::record;

/// A source of [`record`] values addressed by a dense index.
///
/// The trait is deliberately narrow: a dataset reports how many records it
/// holds and returns a borrowed record for an in-range index. It owns no
/// iteration state, so it can be shared immutably while a [`loader`] keeps the
/// request-scoped cursor privately. Implementations decide how records are
/// stored; the returned reference is valid for as long as the dataset is
/// borrowed.
///
/// [`loader`]: crate::loader
pub trait dataset {
    /// Returns the number of records in the dataset.
    fn getlen(&self) -> usize;

    /// Returns `true` when the dataset holds no records.
    fn isempty(&self) -> bool {
        self.getlen() == 0
    }

    /// Returns the record at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`dataerror::indexoutofrange`] when `index` is greater than or
    /// equal to [`Self::getlen`].
    fn getrecord(&self, index: usize) -> Result<&record, dataerror>;
}
