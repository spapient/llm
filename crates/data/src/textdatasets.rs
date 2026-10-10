use crate::dataerror;
use crate::dataset;
use crate::record;
use lib::limits;
use lib::tokenid;

/// An in-memory dataset of tokenized documents.
///
/// `textdataset` is the concrete [`dataset`] used by the loader. It owns a list
/// of records and the accumulated token count. Converting raw text to token
/// identifiers is deliberately not performed here: that composition belongs to
/// a higher layer that owns the tokenizer, so this crate never depends on
/// tokenization (see `docs/crate-boundaries.md` section 5.9). Callers pass
/// already-tokenized documents via [`Self::new`] or [`Self::fromtokens`].
///
/// # Resource limits
///
/// Construction validates both the record count and the total token count
/// against the element budget in the supplied [`limits`]. A dataset that would
/// exceed either bound is rejected with
/// [`dataerror::elementcountlimitexceeded`] instead of being buffered.
#[derive(Clone, Debug)]
pub struct textdataset {
    records: Vec<record>,
    totaltokens: usize,
}

impl textdataset {
    /// Creates a dataset from records, validating it against `lim`.
    ///
    /// # Errors
    ///
    /// Returns [`dataerror::elementcountlimitexceeded`] when the record count or
    /// the total token count exceeds [`limits::getmaxelements`], and
    /// [`dataerror::elementcountoverflow`] when the total token count does not
    /// fit in `usize`.
    pub fn new(records: Vec<record>, lim: &limits) -> Result<Self, dataerror> {
        let count = records.len();
        lim.checkelements(count)
            .map_err(|_| dataerror::elementcountlimitexceeded {
                requested: count,
                limit: lim.getmaxelements(),
            })?;

        let mut totaltokens: usize = 0;
        for rec in &records {
            totaltokens = totaltokens
                .checked_add(rec.getlen())
                .ok_or(dataerror::elementcountoverflow)?;
        }
        lim.checkelements(totaltokens)
            .map_err(|_| dataerror::elementcountlimitexceeded {
                requested: totaltokens,
                limit: lim.getmaxelements(),
            })?;

        Ok(Self {
            records,
            totaltokens,
        })
    }

    /// Creates a dataset from already-tokenized documents.
    ///
    /// Each inner slice becomes one record.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::new`].
    pub fn fromtokens(documents: Vec<Vec<tokenid>>, lim: &limits) -> Result<Self, dataerror> {
        let records = documents.into_iter().map(record::new).collect();
        Self::new(records, lim)
    }

    /// Returns the total number of tokens across all records.
    #[must_use]
    pub const fn gettotaltokens(&self) -> usize {
        self.totaltokens
    }
}

impl dataset for textdataset {
    fn getlen(&self) -> usize {
        self.records.len()
    }

    fn getrecord(&self, index: usize) -> Result<&record, dataerror> {
        self.records.get(index).ok_or(dataerror::indexoutofrange {
            index,
            len: self.records.len(),
        })
    }
}
