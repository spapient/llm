use crate::dataerror;
use crate::record;
use lib::limits;
use lib::tokenid;

/// A rectangular batch of token identifiers in row-major order.
///
/// A batch owns a flat token buffer of exactly `batchsize * seqlength` tokens.
/// Row `r` occupies `[r * seqlength, (r + 1) * seqlength)`. Every row has the
/// same length, so a batch can be viewed as a dense `[batchsize, seqlength]`
/// matrix without gaps.
///
/// # Boundary behavior
///
/// [`Self::fromrecords`] pads every record shorter than `seqlength` with the
/// caller-supplied padding token. A record longer than `seqlength` is rejected
/// with [`dataerror::recordeceedseqlength`] rather than truncated, so a
/// misconfigured `seqlength` fails loudly instead of silently dropping tokens.
/// An empty record set is rejected with [`dataerror::emptybatch`]; a batch
/// always has at least one row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct batch {
    tokens: Vec<tokenid>,
    batchsize: usize,
    seqlength: usize,
}

impl batch {
    /// Creates a batch from a flat token buffer.
    ///
    /// # Errors
    ///
    /// Returns [`dataerror::zerobatchsize`] or [`dataerror::zeroseqlength`] when
    /// a dimension is zero, [`dataerror::lengthmismatch`] when `tokens.len()`
    /// differs from `batchsize * seqlength`,
    /// [`dataerror::elementcountoverflow`] when that product does not fit in
    /// `usize`, and the batch, sequence, or element limit variants when a
    /// dimension or the total element count exceeds the corresponding bound in
    /// `lim`.
    pub fn new(
        tokens: Vec<tokenid>,
        batchsize: usize,
        seqlength: usize,
        lim: &limits,
    ) -> Result<Self, dataerror> {
        checkdimensions(batchsize, seqlength, lim)?;
        let expected = batchsize
            .checked_mul(seqlength)
            .ok_or(dataerror::elementcountoverflow)?;
        if tokens.len() != expected {
            return Err(dataerror::lengthmismatch {
                expected,
                actual: tokens.len(),
            });
        }
        lim.checkelements(expected)
            .map_err(|_| dataerror::elementcountlimitexceeded {
                requested: expected,
                limit: lim.getmaxelements(),
            })?;
        Ok(Self {
            tokens,
            batchsize,
            seqlength,
        })
    }

    /// Builds a rectangular batch by padding each record to `seqlength`.
    ///
    /// # Errors
    ///
    /// Returns [`dataerror::emptybatch`] when `records` is empty, plus the same
    /// validation errors as [`Self::new`] and
    /// [`dataerror::recordeceedseqlength`] when a record is longer than
    /// `seqlength`.
    pub fn fromrecords(
        records: &[record],
        seqlength: usize,
        pad: tokenid,
        lim: &limits,
    ) -> Result<Self, dataerror> {
        if records.is_empty() {
            return Err(dataerror::emptybatch);
        }
        let batchsize = records.len();
        checkdimensions(batchsize, seqlength, lim)?;
        let expected = batchsize
            .checked_mul(seqlength)
            .ok_or(dataerror::elementcountoverflow)?;
        lim.checkelements(expected)
            .map_err(|_| dataerror::elementcountlimitexceeded {
                requested: expected,
                limit: lim.getmaxelements(),
            })?;

        let mut tokens: Vec<tokenid> = Vec::new();
        tokens
            .try_reserve_exact(expected)
            .map_err(|_| dataerror::elementcountlimitexceeded {
                requested: expected,
                limit: lim.getmaxelements(),
            })?;
        for (index, rec) in records.iter().enumerate() {
            let row = rec.gettokens();
            if row.len() > seqlength {
                return Err(dataerror::recordeceedseqlength {
                    index,
                    len: row.len(),
                    seqlength,
                });
            }
            tokens.extend_from_slice(row);
            tokens.resize(tokens.len() + (seqlength - row.len()), pad);
        }
        Self::new(tokens, batchsize, seqlength, lim)
    }

    /// Returns the number of rows.
    #[must_use]
    pub const fn getbatchsize(&self) -> usize {
        self.batchsize
    }

    /// Returns the length of every row.
    #[must_use]
    pub const fn getseqlength(&self) -> usize {
        self.seqlength
    }

    /// Returns the flat token buffer in row-major order.
    #[must_use]
    pub fn gettokens(&self) -> &[tokenid] {
        &self.tokens
    }

    /// Returns row `row`.
    ///
    /// # Errors
    ///
    /// Returns [`dataerror::indexoutofrange`] when `row` is greater than or
    /// equal to [`Self::getbatchsize`].
    pub fn getrow(&self, row: usize) -> Result<&[tokenid], dataerror> {
        if row >= self.batchsize {
            return Err(dataerror::indexoutofrange {
                index: row,
                len: self.batchsize,
            });
        }
        let start = row
            .checked_mul(self.seqlength)
            .ok_or(dataerror::elementcountoverflow)?;
        let end = start
            .checked_add(self.seqlength)
            .ok_or(dataerror::elementcountoverflow)?;
        self.tokens
            .get(start..end)
            .ok_or(dataerror::lengthmismatch {
                expected: end,
                actual: self.tokens.len(),
            })
    }
}

fn checkdimensions(batchsize: usize, seqlength: usize, lim: &limits) -> Result<(), dataerror> {
    if batchsize == 0 {
        return Err(dataerror::zerobatchsize);
    }
    if seqlength == 0 {
        return Err(dataerror::zeroseqlength);
    }
    lim.checkbatch(batchsize)
        .map_err(|_| dataerror::batchlimitexceeded {
            requested: batchsize,
            limit: lim.getmaxbatch(),
        })?;
    lim.checksequencelen(seqlength)
        .map_err(|_| dataerror::sequencelenlimitexceeded {
            requested: seqlength,
            limit: lim.getmaxsequencelen(),
        })?;
    Ok(())
}
