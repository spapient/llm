use crate::dataerror;
use crate::record;
use lib::limits;
use lib::tokenid;

/// Fixed-length sequence packing.
///
/// Packing concatenates the tokens of consecutive records and cuts the stream
/// into sequences of exactly `seqlength` tokens, with a final shorter sequence
/// when the token count is not an exact multiple. Tokens from different records
/// may share one packed sequence; the packing boundary is therefore not a
/// record boundary. This is the standard strategy for turning variable-length
/// documents into dense training sequences.
///
/// # Boundary behavior
///
/// * Empty input yields an empty sequence list.
/// * A record longer than `seqlength` is split across several sequences.
/// * Input whose token count is an exact multiple of `seqlength` yields no
///   trailing partial sequence; every output sequence has length `seqlength`.
/// * Otherwise the last output sequence is shorter than `seqlength` and is left
///   unpadded; callers that require a rectangular batch pad it via [`batch`].
///
/// [`batch`]: crate::batch
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct packing {
    seqlength: usize,
}

impl packing {
    /// Creates a packing strategy with the given target sequence length.
    ///
    /// # Errors
    ///
    /// Returns [`dataerror::zeroseqlength`] when `seqlength` is zero.
    pub fn new(seqlength: usize) -> Result<Self, dataerror> {
        if seqlength == 0 {
            return Err(dataerror::zeroseqlength);
        }
        Ok(Self { seqlength })
    }

    /// Returns the target sequence length.
    #[must_use]
    pub const fn getseqlength(&self) -> usize {
        self.seqlength
    }

    /// Packs `records` into sequences of at most `seqlength` tokens.
    ///
    /// # Errors
    ///
    /// Returns [`dataerror::elementcountoverflow`] when the total token count
    /// does not fit in `usize`, and [`dataerror::elementcountlimitexceeded`]
    /// when it exceeds [`limits::getmaxelements`].
    pub fn pack(&self, records: &[record], lim: &limits) -> Result<Vec<record>, dataerror> {
        let mut total: usize = 0;
        for rec in records {
            total = total
                .checked_add(rec.getlen())
                .ok_or(dataerror::elementcountoverflow)?;
        }
        lim.checkelements(total)
            .map_err(|_| dataerror::elementcountlimitexceeded {
                requested: total,
                limit: lim.getmaxelements(),
            })?;

        let mut packed: Vec<record> = Vec::new();
        let mut buffer: Vec<tokenid> = Vec::new();
        for rec in records {
            for &token in rec.gettokens() {
                buffer.push(token);
                if buffer.len() == self.seqlength {
                    packed.push(record::new(std::mem::take(&mut buffer)));
                }
            }
        }
        if !buffer.is_empty() {
            packed.push(record::new(buffer));
        }
        Ok(packed)
    }
}
