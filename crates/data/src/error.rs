/// Failures produced by the dataset, packing, batching, sampling, and
/// data-loader abstractions owned by this crate.
///
/// Every variant carries the values needed to report the exact rejection, so a
/// caller can react without re-deriving them. Limit variants carry the
/// requested value and the configured bound that rejected it; they do not leak
/// the lower-level limit error type across the crate boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum dataerror {
    /// A requested record index is outside the dataset.
    indexoutofrange {
        /// The requested index.
        index: usize,
        /// The number of records available.
        len: usize,
    },

    /// A sampling operation that requires records was applied to an empty
    /// dataset.
    emptydataset,

    /// A sequence length of zero was supplied where a positive length is
    /// required.
    zeroseqlength,

    /// A batch size of zero was supplied where a positive size is required.
    zerobatchsize,

    /// A batch was requested from an empty set of records.
    emptybatch,

    /// A record is longer than the target sequence length.
    recordeceedseqlength {
        /// The position of the offending record in the batch input.
        index: usize,
        /// The length of the offending record.
        len: usize,
        /// The target sequence length.
        seqlength: usize,
    },

    /// A token buffer length does not match the expected element count.
    lengthmismatch {
        /// The expected token count.
        expected: usize,
        /// The actual token count.
        actual: usize,
    },

    /// A length, offset, or capacity product does not fit in `usize`.
    elementcountoverflow,

    /// The requested element count exceeds the configured element budget.
    elementcountlimitexceeded {
        /// The requested element count.
        requested: usize,
        /// The configured maximum element count.
        limit: usize,
    },

    /// The requested sequence length exceeds the configured sequence budget.
    sequencelenlimitexceeded {
        /// The requested sequence length.
        requested: usize,
        /// The configured maximum sequence length.
        limit: usize,
    },

    /// The requested batch size exceeds the configured batch budget.
    batchlimitexceeded {
        /// The requested batch size.
        requested: usize,
        /// The configured maximum batch size.
        limit: usize,
    },
}

impl std::fmt::Display for dataerror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::indexoutofrange { index, len } => {
                write!(f, "record index {index} is out of range for {len} records")
            }
            Self::emptydataset => f.write_str("cannot sample from an empty dataset"),
            Self::zeroseqlength => f.write_str("sequence length must be greater than zero"),
            Self::zerobatchsize => f.write_str("batch size must be greater than zero"),
            Self::emptybatch => f.write_str("cannot build a batch from zero records"),
            Self::recordeceedseqlength {
                index,
                len,
                seqlength,
            } => write!(
                f,
                "record {index} of length {len} exceeds sequence length {seqlength}"
            ),
            Self::lengthmismatch { expected, actual } => write!(
                f,
                "token length {actual} does not match expected {expected}"
            ),
            Self::elementcountoverflow => f.write_str("element count overflows usize"),
            Self::elementcountlimitexceeded { requested, limit } => write!(
                f,
                "requested element count {requested} exceeds limit {limit}"
            ),
            Self::sequencelenlimitexceeded { requested, limit } => write!(
                f,
                "requested sequence length {requested} exceeds limit {limit}"
            ),
            Self::batchlimitexceeded { requested, limit } => {
                write!(f, "requested batch size {requested} exceeds limit {limit}")
            }
        }
    }
}

impl std::error::Error for dataerror {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
