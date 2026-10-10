use thiserror::Error;

/// Failures produced by the dataset, packing, batching, sampling, and
/// data-loader abstractions owned by this crate.
///
/// Every variant carries the values needed to report the exact rejection, so a
/// caller can react without re-deriving them. Limit variants carry the
/// requested value and the configured bound that rejected it; they do not leak
/// the lower-level limit error type across the crate boundary.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum dataerror {
    /// A requested record index is outside the dataset.
    #[error("record index {index} is out of range for {len} records")]
    indexoutofrange {
        /// The requested index.
        index: usize,
        /// The number of records available.
        len: usize,
    },

    /// A sampling operation that requires records was applied to an empty
    /// dataset.
    #[error("cannot sample from an empty dataset")]
    emptydataset,

    /// A sequence length of zero was supplied where a positive length is
    /// required.
    #[error("sequence length must be greater than zero")]
    zeroseqlength,

    /// A batch size of zero was supplied where a positive size is required.
    #[error("batch size must be greater than zero")]
    zerobatchsize,

    /// A batch was requested from an empty set of records.
    #[error("cannot build a batch from zero records")]
    emptybatch,

    /// A record is longer than the target sequence length.
    #[error("record {index} of length {len} exceeds sequence length {seqlength}")]
    recordeceedseqlength {
        /// The position of the offending record in the batch input.
        index: usize,
        /// The length of the offending record.
        len: usize,
        /// The target sequence length.
        seqlength: usize,
    },

    /// A token buffer length does not match the expected element count.
    #[error("token length {actual} does not match expected {expected}")]
    lengthmismatch {
        /// The expected token count.
        expected: usize,
        /// The actual token count.
        actual: usize,
    },

    /// A length, offset, or capacity product does not fit in `usize`.
    #[error("element count overflows usize")]
    elementcountoverflow,

    /// The requested element count exceeds the configured element budget.
    #[error("requested element count {requested} exceeds limit {limit}")]
    elementcountlimitexceeded {
        /// The requested element count.
        requested: usize,
        /// The configured maximum element count.
        limit: usize,
    },

    /// The requested sequence length exceeds the configured sequence budget.
    #[error("requested sequence length {requested} exceeds limit {limit}")]
    sequencelenlimitexceeded {
        /// The requested sequence length.
        requested: usize,
        /// The configured maximum sequence length.
        limit: usize,
    },

    /// The requested batch size exceeds the configured batch budget.
    #[error("requested batch size {requested} exceeds limit {limit}")]
    batchlimitexceeded {
        /// The requested batch size.
        requested: usize,
        /// The configured maximum batch size.
        limit: usize,
    },
}
