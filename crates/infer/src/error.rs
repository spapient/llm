use context::contexterror;
use models::modelerror;
use thiserror::Error;

/// Failures produced by the generation workflow.
///
/// The enum keeps the distinctions a caller must act on: an underlying model
/// failure, an execution-control failure (cancellation or step-budget
/// exhaustion), an invalid request parameter, an out-of-range token, and a
/// logits vector that cannot be sampled. Underlying errors are carried
/// transparently so no context is lost (ER2).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum infererror {
    /// A model forward pass failed.
    #[error(transparent)]
    model(#[from] modelerror),

    /// Execution control failed, for example cancellation or a step budget.
    #[error(transparent)]
    context(#[from] contexterror),

    /// A generation parameter is invalid.
    #[error("invalid generation request for {field}: {detail}")]
    invalidrequest {
        /// The parameter that was rejected.
        field: &'static str,
        /// A description of why the value is invalid.
        detail: String,
    },

    /// A sampling parameter is invalid.
    #[error("invalid sampling parameters: {detail}")]
    invalidparams {
        /// A static explanation of the rejected parameter.
        detail: &'static str,
    },

    /// A token is outside the model vocabulary.
    #[error("token {token} is outside the vocabulary of size {vocab}")]
    tokenoutofrange {
        /// The rejected token id.
        token: usize,
        /// The model vocabulary size.
        vocab: usize,
    },

    /// The logits vector is empty, so no token can be selected.
    #[error("logits are empty")]
    emptylogits,

    /// The logits carry no finite probability mass and cannot be sampled.
    #[error("logits cannot be sampled: no finite probability mass")]
    degeneratelogits,
}
