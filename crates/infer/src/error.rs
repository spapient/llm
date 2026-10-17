use context::contexterror;
use models::modelerror;

/// Failures produced by the generation workflow.
///
/// The enum keeps the distinctions a caller must act on: an underlying model
/// failure, an execution-control failure (cancellation or step-budget
/// exhaustion), an invalid request parameter, an out-of-range token, and a
/// logits vector that cannot be sampled. Underlying errors are carried
/// transparently so no context is lost (ER2).
#[derive(Debug, PartialEq, Eq)]
pub enum infererror {
    /// A model forward pass failed.
    model(modelerror),

    /// Execution control failed, for example cancellation or a step budget.
    context(contexterror),

    /// A generation parameter is invalid.
    invalidrequest {
        /// The parameter that was rejected.
        field: &'static str,
        /// A description of why the value is invalid.
        detail: String,
    },

    /// A sampling parameter is invalid.
    invalidparams {
        /// A static explanation of the rejected parameter.
        detail: &'static str,
    },

    /// A token is outside the model vocabulary.
    tokenoutofrange {
        /// The rejected token id.
        token: usize,
        /// The model vocabulary size.
        vocab: usize,
    },

    /// The logits vector is empty, so no token can be selected.
    emptylogits,

    /// The logits carry no finite probability mass and cannot be sampled.
    degeneratelogits,
}

impl std::fmt::Display for infererror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::model(f0) => std::fmt::Display::fmt(f0, f),
            Self::context(f0) => std::fmt::Display::fmt(f0, f),
            Self::invalidrequest { field, detail } => {
                write!(f, "invalid generation request for {field}: {detail}")
            }
            Self::invalidparams { detail } => write!(f, "invalid sampling parameters: {detail}"),
            Self::tokenoutofrange { token, vocab } => {
                write!(f, "token {token} is outside the vocabulary of size {vocab}")
            }
            Self::emptylogits => f.write_str("logits are empty"),
            Self::degeneratelogits => {
                f.write_str("logits cannot be sampled: no finite probability mass")
            }
        }
    }
}

impl std::error::Error for infererror {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::model(f0) => Some(f0),
            Self::context(f0) => Some(f0),
            _ => None,
        }
    }
}

impl From<modelerror> for infererror {
    fn from(v: modelerror) -> Self {
        Self::model(v)
    }
}

impl From<contexterror> for infererror {
    fn from(v: contexterror) -> Self {
        Self::context(v)
    }
}
