use api::backenderror;
use lib::limiterror;

/// Failures produced while executing a CPU operator.
///
/// `cpuerror` separates the distinctions a caller of the CPU backend must act
/// on. Kernel-level failures are reported through
/// [`backenderror`], the type fixed by the frozen
/// `api::kernel` contract. Resource-budget and rank rejections are CPU
/// backend policy owned by this crate, and
/// [`backenderror`] intentionally does not model them.
#[derive(Debug, PartialEq, Eq)]
pub enum cpuerror {
    /// A kernel-level failure reported by the frozen backend contract.
    backend(backenderror),

    /// The input exceeds the backend's declared element or byte budget.
    limit(limiterror),

    /// The input rank exceeds the backend's declared maximum rank.
    rankexceeded {
        /// The rejected rank.
        rank: usize,
        /// The backend's declared maximum rank.
        max: usize,
    },
}

impl std::fmt::Display for cpuerror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::backend(f0) => std::fmt::Display::fmt(f0, f),
            Self::limit(f0) => std::fmt::Display::fmt(f0, f),
            Self::rankexceeded { rank, max } => {
                write!(f, "input rank {rank} exceeds backend maximum {max}")
            }
        }
    }
}

impl std::error::Error for cpuerror {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::backend(f0) => Some(f0),
            Self::limit(f0) => Some(f0),
            _ => None,
        }
    }
}

impl From<backenderror> for cpuerror {
    fn from(v: backenderror) -> Self {
        Self::backend(v)
    }
}

impl From<limiterror> for cpuerror {
    fn from(v: limiterror) -> Self {
        Self::limit(v)
    }
}
