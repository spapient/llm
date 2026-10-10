use api::backenderror;
use lib::limiterror;
use thiserror::Error;

/// Failures produced while executing a CPU operator.
///
/// `cpuerror` separates the distinctions a caller of the CPU backend must act
/// on. Kernel-level failures are reported through
/// [`backenderror`], the type fixed by the frozen
/// `api::kernel` contract. Resource-budget and rank rejections are CPU
/// backend policy owned by this crate, and
/// [`backenderror`] intentionally does not model them.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum cpuerror {
    /// A kernel-level failure reported by the frozen backend contract.
    #[error(transparent)]
    backend(#[from] backenderror),

    /// The input exceeds the backend's declared element or byte budget.
    #[error(transparent)]
    limit(#[from] limiterror),

    /// The input rank exceeds the backend's declared maximum rank.
    #[error("input rank {rank} exceeds backend maximum {max}")]
    rankexceeded {
        /// The rejected rank.
        rank: usize,
        /// The backend's declared maximum rank.
        max: usize,
    },
}
