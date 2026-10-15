use crate::deltas::delta;
use crate::error::checkpointerror;
use crate::merge::merge;
use crate::uformat::snapshot;

/// Materializes a chain of deltas onto a base into one full snapshot.
///
/// The chain is folded left: `compact(base, [d1, d2])` equals
/// `merge(merge(base, d1), d2)`. The result's root hash equals the last delta's
/// root hash. Compaction converges a base-plus-N-deltas chain into a single
/// full snapshot, after which the intermediate files are unreachable and can be
/// reclaimed; the caller decides when to run it (for example, once the chain
/// grows past a length or volume threshold) and when the unreachable files are
/// old enough to delete.
///
/// # Errors
///
/// Returns [`checkpointerror::basehashmismatch`] or
/// [`checkpointerror::missingblock`] when any link in the chain is inconsistent.
pub fn compact(base: &snapshot, deltas: &[&delta]) -> Result<snapshot, checkpointerror> {
    let mut current = base.clone();
    for delta in deltas {
        current = merge(&current, delta)?;
    }
    Ok(current)
}

/// Verifies that a chain of deltas links onto a base by content identity.
///
/// Each delta's base identity must equal the previous state's root hash; the
/// chain is `base <- d1 <- d2 <- ...`. A broken link is a structural
/// incompatibility, not a corruption, because the artifacts are individually
/// well formed but the reference chain is wrong.
///
/// # Errors
///
/// Returns [`checkpointerror::basehashmismatch`] at the first broken link.
pub fn verifyprovenance(base: &snapshot, deltas: &[&delta]) -> Result<(), checkpointerror> {
    let mut current = base.roothash();
    for delta in deltas {
        if delta.getbaseid() != current {
            return Err(checkpointerror::basehashmismatch {
                expected: delta.getbaseid(),
                found: current,
            });
        }
        current = delta.getroothash();
    }
    Ok(())
}
