use api::compute;
use models::causallm;

use crate::error::infererror;
use crate::generation::nextlogits;

/// Runs one decode step: one forward pass over the accumulated context.
///
/// Returns the next-token logits of the last context position. While the
/// correctness path recomputes the full context, this is where a cache-aware
/// implementation would process only the newly appended token.
pub(crate) fn decodestep<M: causallm, C: compute>(
    model: &M,
    provider: &C,
    context: &[usize],
) -> Result<Vec<f32>, infererror> {
    nextlogits(model, provider, context)
}
