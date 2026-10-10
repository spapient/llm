use api::compute;
use models::causallm;

use crate::error::infererror;
use crate::generation::nextlogits;

/// Runs the prefill phase: one forward pass over the whole prompt.
///
/// Returns the next-token logits of the last prompt position. Prefill is the
/// only phase that processes more than one token at a time while the correctness
/// path recomputes the full context.
pub(crate) fn prefillprompt<M: causallm, C: compute>(
    model: &M,
    provider: &C,
    prompt: &[usize],
) -> Result<Vec<f32>, infererror> {
    nextlogits(model, provider, prompt)
}
