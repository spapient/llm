use api::compute;
use models::causallm;

use crate::error::infererror;
use crate::kvcache::kvstate;

/// Runs the prefill phase: processes every prompt token in order through the
/// cache-aware forward, populating the request-owned key/value cache.
///
/// Returns the next-token logits of the last prompt position.
pub(crate) fn prefillprompt<M: causallm, C: compute>(
    model: &M,
    provider: &C,
    cache: &mut kvstate,
    prompt: &[usize],
) -> Result<Vec<f32>, infererror> {
    let mut logits = Vec::new();
    for &token in prompt {
        logits = model.forward_cached(provider, cache, token)?;
    }
    Ok(logits)
}
