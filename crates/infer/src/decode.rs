use api::compute;
use models::causallm;

use crate::error::infererror;
use crate::kvcache::kvstate;

/// Runs one decode step for a single newly generated token, reading and
/// appending the request-owned key/value cache.
///
/// Returns the next-token logits of that token.
pub(crate) fn decodestep<M: causallm, C: compute>(
    model: &M,
    provider: &C,
    cache: &mut kvstate,
    token: usize,
) -> Result<Vec<f32>, infererror> {
    model
        .forward_cached(provider, cache, token)
        .map_err(infererror::from)
}
