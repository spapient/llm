use api::{recorder, session};
use context::executioncontext;
use models::{cachedsubmission, causallm, kvcache, llamamodel, modelerror};

use crate::error::infererror;
use crate::generation::generationresult;
use crate::kvcache::kvstate;
use crate::request::generationrequest;
use crate::stopping::stopreason;
use crate::stream::streamsink;

/// A causal model whose cache-aware forward can be recorded into a submission.
///
/// `llamamodel` implements this trait. It exists so the submission engine can be
/// generic without adding a second provider parameter to [`causallm`].
pub trait causallmsubmission: causallm {
    /// Records one cache-aware forward pass into `sub`, returning the logits and
    /// each layer's new key/value handles.
    ///
    /// # Errors
    ///
    /// Returns a [`modelerror`] when the token is out of range, an operand shape
    /// is inconsistent, or a recorded operator fails.
    fn forwardsubmission_cached<R: recorder, K: kvcache>(
        &self,
        sub: &mut R,
        cache: &K,
        token: usize,
    ) -> Result<cachedsubmission, modelerror>;
}

impl causallmsubmission for llamamodel {
    fn forwardsubmission_cached<R: recorder, K: kvcache>(
        &self,
        sub: &mut R,
        cache: &K,
        token: usize,
    ) -> Result<cachedsubmission, modelerror> {
        llamamodel::forwardsubmission_cached(self, sub, cache, token)
    }
}

/// Orchestrates prefill and decode through one request-scoped persistent
/// session.
///
/// Every prompt and decode token records into the session, applies it once,
/// appends the new key/value rows to the cache, and samples from the returned
/// logits. The session pins the model weights on first use, so later tokens do
/// not re-upload them. The first prompt token records against an empty cache,
/// which the model represents with zero-length cached operands. The engine
/// borrows the model and a session factory and owns no per-request state.
pub struct submissionengine<'a, M: causallmsubmission, F> {
    model: &'a M,
    factory: F,
}

impl<'a, M, F, S> submissionengine<'a, M, F>
where
    M: causallmsubmission,
    F: Fn() -> S,
    S: session,
{
    /// Creates an engine over a borrowed model and submission factory.
    #[must_use]
    pub fn new(model: &'a M, factory: F) -> Self {
        Self { model, factory }
    }

    /// Runs one generation attempt for `request`, emitting tokens to `sink`.
    ///
    /// # Errors
    ///
    /// Returns [`infererror`] under the same conditions as the compute engine:
    /// an invalid request, a failed forward or submission, a sampling failure,
    /// or a cancelled/step-exhausted execution context.
    pub fn generate<G: streamsink>(
        &self,
        exec: &executioncontext,
        request: &mut generationrequest,
        sink: &mut G,
    ) -> Result<generationresult, infererror> {
        request.prepare(self.model.getconfig())?;
        exec.getcancellation().checkpoint()?;

        let prompt = request.getprompt().to_vec();
        let mut sub = (self.factory)();
        let mut logits = Vec::new();
        {
            let cache = request.getkvcachemut();
            for &token in &prompt {
                logits = self.decodestep(&mut sub, cache, token)?;
            }
        }
        let mut steps = 0usize;
        let reason = loop {
            exec.checkpoint()?;
            steps = steps.saturating_add(1);
            let token = request.sample(&logits)?;

            if request.getstop().iseos(token) {
                break stopreason::endoftoken;
            }
            request.pushgenerated(token)?;
            let index = request.getgeneratedlen().saturating_sub(1);
            if !sink.ongenerated(index, token) {
                break stopreason::stopped;
            }
            if request.getgeneratedlen() >= request.getmaxnewtokens() {
                break stopreason::maxlength;
            }
            logits = {
                let cache = request.getkvcachemut();
                self.decodestep(&mut sub, cache, token)?
            };
        };
        sub.reset();

        sink.onstop(reason);
        Ok(generationresult::new(
            request.getgenerated().to_vec(),
            reason,
            steps,
        ))
    }

    /// Records, applies, and appends one decode step.
    fn decodestep(
        &self,
        sub: &mut S,
        cache: &mut kvstate,
        token: usize,
    ) -> Result<Vec<f32>, infererror> {
        let handles = self.model.forwardsubmission_cached(sub, cache, token)?;
        let layers = cache.getlayers();
        let mut requested = Vec::with_capacity(1 + 2 * layers);
        requested.push(handles.logits);
        requested.extend(handles.keys.iter().copied());
        requested.extend(handles.values.iter().copied());
        let mut out = sub.apply(&requested).map_err(modelerror::from)?;
        let mut keyrows = out.split_off(1);
        let valuerows = keyrows.split_off(layers);
        for layer in 0..layers {
            cache.append(layer, &keyrows[layer], &valuerows[layer])?;
        }
        Ok(out.remove(0).getdata().to_vec())
    }
}
