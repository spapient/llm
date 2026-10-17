use api::compute;
use context::executioncontext;
use models::causallm;

use crate::decode::decodestep;
use crate::error::infererror;
use crate::generation::generationresult;
use crate::prefill::prefillprompt;
use crate::request::generationrequest;
use crate::stopping::stopreason;
use crate::stream::streamsink;

/// Orchestrates prefill and decode over an immutable causal language model.
///
/// The engine borrows the model and the compute provider and owns no
/// per-request state, so one engine serves independent requests (OW2). It
/// performs the prefill pass, then repeatedly samples a token, appends it to the
/// request context, and runs a decode pass until a stop condition, the new-token
/// cap, or the sink ends generation.
///
/// Cancellation and the step budget are enforced by the caller-owned
/// [`executioncontext`] checkpoint at every step, so a cancelled or
/// budget-exhausted run stops before the next forward pass and returns a
/// [`infererror::context`] error rather than a successful result (CC5, OW3).
pub struct generationengine<'a, M: causallm, C: compute> {
    model: &'a M,
    provider: &'a C,
}

impl<'a, M: causallm, C: compute> generationengine<'a, M, C> {
    /// Creates an engine over a borrowed model and compute provider.
    #[must_use]
    pub fn new(model: &'a M, provider: &'a C) -> Self {
        Self { model, provider }
    }

    /// Runs one generation attempt for `request`, emitting tokens to `sink`.
    ///
    /// The request is reset to a fresh attempt first, so residual state from a
    /// prior run cannot affect the result. All mutable state is owned by
    /// `request`; the engine allocates nothing that outlives the call.
    ///
    /// # Errors
    ///
    /// Returns [`infererror`] when the request is invalid for the model, when a
    /// forward pass fails, when the sampling fails, or when the execution
    /// checkpoint reports cancellation or step-budget exhaustion.
    pub fn generate<S: streamsink>(
        &self,
        exec: &executioncontext,
        request: &mut generationrequest,
        sink: &mut S,
    ) -> Result<generationresult, infererror> {
        request.prepare(self.model.getconfig())?;
        exec.getcancellation().checkpoint()?;

        let prompt = request.getprompt().to_vec();
        let mut logits = {
            let cache = request.getkvcachemut();
            prefillprompt(self.model, self.provider, cache, &prompt)?
        };
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
                decodestep(self.model, self.provider, cache, token)?
            };
        };

        sink.onstop(reason);
        Ok(generationresult::new(
            request.getgenerated().to_vec(),
            reason,
            steps,
        ))
    }
}
