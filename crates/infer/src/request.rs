use models::llamaconfig;

use crate::error::infererror;
use crate::kvcache::kvstate;
use crate::sampling::{sampler, samplingparams};
use crate::stopping::stopcondition;

/// Static parameters for one generation request.
///
/// The parameters name the stop criteria, the sampling parameters, and the
/// capacity a request needs: the maximum context length and the shape of the
/// request-owned key/value cache. They are validated once and then bound to a
/// [`generationrequest`].
#[derive(Clone, Debug, PartialEq)]
pub struct generationparams {
    stop: stopcondition,
    sampling: samplingparams,
    maxseq: usize,
    kvlayers: usize,
    kvwidth: usize,
}

impl generationparams {
    /// Builds validated request parameters.
    ///
    /// # Errors
    ///
    /// Returns [`infererror::invalidrequest`] when `maxseq`, `kvlayers`, or
    /// `kvwidth` is zero.
    pub fn new(
        stop: stopcondition,
        sampling: samplingparams,
        maxseq: usize,
        kvlayers: usize,
        kvwidth: usize,
    ) -> Result<Self, infererror> {
        if maxseq == 0 {
            return Err(infererror::invalidrequest {
                field: "maxseq",
                detail: "must be non-zero".to_string(),
            });
        }
        if kvlayers == 0 {
            return Err(infererror::invalidrequest {
                field: "kvlayers",
                detail: "must be non-zero".to_string(),
            });
        }
        if kvwidth == 0 {
            return Err(infererror::invalidrequest {
                field: "kvwidth",
                detail: "must be non-zero".to_string(),
            });
        }
        Ok(Self {
            stop,
            sampling,
            maxseq,
            kvlayers,
            kvwidth,
        })
    }

    /// Builds request parameters sized from a validated model configuration.
    ///
    /// This is the ordinary constructor: the context capacity and cache shape
    /// are taken from the configuration the model was built with.
    #[must_use]
    pub fn formodel(config: &llamaconfig, stop: stopcondition, sampling: samplingparams) -> Self {
        Self {
            stop,
            sampling,
            maxseq: config.getmaxseq(),
            kvlayers: config.getnlayers(),
            kvwidth: config.getkvwidth(),
        }
    }

    /// Returns the stop condition.
    #[must_use]
    pub fn getstop(&self) -> &stopcondition {
        &self.stop
    }

    /// Returns the sampling parameters.
    #[must_use]
    pub const fn getsampling(&self) -> samplingparams {
        self.sampling
    }

    /// Returns the maximum context length.
    #[must_use]
    pub const fn getmaxseq(&self) -> usize {
        self.maxseq
    }

    /// Returns the cache layer count.
    #[must_use]
    pub const fn getkvlayers(&self) -> usize {
        self.kvlayers
    }

    /// Returns the cache width.
    #[must_use]
    pub const fn getkvwidth(&self) -> usize {
        self.kvwidth
    }
}

/// The request-scoped mutable state of one generation attempt.
///
/// A `generationrequest` owns everything that changes while generating: the
/// immutable prompt, the accumulated context (prompt followed by generated
/// tokens), the sampler position, the stop criteria, and the request-owned
/// [`kvstate`]. It is created per request and never stored in shared model
/// weights, so two requests cannot share mutable generation state (OW2). All
/// state is released when the request is dropped; [`Self::reset`] returns it to
/// the state of a fresh attempt so a request cannot carry residual state from a
/// cancelled or failed run (OW3).
#[derive(Debug)]
pub struct generationrequest {
    prompt: Vec<usize>,
    context: Vec<usize>,
    stop: stopcondition,
    sampler: sampler,
    maxseq: usize,
    kvcache: kvstate,
}

impl generationrequest {
    /// Builds a request from a prompt and validated parameters.
    ///
    /// # Errors
    ///
    /// Returns [`infererror::invalidrequest`] when the prompt is empty, when the
    /// prompt is longer than `maxseq`, or when the prompt plus `maxnewtokens`
    /// exceeds `maxseq`. Returns [`infererror`] from [`kvstate::new`] when the
    /// cache dimensions are invalid.
    pub fn new(prompt: Vec<usize>, params: generationparams) -> Result<Self, infererror> {
        if prompt.is_empty() {
            return Err(infererror::invalidrequest {
                field: "prompt",
                detail: "must be non-empty".to_string(),
            });
        }
        if prompt.len() > params.maxseq {
            return Err(infererror::invalidrequest {
                field: "prompt",
                detail: format!("length {} exceeds maxseq {}", prompt.len(), params.maxseq),
            });
        }
        let maxnew = params.stop.getmaxnewtokens();
        let requested_len = prompt.len().checked_add(maxnew).ok_or_else(|| {
            infererror::invalidrequest {
                field: "maxnewtokens",
                detail: "prompt length plus maxnewtokens overflows usize".to_string(),
            }
        })?;
        if requested_len > params.maxseq {
            return Err(infererror::invalidrequest {
                field: "maxnewtokens",
                detail: format!(
                    "prompt length {} plus maxnewtokens {} exceeds maxseq {}",
                    prompt.len(),
                    maxnew,
                    params.maxseq
                ),
            });
        }
        let kvcache = kvstate::new(params.kvlayers, params.kvwidth, params.maxseq)?;
        let context = prompt.clone();
        let maxseq = params.maxseq;
        Ok(Self {
            prompt,
            context,
            stop: params.stop,
            sampler: sampler::new(params.sampling),
            maxseq,
            kvcache,
        })
    }

    /// Returns the immutable prompt.
    #[must_use]
    pub fn getprompt(&self) -> &[usize] {
        &self.prompt
    }

    /// Returns the accumulated context: the prompt followed by generated tokens.
    #[must_use]
    pub fn getcontext(&self) -> &[usize] {
        &self.context
    }

    /// Returns the generated tokens in order.
    #[must_use]
    pub fn getgenerated(&self) -> &[usize] {
        &self.context[self.prompt.len()..]
    }

    /// Returns the number of generated tokens.
    #[must_use]
    pub fn getgeneratedlen(&self) -> usize {
        self.context.len() - self.prompt.len()
    }

    /// Returns the stop condition.
    #[must_use]
    pub fn getstop(&self) -> &stopcondition {
        &self.stop
    }

    /// Returns the maximum number of new tokens.
    #[must_use]
    pub const fn getmaxnewtokens(&self) -> usize {
        self.stop.getmaxnewtokens()
    }

    /// Returns the maximum context length.
    #[must_use]
    pub const fn getmaxseq(&self) -> usize {
        self.maxseq
    }

    /// Returns the request-owned key/value cache.
    #[must_use]
    pub fn getkvcache(&self) -> &kvstate {
        &self.kvcache
    }

    /// Returns the request-owned key/value cache mutably.
    #[must_use]
    pub fn getkvcachemut(&mut self) -> &mut kvstate {
        &mut self.kvcache
    }

    /// Returns the sampler parameters.
    #[must_use]
    pub const fn getsamplerparams(&self) -> samplingparams {
        self.sampler.getparams()
    }

    /// Resets the request to a fresh attempt over the same prompt.
    ///
    /// Clears the generated context, rewinds the sampler to its seeded start,
    /// and empties the key/value cache, so no residual state survives a
    /// cancellation, a failure, or a completed run (OW3).
    pub fn reset(&mut self) {
        self.context.clear();
        self.context.extend_from_slice(&self.prompt);
        self.sampler.reset();
        self.kvcache.clear();
    }

    /// Selects one token from `logits` using the request's sampler.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`sampler::getsample`].
    pub fn sample(&mut self, logits: &[f32]) -> Result<usize, infererror> {
        self.sampler.getsample(logits)
    }

    /// Appends one generated `token` to the context.
    ///
    /// # Errors
    ///
    /// Returns [`infererror::invalidrequest`] when the context is already at
    /// `maxseq`, so the context can never exceed its configured bound.
    pub fn pushgenerated(&mut self, token: usize) -> Result<(), infererror> {
        if self.context.len() >= self.maxseq {
            return Err(infererror::invalidrequest {
                field: "context",
                detail: format!(
                    "length {} reached maxseq {}",
                    self.context.len(),
                    self.maxseq
                ),
            });
        }
        self.context.push(token);
        Ok(())
    }

    /// Resets the request and validates its tokens against `config`.
    ///
    /// # Errors
    ///
    /// Returns [`infererror::tokenoutofrange`] when a prompt or end-of-sequence
    /// token is outside the model vocabulary, and
    /// [`infererror::invalidrequest`] when the prompt exceeds the model's
    /// maximum sequence length.
    pub(crate) fn prepare(&mut self, config: &llamaconfig) -> Result<(), infererror> {
        self.reset();
        let vocab = config.getvocab();
        for &token in &self.prompt {
            if token >= vocab {
                return Err(infererror::tokenoutofrange { token, vocab });
            }
        }
        for &token in self.stop.geteostokens() {
            if token >= vocab {
                return Err(infererror::tokenoutofrange { token, vocab });
            }
        }
        let model_maxseq = config.getmaxseq();
        if self.maxseq > model_maxseq {
            return Err(infererror::invalidrequest {
                field: "maxseq",
                detail: format!(
                    "request maxseq {} exceeds model maxseq {}",
                    self.maxseq, model_maxseq
                ),
            });
        }
        let requested_len = self
            .prompt
            .len()
            .checked_add(self.getmaxnewtokens())
            .ok_or_else(|| infererror::invalidrequest {
                field: "maxnewtokens",
                detail: "prompt length plus maxnewtokens overflows usize".to_string(),
            })?;
        if requested_len > model_maxseq {
            return Err(infererror::invalidrequest {
                field: "maxnewtokens",
                detail: format!(
                    "prompt length {} plus maxnewtokens {} exceeds model maxseq {}",
                    self.prompt.len(),
                    self.getmaxnewtokens(),
                    model_maxseq
                ),
            });
        }
        Ok(())
    }
}
