use lib::seed;

use crate::error::infererror;

/// Parameters that shape how a token is selected from next-token logits.
///
/// A temperature of exactly `0.0` selects greedy decoding. Otherwise the logits
/// are scaled by `1 / temperature`, optionally restricted to the `topk` largest
/// and the smallest prefix whose cumulative probability reaches `topp`, and one
/// token is drawn from the resulting categorical distribution using a
/// deterministic stream seeded by `seed`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct samplingparams {
    temperature: f32,
    topk: Option<usize>,
    topp: Option<f32>,
    seed: seed,
}

impl samplingparams {
    /// Builds greedy parameters: temperature `0.0`, no top-k or top-p filter.
    #[must_use]
    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            topk: None,
            topp: None,
            seed: seed::new(0),
        }
    }

    /// Builds validated sampling parameters.
    ///
    /// # Errors
    ///
    /// Returns [`infererror::invalidparams`] when `temperature` is not finite
    /// and non-negative, `topk` is `Some(0)`, or `topp` is outside `(0, 1]`.
    pub fn new(
        temperature: f32,
        topk: Option<usize>,
        topp: Option<f32>,
        s: seed,
    ) -> Result<Self, infererror> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(infererror::invalidparams {
                detail: "temperature must be finite and non-negative",
            });
        }
        if matches!(topk, Some(0)) {
            return Err(infererror::invalidparams {
                detail: "topk must be non-zero",
            });
        }
        if let Some(p) = topp
            && (!p.is_finite() || p <= 0.0 || p > 1.0)
        {
            return Err(infererror::invalidparams {
                detail: "topp must be in (0, 1]",
            });
        }
        Ok(Self {
            temperature,
            topk,
            topp,
            seed: s,
        })
    }

    /// Returns the temperature.
    #[must_use]
    pub const fn gettemperature(&self) -> f32 {
        self.temperature
    }

    /// Returns the top-k bound, if any.
    #[must_use]
    pub const fn gettopk(&self) -> Option<usize> {
        self.topk
    }

    /// Returns the top-p bound, if any.
    #[must_use]
    pub const fn gettopp(&self) -> Option<f32> {
        self.topp
    }

    /// Returns the reproducible seed.
    #[must_use]
    pub const fn getseed(&self) -> seed {
        self.seed
    }
}

/// A deterministic next-token selector.
///
/// A sampler owns only its parameters and its generator position; it holds no
/// request data and no global state. Re-seeding a fresh sampler with the same
/// [`samplingparams`] reproduces the same sequence of selections for the same
/// sequence of logits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct sampler {
    params: samplingparams,
    state: u64,
}

impl sampler {
    /// Creates a sampler positioned at the start of its seeded stream.
    #[must_use]
    pub const fn new(params: samplingparams) -> Self {
        Self {
            params,
            state: params.getseed().get(),
        }
    }

    /// Returns the sampler parameters.
    #[must_use]
    pub const fn getparams(&self) -> samplingparams {
        self.params
    }

    /// Rewinds the generator to the start of its seeded stream.
    pub fn reset(&mut self) {
        self.state = self.params.getseed().get();
    }

    /// Selects a token from `logits`.
    ///
    /// # Errors
    ///
    /// Returns [`infererror::emptylogits`] when `logits` is empty, and
    /// [`infererror::degeneratelogits`] when a logit is not finite or the
    /// resulting distribution has no finite positive mass. Parameters are
    /// validated once at construction, so they are not re-checked here.
    pub fn getsample(&mut self, logits: &[f32]) -> Result<usize, infererror> {
        if logits.is_empty() {
            return Err(infererror::emptylogits);
        }
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(infererror::degeneratelogits);
        }
        if self.params.gettemperature() == 0.0 {
            return Ok(greedy(logits));
        }
        self.samplecategorical(logits)
    }

    fn samplecategorical(&mut self, logits: &[f32]) -> Result<usize, infererror> {
        let temperature = f64::from(self.params.gettemperature());
        let scaled: Vec<f64> = logits
            .iter()
            .map(|value| f64::from(*value) / temperature)
            .collect();
        let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !max.is_finite() {
            return Err(infererror::degeneratelogits);
        }
        let mut probs: Vec<f64> = scaled.iter().map(|value| (value - max).exp()).collect();
        let total: f64 = probs.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(infererror::degeneratelogits);
        }
        let mut order: Vec<usize> = (0..probs.len()).collect();
        // Descending probability; the sort is stable, so equal probabilities
        // keep ascending index order and the selection stays deterministic.
        order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        let keep = keptcount(
            &order,
            &probs,
            total,
            self.params.gettopk(),
            self.params.gettopp(),
        );
        for &index in order.iter().skip(keep) {
            probs[index] = 0.0;
        }
        let sum: f64 = probs.iter().sum();
        let draw = self.nextunit() * sum;
        let mut cumulative = 0.0;
        for (index, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if draw < cumulative {
                return Ok(index);
            }
        }
        Ok(order.get(keep - 1).copied().unwrap_or(0))
    }

    fn nextunit(&mut self) -> f64 {
        let raw = self.nextu64();
        (raw >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn nextu64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Returns the index of the largest finite logit, ties broken by lowest index.
fn greedy(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bestvalue = f32::NEG_INFINITY;
    for (index, &value) in logits.iter().enumerate() {
        if value > bestvalue {
            bestvalue = value;
            best = index;
        }
    }
    best
}

/// Returns how many of the leading `order` entries survive the top-k and top-p
/// filters.
fn keptcount(
    order: &[usize],
    probs: &[f64],
    total: f64,
    topk: Option<usize>,
    topp: Option<f32>,
) -> usize {
    let mut keep = order.len();
    if let Some(k) = topk {
        keep = keep.min(k);
    }
    if let Some(p) = topp {
        let threshold = f64::from(p);
        let mut cumulative = 0.0;
        let mut count = 0usize;
        for &index in order.iter().take(keep) {
            cumulative += probs[index];
            count += 1;
            if cumulative / total >= threshold {
                break;
            }
        }
        keep = count.max(1);
    }
    keep.max(1)
}
