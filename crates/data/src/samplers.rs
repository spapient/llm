use crate::dataerror;
use lib::limits;
use lib::seed;

/// A deterministic SplitMix64 generator.
///
/// The state advances by adding the SplitMix64 gamma and then applying the
/// SplitMix64 finalizer. The algorithm is fixed so that a given [`seed`]
/// produces the same stream on every run and platform.
#[derive(Clone, Debug)]
struct rng {
    state: u64,
}

impl rng {
    const fn new(s: seed) -> Self {
        Self { state: s.get() }
    }

    fn nextu64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns a value in `[0, upper)` using rejection sampling to avoid the
    /// modulo bias, then reduces modulo `upper`.
    fn bound(&mut self, upper: usize) -> usize {
        // Precondition: `upper > 0`. `upper` is a `usize`, so `upper as u64` is
        // at most `u64::MAX`; `u64::MAX - b + 1` is the count of values that
        // would introduce bias and never underflows for `b >= 1`.
        let b = upper as u64;
        let threshold = (u64::MAX - b + 1) % b;
        loop {
            let r = self.nextu64();
            if r >= threshold {
                return (r % b) as usize;
            }
        }
    }
}

/// A reproducible index sampler.
///
/// A sampler owns its generator state and is seeded once. Two samplers created
/// from the same [`seed`] produce identical index sequences, and each mutating
/// call advances the stream. It draws only indices; it never reads a dataset,
/// so it stays independent of storage.
///
/// # Reproducibility
///
/// The generator is SplitMix64 and the shuffle is an in-place
/// Fisher-Yates pass from the last index to the second. Both are fixed
/// algorithms, so the derived order is fully specified by the seed.
#[derive(Clone, Debug)]
pub struct sampler {
    rng: rng,
}

impl sampler {
    /// Creates a sampler from a reproducible seed.
    #[must_use]
    pub const fn new(s: seed) -> Self {
        Self { rng: rng::new(s) }
    }

    /// Returns a deterministic permutation of `0..len`.
    ///
    /// The result is a permutation: it contains every index in `0..len` exactly
    /// once. For `len < 2` the identity order is returned unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`dataerror::elementcountlimitexceeded`] when `len` exceeds
    /// [`limits::getmaxelements`].
    pub fn shuffle(&mut self, len: usize, lim: &limits) -> Result<Vec<usize>, dataerror> {
        lim.checkelements(len)
            .map_err(|_| dataerror::elementcountlimitexceeded {
                requested: len,
                limit: lim.getmaxelements(),
            })?;
        let mut indices: Vec<usize> = (0..len).collect();
        for i in (1..len).rev() {
            let j = self.rng.bound(i + 1);
            indices.swap(i, j);
        }
        Ok(indices)
    }

    /// Returns `count` indices drawn uniformly from `0..len`, with replacement.
    ///
    /// # Errors
    ///
    /// Returns [`dataerror::elementcountlimitexceeded`] when `count` exceeds
    /// [`limits::getmaxelements`], and [`dataerror::emptydataset`] when `count`
    /// is positive while `len` is zero.
    pub fn sample(
        &mut self,
        len: usize,
        count: usize,
        lim: &limits,
    ) -> Result<Vec<usize>, dataerror> {
        lim.checkelements(count)
            .map_err(|_| dataerror::elementcountlimitexceeded {
                requested: count,
                limit: lim.getmaxelements(),
            })?;
        if count == 0 {
            return Ok(Vec::new());
        }
        if len == 0 {
            return Err(dataerror::emptydataset);
        }
        let mut indices: Vec<usize> = Vec::with_capacity(count);
        for _ in 0..count {
            indices.push(self.rng.bound(len));
        }
        Ok(indices)
    }
}
