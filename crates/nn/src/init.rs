use ::tensor::{shape, tensor};
use lib::seed;

use crate::error::nnerror;

/// The splitmix64 increment constant.
const SPLITMIXGAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// The first splitmix64 mixing constant.
const SPLITMIXMIX1: u64 = 0xBF58_476D_1CE4_E5B9;

/// The second splitmix64 mixing constant.
const SPLITMIXMIX2: u64 = 0x94D0_49BB_1331_11EB;

/// A deterministic counter-based random source.
///
/// The generator is a SplitMix64 stream seeded from a [`seed`]. It has no
/// global state, so two generators built from the same seed produce identical
/// streams and two builders run in any order produce the same parameters. That
/// is the reproducibility contract initialization relies on.
#[derive(Clone, Debug)]
struct generator {
    state: u64,
}

impl generator {
    /// Creates a generator from a reproducible seed.
    fn fromseed(s: seed) -> Self {
        Self { state: s.get() }
    }

    /// Returns the next 64-bit output.
    fn nextu64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIXGAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(SPLITMIXMIX1);
        z = (z ^ (z >> 27)).wrapping_mul(SPLITMIXMIX2);
        z ^ (z >> 31)
    }

    /// Returns the next value in `[0, 1)` using the top 24 bits.
    fn nextf32(&mut self) -> f32 {
        let bits = (self.nextu64() >> 40) as u32;
        bits as f32 / (1_u32 << 24) as f32
    }

    /// Returns the next standard normal sample (Box-Muller).
    fn nextnormal(&mut self) -> f32 {
        let u1 = self.nextf32().max(f32::MIN_POSITIVE);
        let u2 = self.nextf32();
        (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
    }
}

/// Builds a tensor of `shape` filled with zeros.
///
/// # Errors
///
/// Returns a [`nnerror::tensor`] error when the shape is empty or its element
/// count overflows.
pub fn fillzero(s: shape) -> Result<tensor<f32>, nnerror> {
    fillvalue(s, 0.0)
}

/// Builds a tensor of `shape` filled with ones.
///
/// # Errors
///
/// Returns a [`nnerror::tensor`] error when the shape is empty or its element
/// count overflows.
pub fn fillone(s: shape) -> Result<tensor<f32>, nnerror> {
    fillvalue(s, 1.0)
}

/// Builds a tensor of `shape` filled with `value`.
///
/// # Errors
///
/// Returns a [`nnerror::tensor`] error when the shape is empty or its element
/// count overflows.
pub fn fillvalue(s: shape, value: f32) -> Result<tensor<f32>, nnerror> {
    let n = s.getelementcount()?;
    Ok(tensor::new(s, vec![value; n])?)
}

/// Builds a tensor of `shape` with samples uniform on `[lo, hi)`.
///
/// # Errors
///
/// Returns [`nnerror::invalidconfig`] when `lo` is greater than `hi`, or a
/// [`nnerror::tensor`] error when the shape is invalid.
pub fn filluniform(s: shape, lo: f32, hi: f32, s_seed: seed) -> Result<tensor<f32>, nnerror> {
    if lo > hi {
        return Err(nnerror::config("filluniform", "lo must not exceed hi"));
    }
    let n = s.getelementcount()?;
    let mut rng = generator::fromseed(s_seed);
    let span = hi - lo;
    let data: Vec<f32> = (0..n).map(|_| lo + rng.nextf32() * span).collect();
    Ok(tensor::new(s, data)?)
}

/// Builds a tensor of `shape` with normal samples `N(mean, std^2)`.
///
/// # Errors
///
/// Returns [`nnerror::invalidconfig`] when `std` is negative, or a
/// [`nnerror::tensor`] error when the shape is invalid.
pub fn fillnormal(s: shape, mean: f32, std: f32, s_seed: seed) -> Result<tensor<f32>, nnerror> {
    if std < 0.0 {
        return Err(nnerror::config("fillnormal", "std must not be negative"));
    }
    let n = s.getelementcount()?;
    let mut rng = generator::fromseed(s_seed);
    let data: Vec<f32> = (0..n).map(|_| mean + std * rng.nextnormal()).collect();
    Ok(tensor::new(s, data)?)
}

/// Builds an Xavier-uniform weight of `shape`.
///
/// The limit is `sqrt(6 / (fan_in + fan_out))` with the last two dimensions
/// read as `[.., fan_in, fan_out]`, matching the layer weight layout in this
/// crate.
///
/// # Errors
///
/// Returns [`nnerror::invalidconfig`] when the rank is below two or the fan
/// sum is zero, or a [`nnerror::tensor`] error when the shape is invalid.
pub fn initxavier(s: shape, s_seed: seed) -> Result<tensor<f32>, nnerror> {
    let dims = s.getdims().to_vec();
    if dims.len() < 2 {
        return Err(nnerror::config("initxavier", "rank must be at least two"));
    }
    let fan_in = dims[dims.len() - 2];
    let fan_out = dims[dims.len() - 1];
    let fan_sum = fan_in + fan_out;
    if fan_sum == 0 {
        return Err(nnerror::config(
            "initxavier",
            "fan_in + fan_out must be non-zero",
        ));
    }
    let limit = (6.0 / fan_sum as f32).sqrt();
    filluniform(s, -limit, limit, s_seed)
}
