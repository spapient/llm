//! Independent scalar reference implementations.
//!
//! These functions are deliberately written with plain `f32` arithmetic and
//! standard-library math, independent of the `cpu` crate, so that they can
//! detect errors in the optimized kernels they cross-check
//! (see `docs/architecture.md`). They are not a production
//! operator implementation and are not used on any runtime path.

/// Scalar ReLU: `max(0, x)`.
#[must_use]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Scalar SiLU: `x * sigmoid(x)`, computed as `x / (1 + exp(-x))`.
#[must_use]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Scalar tanh-approximation GELU.
///
/// Uses `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
#[must_use]
pub fn gelutanh(x: f32) -> f32 {
    /// `sqrt(2 / pi)`.
    const SQRT2OVERPI: f32 = 0.797_884_6;
    /// The cubic coefficient of the tanh GELU approximation.
    const CUBIC: f32 = 0.044_715;
    0.5 * x * (1.0 + (SQRT2OVERPI * (x + CUBIC * x * x * x)).tanh())
}

/// Applies a scalar function to every element of `input`.
#[must_use]
pub fn map(input: &[f32], f: impl Fn(f32) -> f32) -> Vec<f32> {
    input.iter().copied().map(f).collect()
}

/// Numerically stable softmax over one row.
///
/// Subtracts the row maximum before exponentiating. Returns an empty vector for
/// empty input; callers that require a non-empty last dimension validate that
/// precondition themselves.
#[must_use]
pub fn softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Scalar matrix multiplication of `a[m, k]` by `b[k, n]` into `[m, n]`.
///
/// Accumulates in a plain left-to-right `f32` sum, matching the production
/// kernel's deterministic accumulation order for cross-checking.
#[must_use]
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                out[i * n + j] += av * b[p * n + j];
            }
        }
    }
    out
}

/// Scalar RMS normalization of one `row` scaled by `weight`.
///
/// Computes `x_i / sqrt(mean(x_j^2) + eps) * w_i`.
#[must_use]
pub fn rmsnorm(row: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let sumsq: f32 = row.iter().map(|&x| x * x).sum();
    let mean = sumsq / row.len() as f32;
    let inv = 1.0 / (mean + eps).sqrt();
    row.iter()
        .zip(weight.iter())
        .map(|(&x, &w)| x * inv * w)
        .collect()
}

/// Scalar rotary position embedding for one position over `heads * headdim`.
///
/// Applies the pair rotation `(2i, 2i + 1)` with frequency
/// `base^(-2i / headdim)` to each contiguous `headdim` head slice.
#[must_use]
pub fn rope(row: &[f32], position: usize, heads: usize, headdim: usize, base: f32) -> Vec<f32> {
    let half = headdim / 2;
    let mut out = vec![0.0_f32; row.len()];
    for head in 0..heads {
        let offset = head * headdim;
        for i in 0..half {
            let angle = position as f32 * base.powf(-2.0 * i as f32 / headdim as f32);
            let (sin, cos) = angle.sin_cos();
            let even = row[offset + 2 * i];
            let odd = row[offset + 2 * i + 1];
            out[offset + 2 * i] = even * cos - odd * sin;
            out[offset + 2 * i + 1] = even * sin + odd * cos;
        }
    }
    out
}

/// Scalar row gather from a table with the given row `width`.
#[must_use]
pub fn embedding(table: &[f32], width: usize, indices: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(indices.len() * width);
    for &i in indices {
        out.extend_from_slice(&table[i * width..(i + 1) * width]);
    }
    out
}

/// Scalar serial semigroup scan (diagonal `a`): `h_t = a ⊙ h_{t-1} + b ⊙ x_t`,
/// `y_t = c · h_t + skip ⊙ x_t`, per channel, elementwise over the state width.
///
/// `input` is `[n, d]`, `init`/`a`/`b`/`c` are `[d, N]` (row-major: channel
/// then state), and `skip` is `[d]`. Returns the `[n, d]` output and the
/// `[d, N]` final state; for `n == 0` the output is empty and the final state
/// is `init` unchanged. This is the serial reference the parallel scan must
/// match within tolerance.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn scan(
    input: &[f32],
    init: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    skip: &[f32],
    n: usize,
    d: usize,
    statewidth: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut state = init.to_vec();
    let mut out = vec![0.0_f32; n * d];
    for t in 0..n {
        for i in 0..d {
            let x = input[t * d + i];
            let mut acc = 0.0_f32;
            for j in 0..statewidth {
                let idx = i * statewidth + j;
                state[idx] = a[idx] * state[idx] + b[idx] * x;
                acc += c[idx] * state[idx];
            }
            out[t * d + i] = acc + skip[i] * x;
        }
    }
    (out, state)
}
