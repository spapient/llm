use ::tensor::{shape, tensor, tensorview};
use api::backenderror;

use crate::support::finish;

/// The stable name reported by [`cpuscan`].
const SCANOP: &str = "scan";

/// Serial semigroup scan: `h_t = A h_{t-1} + B x_t`, `y_t = C h_t + D x_t`,
/// with a diagonal (or 2×2 block-diagonal) `A`, per channel, elementwise over
/// the state width.
///
/// `input` is `[n, d]`, `init`/`a`/`b`/`c` are `[d, N]`, and `skip` is `[d]`.
/// The spectral precondition `|a| <= 1` is enforced: a diagonal entry outside
/// `[-1, 1]`, or non-finite, is rejected as [`backenderror::invalidparameter`].
/// This is the deterministic serial path the parallel scan must match within
/// tolerance, and the oracle-identical reference for cross-checking.
#[derive(Clone, Copy, Debug, Default)]
pub struct cpuscan;

impl cpuscan {
    /// Applies the full serial scan, returning the `[n, d]` output and the
    /// `[d, N]` final state.
    ///
    /// # Errors
    ///
    /// Returns [`backenderror::shapemismatch`] for inconsistent ranks or
    /// widths, [`backenderror::invalidparameter`] for a spectral violation, or
    /// [`backenderror::sizeoverflow`] when a shape product overflows.
    pub fn apply(
        &self,
        input: &tensorview<'_, f32>,
        init: &tensorview<'_, f32>,
        a: &tensorview<'_, f32>,
        b: &tensorview<'_, f32>,
        c: &tensorview<'_, f32>,
        skip: &tensorview<'_, f32>,
    ) -> Result<(tensor<f32>, tensor<f32>), backenderror> {
        let (n, d, statewidth) = scanplan(input, init, a, b, c, skip)?;
        let mut state = init.getdata().to_vec();
        let mut out = vec![0.0_f32; n * d];
        let xdata = input.getdata();
        let adata = a.getdata();
        let bdata = b.getdata();
        let cdata = c.getdata();
        let skipdata = skip.getdata();
        for t in 0..n {
            for i in 0..d {
                let x = xdata[t * d + i];
                let mut acc = 0.0_f32;
                for j in 0..statewidth {
                    let idx = i * statewidth + j;
                    state[idx] = adata[idx] * state[idx] + bdata[idx] * x;
                    acc += cdata[idx] * state[idx];
                }
                out[t * d + i] = acc + skipdata[i] * x;
            }
        }
        let outshape = shape::new(vec![n, d]).map_err(|_| backenderror::sizeoverflow)?;
        let stateshape = shape::new(vec![d, statewidth]).map_err(|_| backenderror::sizeoverflow)?;
        Ok((finish(&outshape, out)?, finish(&stateshape, state)?))
    }

    /// Applies one incremental step for a single `[1, d]` token.
    ///
    /// # Errors
    ///
    /// As [`Self::apply`], plus [`backenderror::shapemismatch`] when `input` is
    /// not a single row.
    pub fn apply_cached(
        &self,
        input: &tensorview<'_, f32>,
        state: &tensorview<'_, f32>,
        a: &tensorview<'_, f32>,
        b: &tensorview<'_, f32>,
        c: &tensorview<'_, f32>,
        skip: &tensorview<'_, f32>,
    ) -> Result<(tensor<f32>, tensor<f32>), backenderror> {
        let (_, d, statewidth) = scanplan(input, state, a, b, c, skip)?;
        if input.getshape().getdims()[0] != 1 {
            return Err(backenderror::shapemismatch { op: SCANOP });
        }
        let mut newstate = state.getdata().to_vec();
        let mut out = vec![0.0_f32; d];
        let xdata = input.getdata();
        let adata = a.getdata();
        let bdata = b.getdata();
        let cdata = c.getdata();
        let skipdata = skip.getdata();
        for i in 0..d {
            let x = xdata[i];
            let mut acc = 0.0_f32;
            for j in 0..statewidth {
                let idx = i * statewidth + j;
                newstate[idx] = adata[idx] * newstate[idx] + bdata[idx] * x;
                acc += cdata[idx] * newstate[idx];
            }
            out[i] = acc + skipdata[i] * x;
        }
        let outshape = shape::new(vec![1, d]).map_err(|_| backenderror::sizeoverflow)?;
        let stateshape = shape::new(vec![d, statewidth]).map_err(|_| backenderror::sizeoverflow)?;
        Ok((finish(&outshape, out)?, finish(&stateshape, newstate)?))
    }

    /// Applies the scan with a chunked three-phase reorganization: chunk-local
    /// serial scan, inter-chunk prefix, then chunk-local correction. The
    /// reorganization reassociates the floating-point order, so the result
    /// matches [`Self::apply`] within the parallel-reduction tolerance (§5.10),
    /// not bit-for-bit.
    ///
    /// # Errors
    ///
    /// As [`Self::apply`], plus [`backenderror::shapemismatch`] when `chunk`
    /// is zero.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_chunked(
        &self,
        input: &tensorview<'_, f32>,
        init: &tensorview<'_, f32>,
        a: &tensorview<'_, f32>,
        b: &tensorview<'_, f32>,
        c: &tensorview<'_, f32>,
        skip: &tensorview<'_, f32>,
        chunk: usize,
    ) -> Result<(tensor<f32>, tensor<f32>), backenderror> {
        let (n, d, statewidth) = scanplan(input, init, a, b, c, skip)?;
        if chunk == 0 {
            return Err(backenderror::shapemismatch { op: SCANOP });
        }
        if n <= chunk {
            return self.apply(input, init, a, b, c, skip);
        }
        let prefixes = chunkprefixes(
            n,
            d,
            statewidth,
            chunk,
            input.getdata(),
            a.getdata(),
            b.getdata(),
        );
        let (out, state) = chunkapply(
            n,
            d,
            statewidth,
            chunk,
            input.getdata(),
            init.getdata(),
            a.getdata(),
            b.getdata(),
            c.getdata(),
            skip.getdata(),
            &prefixes,
        );
        let outshape = shape::new(vec![n, d]).map_err(|_| backenderror::sizeoverflow)?;
        let stateshape = shape::new(vec![d, statewidth]).map_err(|_| backenderror::sizeoverflow)?;
        Ok((finish(&outshape, out)?, finish(&stateshape, state)?))
    }
}

/// Computes the inter-chunk prefix of each chunk's semigroup combine, in chunk
/// order. `prefixes[k]` is the combine of every chunk before `k`, so applying
/// `prefixes[k]` to a chunk-local scan recovers the global prefix up to that
/// chunk.
fn chunkprefixes(
    n: usize,
    d: usize,
    statewidth: usize,
    chunk: usize,
    x: &[f32],
    a: &[f32],
    b: &[f32],
) -> Vec<(Vec<f32>, Vec<f32>)> {
    let width = d * statewidth;
    let mut prefixes = Vec::new();
    let mut prefix_a = vec![1.0_f32; width];
    let mut prefix_b = vec![0.0_f32; width];
    let mut start = 0;
    while start < n {
        let end = (start + chunk).min(n);
        let mut block_a = vec![1.0_f32; width];
        let mut block_b = vec![0.0_f32; width];
        for t in start..end {
            for i in 0..d {
                let xv = x[t * d + i];
                for j in 0..statewidth {
                    let idx = i * statewidth + j;
                    block_b[idx] = a[idx] * block_b[idx] + b[idx] * xv;
                    block_a[idx] *= a[idx];
                }
            }
        }
        prefixes.push((prefix_a.clone(), prefix_b.clone()));
        for idx in 0..width {
            prefix_b[idx] = block_a[idx] * prefix_b[idx] + block_b[idx];
            prefix_a[idx] *= block_a[idx];
        }
        start = end;
    }
    prefixes
}

/// Applies each chunk's local serial scan (from `init`) and corrects it with
/// the chunk's prefix, returning the output and the final state.
#[allow(clippy::too_many_arguments)]
fn chunkapply(
    n: usize,
    d: usize,
    statewidth: usize,
    chunk: usize,
    x: &[f32],
    init: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    skip: &[f32],
    prefixes: &[(Vec<f32>, Vec<f32>)],
) -> (Vec<f32>, Vec<f32>) {
    let width = d * statewidth;
    let mut out = vec![0.0_f32; n * d];
    let mut state = init.to_vec();
    let mut start = 0;
    let mut block = 0;
    while start < n {
        let end = (start + chunk).min(n);
        let (pa, pb) = &prefixes[block];
        let mut h = Vec::with_capacity(width);
        for idx in 0..width {
            h.push(pa[idx] * init[idx] + pb[idx]);
        }
        for t in start..end {
            for i in 0..d {
                let xv = x[t * d + i];
                let mut acc = 0.0_f32;
                for j in 0..statewidth {
                    let idx = i * statewidth + j;
                    h[idx] = a[idx] * h[idx] + b[idx] * xv;
                    acc += c[idx] * h[idx];
                }
                out[t * d + i] = acc + skip[i] * xv;
            }
        }
        state = h;
        block += 1;
        start = end;
    }
    (out, state)
}

/// Validates ranks, channel alignment, state-width alignment, and the spectral
/// precondition, returning `(n, d, statewidth)`.
fn scanplan(
    input: &tensorview<'_, f32>,
    init: &tensorview<'_, f32>,
    a: &tensorview<'_, f32>,
    b: &tensorview<'_, f32>,
    c: &tensorview<'_, f32>,
    skip: &tensorview<'_, f32>,
) -> Result<(usize, usize, usize), backenderror> {
    let id = input.getshape().getdims();
    let initd = init.getshape().getdims();
    let ad = a.getshape().getdims();
    let bd = b.getshape().getdims();
    let cd = c.getshape().getdims();
    let skipd = skip.getshape().getdims();
    if id.len() != 2
        || initd.len() != 2
        || ad.len() != 2
        || bd.len() != 2
        || cd.len() != 2
        || skipd.len() != 1
    {
        return Err(backenderror::shapemismatch { op: SCANOP });
    }
    let d = id[1];
    if ad[0] != d || bd[0] != d || cd[0] != d || skipd[0] != d || initd[0] != d {
        return Err(backenderror::shapemismatch { op: SCANOP });
    }
    let statewidth = initd[1];
    if ad[1] != statewidth || bd[1] != statewidth || cd[1] != statewidth {
        return Err(backenderror::shapemismatch { op: SCANOP });
    }
    checkspectrum(a.getdata(), SCANOP)?;
    Ok((id[0], d, statewidth))
}

/// Rejects any diagonal entry whose magnitude exceeds one or is non-finite.
fn checkspectrum(a: &[f32], op: &'static str) -> Result<(), backenderror> {
    for &value in a {
        if value.abs() > 1.0 || value.is_nan() {
            return Err(backenderror::invalidparameter { op });
        }
    }
    Ok(())
}
