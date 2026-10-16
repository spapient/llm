use ::tensor::{tensor, tensorview};
use api::{backenderror, weightedkernel};
use archmage::prelude::*;

use crate::support::finish;

/// The stable name reported by [`cpurmsnorm`].
const RMSNORMOP: &str = "rmsnorm";

/// RMS normalization over the last axis, scaled by a learnable weight.
///
/// For each row `x` of the last axis and weight `w` of the same length,
///
/// ```text
/// rms     = sqrt(mean(x_i^2) + eps)
/// out_i   = (x_i / rms) * w_i
/// ```
///
/// The epsilon is carried by the kernel instance so that no per-model value is
/// hidden in a global. `eps` MUST be finite and non-negative; a negative
/// epsilon can drive the denominator to zero and is not validated here because
/// it is a caller configuration precondition, not external input.
#[derive(Clone, Copy, Debug)]
pub struct cpurmsnorm {
    eps: f32,
}

impl cpurmsnorm {
    /// The default epsilon used by [`Default`].
    pub const DEFAULTEPS: f32 = 1.0e-5;

    /// Creates an RMS normalization kernel with the given epsilon.
    #[must_use]
    pub const fn new(eps: f32) -> Self {
        Self { eps }
    }

    /// Returns the configured epsilon.
    #[must_use]
    pub const fn geteps(self) -> f32 {
        self.eps
    }
}

impl Default for cpurmsnorm {
    fn default() -> Self {
        Self::new(Self::DEFAULTEPS)
    }
}

impl weightedkernel<f32> for cpurmsnorm {
    fn getname(&self) -> &'static str {
        RMSNORMOP
    }

    fn apply(
        &self,
        input: &tensorview<'_, f32>,
        weight: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        self.applywith(input, weight, 1)
    }
}

impl cpurmsnorm {
    /// Applies the kernel, splitting the rows over up to `workers` scoped
    /// threads. Every row's arithmetic is identical to the serial kernel, so
    /// the result is bit-for-bit independent of `workers`.
    pub(crate) fn applywith(
        &self,
        input: &tensorview<'_, f32>,
        weight: &tensorview<'_, f32>,
        workers: usize,
    ) -> Result<tensor<f32>, backenderror> {
        let Some(&width) = input.getshape().getdims().last() else {
            return Err(backenderror::unsupportedoperation { op: RMSNORMOP });
        };
        if width == 0 {
            return Err(backenderror::unsupportedoperation { op: RMSNORMOP });
        }
        if weight.getdata().len() != width {
            return Err(backenderror::shapemismatch { op: RMSNORMOP });
        }
        let src = input.getdata();
        let w = weight.getdata();
        let mut data = vec![0.0_f32; src.len()];
        rmsnormrowsparallel(src, &mut data, w, self.eps, width, workers)?;
        finish(input.getshape(), data)
    }
}

/// Normalizes every row of `src` into `data`, dispatching the SIMD tier once
/// per call and splitting the rows over scoped threads when asked.
///
/// The tier proof is summoned before any row runs, so the runtime detection
/// cost is paid once per call, not once per row.
fn rmsnormrowsparallel(
    src: &[f32],
    data: &mut [f32],
    weight: &[f32],
    eps: f32,
    width: usize,
    workers: usize,
) -> Result<(), backenderror> {
    let rows = src.len() / width;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(token) = X64V3Token::summon() {
        return runrows(rows, src, data, width, workers, |row, out| {
            rmsnormrow_v3(token, row, out, weight, eps)
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(token) = NeonToken::summon() {
        return runrows(rows, src, data, width, workers, |row, out| {
            rmsnormrow_neon(token, row, out, weight, eps)
        });
    }
    runrows(rows, src, data, width, workers, |row, out| {
        rmsnormrow_scalar(ScalarToken, row, out, weight, eps)
    })
}

/// Runs one row kernel over every row of `src` into `data`, in parallel when
/// `workers > 1` and there is more than one row.
///
/// Each thread claims row indices from an atomic counter and writes the row
/// directly through a mutex-guarded output slice, so no merge copy is added
/// to a bandwidth-bound operation. The per-row arithmetic is identical to the
/// serial order.
fn runrows<F>(
    rows: usize,
    src: &[f32],
    data: &mut [f32],
    width: usize,
    workers: usize,
    rowkernel: F,
) -> Result<(), backenderror>
where
    F: Fn(&[f32], &mut [f32]) + Send + Sync,
{
    if workers <= 1 || rows <= 1 {
        for i in 0..rows {
            rowkernel(
                &src[i * width..(i + 1) * width],
                &mut data[i * width..(i + 1) * width],
            );
        }
        return Ok(());
    }
    // The row pool and the claim counter live outside the scope so the
    // closures can borrow them for the whole scoped lifetime.
    let pool: Vec<std::sync::Mutex<&mut [f32]>> =
        data.chunks_mut(width).map(std::sync::Mutex::new).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let pool = &pool;
    let next = &next;
    let rowkernel = &rowkernel;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            handles.push(scope.spawn(move || {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= rows {
                        break;
                    }
                    let Ok(mut outrow) = pool[i].lock() else {
                        return false;
                    };
                    rowkernel(&src[i * width..(i + 1) * width], &mut outrow);
                }
                true
            }));
        }
        for handle in handles {
            match handle.join() {
                Ok(true) => {}
                Ok(false) | Err(_) => {
                    return Err(backenderror::executionfailed {
                        backend: "cpu",
                        kernel: RMSNORMOP.into(),
                        message: "an rmsnorm worker thread panicked".into(),
                    });
                }
            }
        }
        Ok(())
    })
}

/// RMS-normalizes one `row` on the AVX2+FMA tier, eight lanes per vector.
///
/// The safe intrinsic imports take array references, so the row is walked
/// through [`slice::as_chunks`] and the scalar tail keeps the same order.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
fn rmsnormrow_v3(_token: X64V3Token, row: &[f32], out: &mut [f32], weight: &[f32], eps: f32) {
    let n = row.len();
    let (chunks, tail) = row.as_chunks::<8>();
    let (wchunks, _) = weight.as_chunks::<8>();
    let mut acc = _mm256_setzero_ps();
    for x in chunks {
        let xv = _mm256_loadu_ps(x);
        acc = _mm256_fmadd_ps(xv, xv, acc);
    }
    let mut lanes = [0.0_f32; 8];
    _mm256_storeu_ps(&mut lanes, acc);
    let mut sumsq = lanes.iter().sum::<f32>();
    for &x in tail {
        sumsq += x * x;
    }
    let inv = 1.0 / (sumsq / n as f32 + eps).sqrt();
    let invv = _mm256_set1_ps(inv);
    let (ochunks, _) = out.as_chunks_mut::<8>();
    for (o, (&x, &w)) in ochunks.iter_mut().zip(chunks.iter().zip(wchunks.iter())) {
        let xv = _mm256_loadu_ps(&x);
        let wv = _mm256_loadu_ps(&w);
        _mm256_storeu_ps(o, _mm256_mul_ps(_mm256_mul_ps(xv, invv), wv));
    }
    for i in n - tail.len()..n {
        out[i] = row[i] * inv * weight[i];
    }
}

/// RMS-normalizes one `row` on the NEON tier, four lanes per vector.
#[cfg(target_arch = "aarch64")]
#[arcane(import_intrinsics)]
fn rmsnormrow_neon(_token: NeonToken, row: &[f32], out: &mut [f32], weight: &[f32], eps: f32) {
    let n = row.len();
    let (chunks, tail) = row.as_chunks::<4>();
    let (wchunks, _) = weight.as_chunks::<4>();
    let mut acc = vdupq_n_f32(0.0);
    for x in chunks {
        let xv = vld1q_f32(x);
        acc = vfmaq_f32(acc, xv, xv);
    }
    let mut sumsq = vaddvq_f32(acc);
    for &x in tail {
        sumsq += x * x;
    }
    let inv = 1.0 / (sumsq / n as f32 + eps).sqrt();
    let invv = vdupq_n_f32(inv);
    let (ochunks, _) = out.as_chunks_mut::<4>();
    for (o, (&x, &w)) in ochunks.iter_mut().zip(chunks.iter().zip(wchunks.iter())) {
        let xv = vld1q_f32(&x);
        let wv = vld1q_f32(&w);
        vst1q_f32(o, vmulq_f32(vmulq_f32(xv, invv), wv));
    }
    for i in n - tail.len()..n {
        out[i] = row[i] * inv * weight[i];
    }
}

/// Computes one reciprocal RMS per row of `input`, with the same per-row
/// square-sum reassociation as the standalone kernel tiers, so the scale the
/// fused `rmsnormmatmul` applies equals the standalone `rmsnorm` divisor
/// exactly. The weight is not consulted: the scale is `1 / sqrt(mean(x^2)+eps)`.
pub(crate) fn rowinvtable(input: &[f32], width: usize, eps: f32) -> Vec<f32> {
    let rows = input.len() / width;
    let mut inv = vec![0.0_f32; rows];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(token) = X64V3Token::summon() {
        for r in 0..rows {
            inv[r] = rowinv_v3(token, &input[r * width..(r + 1) * width], eps);
        }
        return inv;
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(token) = NeonToken::summon() {
        for r in 0..rows {
            inv[r] = rowinv_neon(token, &input[r * width..(r + 1) * width], eps);
        }
        return inv;
    }
    for r in 0..rows {
        inv[r] = rowinv_scalar(&input[r * width..(r + 1) * width], eps);
    }
    inv
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
fn rowinv_v3(_token: X64V3Token, row: &[f32], eps: f32) -> f32 {
    let n = row.len();
    let (chunks, tail) = row.as_chunks::<8>();
    let mut acc = _mm256_setzero_ps();
    for x in chunks {
        let xv = _mm256_loadu_ps(x);
        acc = _mm256_fmadd_ps(xv, xv, acc);
    }
    let mut lanes = [0.0_f32; 8];
    _mm256_storeu_ps(&mut lanes, acc);
    let mut sumsq = lanes.iter().sum::<f32>();
    for &x in tail {
        sumsq += x * x;
    }
    1.0 / (sumsq / n as f32 + eps).sqrt()
}

#[cfg(target_arch = "aarch64")]
#[arcane(import_intrinsics)]
fn rowinv_neon(_token: NeonToken, row: &[f32], eps: f32) -> f32 {
    let n = row.len();
    let (chunks, tail) = row.as_chunks::<4>();
    let mut acc = vdupq_n_f32(0.0);
    for x in chunks {
        let xv = vld1q_f32(x);
        acc = vfmaq_f32(acc, xv, xv);
    }
    let mut sumsq = vaddvq_f32(acc);
    for &x in tail {
        sumsq += x * x;
    }
    1.0 / (sumsq / n as f32 + eps).sqrt()
}

fn rowinv_scalar(row: &[f32], eps: f32) -> f32 {
    let n = row.len();
    let mut sumsq = 0.0_f32;
    for &x in row {
        sumsq += x * x;
    }
    1.0 / (sumsq / n as f32 + eps).sqrt()
}

/// RMS-normalizes one `row` on the scalar tier.
fn rmsnormrow_scalar(_token: ScalarToken, row: &[f32], out: &mut [f32], weight: &[f32], eps: f32) {
    let n = row.len();
    let mut sumsq = 0.0_f32;
    for &x in row {
        sumsq += x * x;
    }
    let inv = 1.0 / (sumsq / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = row[i] * inv * weight[i];
    }
}
