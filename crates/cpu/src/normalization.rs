use ::tensor::{tensor, tensorview};
use api::{backenderror, weightedkernel};

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
/// The tier is detected once before any row runs, so the runtime detection
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
    let isa = simd::detect();
    runrows(rows, src, data, width, workers, |row, out| {
        simd::rmsnormrow(isa, row, out, weight, eps)
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

/// Computes one reciprocal RMS per row of `input`, with the same per-row
/// square-sum reassociation as the standalone kernel tiers, so the scale the
/// fused `rmsnormmatmul` applies equals the standalone `rmsnorm` divisor
/// exactly. The weight is not consulted: the scale is `1 / sqrt(mean(x^2)+eps)`.
pub(crate) fn rowinvtable(input: &[f32], width: usize, eps: f32) -> Vec<f32> {
    let rows = input.len() / width;
    let mut inv = vec![0.0_f32; rows];
    let isa = simd::detect();
    for r in 0..rows {
        inv[r] = simd::rowinv(isa, &input[r * width..(r + 1) * width], eps);
    }
    inv
}
