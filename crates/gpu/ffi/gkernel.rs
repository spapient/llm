#![cfg_attr(
    not(any(rocm_toolkit, sycl_toolkit, cann_toolkit, mlu_toolkit)),
    allow(dead_code)
)]
//! Metal's bindings are Objective-C and therefore stay in `metal/ffi.rs`.

// ---------------------------------------------------------------------------
// Shared GRID-family driver core (no `unsafe`; every platform call stays
// inside each island's single driver-handoff block).
// ---------------------------------------------------------------------------

/// The operand, scalar, and launch-geometry types shared by the four
/// toolchain-gated backends (ROCm, SYCL, CANN, MLU).
///
/// On hosts without any of the GRID-family toolkits every island compiles
/// out, so the shared core is intentionally unreferenced; the cfg_attr keeps
/// those hosts warning-free while the toolchain builds still check it
/// normally.
/// A scalar kernel argument. Every kernel takes its buffer operands first
/// and its scalar parameters last, so the launch argument list is exactly
/// the bound device pointers followed by these scalars.
#[derive(Clone, Copy, Debug)]
pub(crate) enum scalar {
    /// An `unsigned` kernel parameter.
    u32(u32),
    /// An `unsigned long long` kernel parameter.
    u64(u64),
    /// A `float` kernel parameter.
    f32(f32),
}

/// A bound operand of a dispatch.
///
/// Every dispatch is run-to-completion: host operands are uploaded, the
/// output is downloaded, and this dispatch's temporary allocations are
/// released inside the single handoff block.
#[derive(Clone, Copy, Debug)]
pub(crate) enum buf<'a> {
    /// Upload a host `f32` slice into a fresh device allocation.
    host(&'a [f32]),
    /// Upload a host `u64` slice into a fresh device allocation.
    hostu64(&'a [u64]),
    /// Upload a host `f32` slice, keep the allocation live through the
    /// launch, and download it afterwards (in-place kernels such as
    /// `scan`'s state).
    inout(&'a [f32]),
    /// Allocate the output buffer and download it after the kernel
    /// completes.
    out(usize),
}

/// Launch geometry for one kernel.
#[derive(Clone, Copy, Debug)]
pub(crate) struct launchspec {
    pub(crate) grid: (u32, u32, u32),
    pub(crate) block: (u32, u32, u32),
    pub(crate) shared: u32,
}

/// Flat 1D grid (elementwise / rope / rope_at / embedding / scan):
/// `blockDim = (256, 1, 1)`, `gridDim.x` covers the element count.
pub(crate) fn flat(count: usize) -> launchspec {
    launchspec {
        grid: (divceil(count, 256), 1, 1),
        block: (256, 1, 1),
        shared: 0,
    }
}

/// Row-wise grid (rmsnorm / softmax): one block per row, a power-of-two
/// `lanes` per block, and `lanes` floats of on-chip memory for the block
/// reduction. Grid-family kernels that partition rows inside the kernel
/// (CANN / MLU) ignore the grid and reuse the on-chip budget.
pub(crate) fn rows(rows: usize, lanes: u32) -> launchspec {
    launchspec {
        grid: (rows as u32, 1, 1),
        block: (lanes, 1, 1),
        shared: lanes * 4,
    }
}

/// Attention grid: one block per (query position, head) tile, with
/// `(2 * lanes + headdim)` floats of on-chip memory (reduce + weights +
/// accumulator).
pub(crate) fn attention(tiles: usize, headdim: usize, lanes: u32) -> launchspec {
    launchspec {
        grid: (tiles as u32, 1, 1),
        block: (lanes, 1, 1),
        shared: ((2 * lanes as usize + headdim) * 4) as u32,
    }
}

/// Matmul grid: `blockIdx.x` = column tile, `blockIdx.y` = row tile,
/// `blockIdx.z` = batch slice, `16 x 16` threads.
pub(crate) fn matmul(m: usize, n: usize, batch: usize) -> launchspec {
    launchspec {
        grid: (divceil(n, 16), divceil(m, 16), batch as u32),
        block: (16, 16, 1),
        shared: 0,
    }
}

fn divceil(a: usize, b: usize) -> u32 {
    a.div_ceil(b) as u32
}

/// Host scalar softmax over every contiguous `width`-element row.
///
/// The GRID kernels seed each row maximum with a finite sentinel
/// (`GRID_NEGLARGE`) so a fully finite row never underflows; that sentinel
/// turns an all-(-inf) row into zeros instead of the NaN the scalar
/// reference propagates, and a NaN/+-inf value otherwise needs ordinary
/// IEEE propagation. An operator that still holds the host operand runs
/// any input containing a non-finite value here rather than launching.
///
/// This is bit-compatible with the CPU scalar softmax row: the maximum
/// folds from negative infinity via `f32::max`, the shifted exponent uses
/// libm `exp`, and a row whose shifted-exponent sum is not strictly
/// positive (the all-(-inf) row) keeps the NaN it already wrote rather
/// than dividing by zero.
pub(crate) fn scalarsoftmax(input: &[f32], width: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; input.len()];
    if width == 0 {
        return out;
    }
    for (row, dst) in input.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for (d, &x) in dst.iter_mut().zip(row.iter()) {
            let e = (x - max).exp();
            *d = e;
            sum += e;
        }
        if sum > 0.0_f32 {
            for d in dst.iter_mut() {
                *d /= sum;
            }
        }
    }
    out
}
