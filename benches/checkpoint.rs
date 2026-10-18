//! `.u` snapshot state-transition benchmarks: encode, decode, delta creation,
//! and merge.
//!
//! The workload is a small deterministic model of 16 square `512 × 512` `f32`
//! weights (16 MiB raw). Fifteen weights are zeroed (LZ4-compressible) and the
//! last is pseudo-random (stored raw), so each timed path exercises both the
//! compression and raw-store code paths. The base snapshot is all zeroed; the
//! changed snapshot differs only in the last weight, so `delta/create` dedups
//! 15 of 16 tensors and `merge/full` resolves 15 blocks from the base and one
//! from the delta.
//!
//! The compression ratio is implied by the workload: 16 MiB of mostly-zero
//! weights encode to roughly 1 MiB (the one incompressible weight) plus a small
//! manifest, i.e. ≈15×. The ratio itself is pinned by the checkpoint unit
//! tests, not by these timings.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;

use ::bench::harness::{report, run};
use ::checkpoint::deltas::delta;
use ::checkpoint::merge::merge;
use ::checkpoint::quantize::{dequantize, quantize};
use ::checkpoint::uformat::snapshot;
use ::tensor::{dtype, shape};

/// Square weight width of the workload.
const DIM: usize = 512;

/// Number of square weights in the workload.
const TENSORS: usize = 16;

/// Fills `bytes` with a deterministic pseudo-random stream (an LCG), so the
/// content-defined chunks of the last weight do not compress.
fn lcgfill(bytes: &mut [u8]) {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for chunk in bytes.chunks_mut(8) {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        for (dst, src) in chunk.iter_mut().zip(state.to_le_bytes()) {
            *dst = src;
        }
    }
}

/// Builds a snapshot of [`TENSORS`] square `f32` weights. The last weight is
/// pseudo-random when `random_last` is true, and zeroed otherwise.
fn snapshotwith(random_last: bool) -> snapshot {
    let mut snap = snapshot::new(1);
    for i in 0..TENSORS {
        let mut bytes = vec![0u8; DIM * DIM * 4];
        if random_last && i == TENSORS - 1 {
            lcgfill(&mut bytes);
        }
        snap.addtensor(
            format!("w{i}"),
            dtype::f32,
            shape::new(vec![DIM, DIM]).unwrap(),
            &bytes,
        )
        .unwrap();
    }
    snap
}

fn encode() {
    let snap = snapshotwith(true);
    report(&run("snapshot/encode", || {
        black_box(snap.encode().unwrap())
    }));
}

fn decode() {
    let snap = snapshotwith(true);
    let bytes = snap.encode().unwrap();
    report(&run("snapshot/decode", || {
        black_box(snapshot::decode(&bytes).unwrap())
    }));
}

fn deltacreate() {
    let base = snapshotwith(false);
    let new = snapshotwith(true);
    report(&run("delta/create", || {
        black_box(delta::fromsnapshots(&base, &new))
    }));
}

fn mergefull() {
    let base = snapshotwith(false);
    let new = snapshotwith(true);
    let d = delta::fromsnapshots(&base, &new);
    report(&run("merge/full", || black_box(merge(&base, &d).unwrap())));
}

/// Builds a deterministic pseudo-random `f32` slice (an LCG stream).
fn f32data(len: usize) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

fn quantizebench() {
    let data = f32data(DIM * DIM * TENSORS);
    report(&run("quantize/per_tensor", || {
        black_box(quantize(&data, 0).unwrap())
    }));
    report(&run("quantize/per_group_128", || {
        black_box(quantize(&data, 128).unwrap())
    }));
}

fn dequantizebench() {
    let data = f32data(DIM * DIM * TENSORS);
    let (q, scale) = quantize(&data, 0).unwrap();
    report(&run("dequantize/per_tensor", || {
        black_box(dequantize(&q, &scale, 0).unwrap())
    }));
}

fn main() {
    encode();
    decode();
    deltacreate();
    mergefull();
    quantizebench();
    dequantizebench();
}
