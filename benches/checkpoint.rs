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

use ::checkpoint::deltas::delta;
use ::checkpoint::merge::merge;
use ::checkpoint::uformat::snapshot;
use ::tensor::{dtype, shape};
use criterion::{Criterion, criterion_group, criterion_main};

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

fn encode(c: &mut Criterion) {
    let snap = snapshotwith(true);
    c.bench_function("snapshot/encode", |b| {
        b.iter(|| black_box(snap.encode().unwrap()))
    });
}

fn decode(c: &mut Criterion) {
    let snap = snapshotwith(true);
    let bytes = snap.encode().unwrap();
    c.bench_function("snapshot/decode", |b| {
        b.iter(|| black_box(snapshot::decode(&bytes).unwrap()))
    });
}

fn deltacreate(c: &mut Criterion) {
    let base = snapshotwith(false);
    let new = snapshotwith(true);
    c.bench_function("delta/create", |b| {
        b.iter(|| black_box(delta::fromsnapshots(&base, &new)))
    });
}

fn mergefull(c: &mut Criterion) {
    let base = snapshotwith(false);
    let new = snapshotwith(true);
    let d = delta::fromsnapshots(&base, &new);
    c.bench_function("merge/full", |b| {
        b.iter(|| black_box(merge(&base, &d).unwrap()))
    });
}

criterion_group!(benches, encode, decode, deltacreate, mergefull);
criterion_main!(benches);
