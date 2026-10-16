//! bit-shuffle + delta 变换基准（`FLAGTRANSFORMED`）。
//!
//! 两个层面：(1) 原语吞吐——`bitshuffle_delta`/`unbitshuffle_delta` 在
//! 16 MiB 相关 f32 权重上的变换速度；(2) 快照级写/读路径——`snapshot::encode`
//! 与 `snapshot::decode` 的吞吐与压缩率，覆盖「相关 f32 权重（变换生效）」
//! 与「纯随机 f32（raw 回退）」两个剖面。
//!
//! 压缩率基线（16 MiB、256 KiB 块、release）：相关 f32
//! 权重裸 gate ≈0%（略膨胀），`bit+delta + gate` 省 41.6%；纯随机两者都≈0%
//! （raw 回退）。本基准在快照路径上重测同一事实。

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;

use ::checkpoint::blocks::{bitshuffle_delta, unbitshuffle_delta};
use ::checkpoint::uformat::snapshot;
use ::tensor::{dtype, shape};
use criterion::{Criterion, criterion_group, criterion_main};

const SIZE: usize = 16 * 1024 * 1024;

fn xorshift64(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

/// Correlated random walk: the realistic weight structure the pre-pass is
/// designed for (70% small steps, 20% jumps, 10% repeats).
fn gen_weights_real(seed: &mut u64, n: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n);
    let mut val: f32 = 0.0;
    while v.len() < n {
        let r = xorshift64(seed) % 100;
        let delta: f32 = if r < 70 {
            (xorshift64(seed) % 2000) as f32 / 1e6 - 1e-3
        } else if r < 90 {
            (xorshift64(seed) % 200_000) as f32 / 1e6 - 0.1
        } else {
            0.0
        };
        val += delta;
        if val > 1.0 {
            val -= 2.0;
        }
        if val < -1.0 {
            val += 2.0;
        }
        v.extend_from_slice(&val.to_le_bytes());
    }
    v
}

fn gen_random(seed: &mut u64, n: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let x = xorshift64(seed) as u32;
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

fn snap(bytes: &[u8]) -> snapshot {
    let mut snap = snapshot::new(1);
    snap.addtensor(
        "w".to_string(),
        dtype::f32,
        shape::new(vec![bytes.len() / 4]).unwrap(),
        bytes,
    )
    .unwrap();
    snap
}

fn bench_primitives(c: &mut Criterion) {
    let mut seed = 0x1234_5678_9abc_def0u64;
    let data = gen_weights_real(&mut seed, SIZE);
    let transformed = bitshuffle_delta(&data, 4);
    c.bench_function("bitshuffle_delta/16MiB-f32", |b| {
        b.iter(|| black_box(bitshuffle_delta(black_box(&data), 4)))
    });
    c.bench_function("unbitshuffle_delta/16MiB-f32", |b| {
        b.iter(|| black_box(unbitshuffle_delta(black_box(&transformed), 4)))
    });
}

fn bench_snapshot(c: &mut Criterion) {
    let mut seed = 0x1234_5678_9abc_def0u64;
    let real = gen_weights_real(&mut seed, SIZE);
    let random = gen_random(&mut seed, SIZE);
    let real_snap = snap(&real);
    let random_snap = snap(&random);

    let real_encoded = real_snap.encode().unwrap();
    let random_encoded = random_snap.encode().unwrap();
    println!(
        "correlated-f32: encoded {} / input {} = {:.2}% remaining (saved {:.1}%)",
        real_encoded.len(),
        real.len(),
        100.0 * real_encoded.len() as f64 / real.len() as f64,
        100.0 * (1.0 - real_encoded.len() as f64 / real.len() as f64)
    );
    println!(
        "random-f32:     encoded {} / input {} = {:.2}% remaining (raw fallback)",
        random_encoded.len(),
        random.len(),
        100.0 * random_encoded.len() as f64 / random.len() as f64
    );

    c.bench_function("encode/correlated-f32", |b| {
        b.iter(|| black_box(real_snap.encode().unwrap()))
    });
    c.bench_function("decode/correlated-f32", |b| {
        b.iter(|| black_box(snapshot::decode(black_box(&real_encoded)).unwrap()))
    });
    c.bench_function("encode/random-f32", |b| {
        b.iter(|| black_box(random_snap.encode().unwrap()))
    });
}

criterion_group!(benches, bench_primitives, bench_snapshot);
criterion_main!(benches);
