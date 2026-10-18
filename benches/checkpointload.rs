//! End-to-end checkpoint loading benchmarks: three competing load paths.
//!
//! The three measured paths load the same deterministic model workload and
//! decode every tensor into a typed [`tensor`]:
//!
//! * `load/u_full` — the training `.u` snapshot through
//!   [`checkpointreader::openfullfile`], one typed read per tensor;
//! * `load/raw_typed` — the inference raw format through
//!   [`rawreader::fromfile`], one typed read per section;
//! * `load/raw_readall` — the raw format through [`rawreader::readall`], a
//!   single sequential pass over the data section followed by per-section
//!   decoding.
//!
//! Files are materialized once in a scratch directory; each iteration reopens
//! the file and reloads every tensor, so the measurement covers the full
//! open-parse-read-decode path under a warm page cache. The workload bytes are
//! deterministic pseudo-random, so the `.u` encoder stores raw blocks and the
//! comparison isolates format and read strategy rather than compression.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use ::checkpoint::checkpointreader;
use ::checkpoint::rawfmt::rawexport;
use ::checkpoint::rawio::rawreader;
use ::checkpoint::uformat::snapshot;
use ::tensor::f16;
use ::tensor::{dtype, shape, tensor};
use criterion::{Criterion, criterion_group, criterion_main};

/// Vocabulary size of the embedding workload.
const VOCAB: usize = 4096;

/// Hidden width of the workload.
const DIM: usize = 1024;

/// Transformer layer count.
const LAYERS: usize = 24;

/// Number of half-precision tensors interspersed in the workload.
const F16COUNT: usize = 6;

/// The names of the `f32` tensors, in workload order.
fn f32names() -> Vec<String> {
    let mut names = vec!["embed".to_string()];
    for i in 0..LAYERS {
        for suffix in ["q", "k", "v"] {
            names.push(format!("layer{i}.{suffix}"));
        }
    }
    names
}

/// The names of the `f16` tensors.
fn f16names() -> Vec<String> {
    (0..F16COUNT).map(|i| format!("fp16{i}")).collect()
}

/// Fills `bytes` with a deterministic pseudo-random stream (an LCG), so the
/// `.u` encoder's content-defined chunks do not compress and both formats store
/// raw data.
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

/// Builds the deterministic model workload as a training `.u` snapshot.
fn workload() -> snapshot {
    let mut snap = snapshot::new(1);
    let mut bytes = vec![0u8; VOCAB * DIM * 4];
    lcgfill(&mut bytes);
    snap.addtensor(
        "embed".to_string(),
        dtype::f32,
        shape::new(vec![VOCAB, DIM]).unwrap(),
        &bytes,
    )
    .unwrap();
    for i in 0..LAYERS {
        for suffix in ["q", "k", "v"] {
            let mut bytes = vec![0u8; DIM * DIM * 4];
            lcgfill(&mut bytes);
            snap.addtensor(
                format!("layer{i}.{suffix}"),
                dtype::f32,
                shape::new(vec![DIM, DIM]).unwrap(),
                &bytes,
            )
            .unwrap();
        }
    }
    for i in 0..F16COUNT {
        let mut bytes = vec![0u8; DIM * DIM * 2];
        lcgfill(&mut bytes);
        snap.addtensor(
            format!("fp16{i}"),
            dtype::f16,
            shape::new(vec![DIM, DIM]).unwrap(),
            &bytes,
        )
        .unwrap();
    }
    snap
}

/// A scratch directory holding the two workload artifacts.
struct scratch {
    u_path: PathBuf,
    raw_path: PathBuf,
}

impl scratch {
    fn build() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir =
            std::env::temp_dir().join(format!("llm-load-bench-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let u_path = dir.join("workload.u");
        let raw_path = dir.join("workload.raw");
        let snap = workload();
        std::fs::write(&u_path, snap.encode().unwrap()).unwrap();
        rawexport(&u_path, &raw_path).unwrap();
        Self { u_path, raw_path }
    }
}

impl Drop for scratch {
    fn drop(&mut self) {
        if let Some(parent) = self.u_path.parent() {
            let _ = std::fs::remove_dir_all(parent);
        }
    }
}

fn ufull(c: &mut Criterion) {
    let s = scratch::build();
    c.bench_function("load/u_full", |b| {
        b.iter(|| {
            let reader = checkpointreader::openfullfile(&s.u_path).unwrap();
            let mut total = 0usize;
            for name in f32names() {
                let t = reader.readtensortyped::<f32>(&name).unwrap().unwrap();
                total += t.getlen();
            }
            for name in f16names() {
                let t = reader.readtensortyped::<f16>(&name).unwrap().unwrap();
                total += t.getlen();
            }
            black_box(total);
        });
    });
}

fn rawtyped(c: &mut Criterion) {
    let s = scratch::build();
    c.bench_function("load/raw_typed", |b| {
        b.iter(|| {
            let reader = rawreader::fromfile(&s.raw_path).unwrap();
            let mut total = 0usize;
            for name in f32names() {
                let t = reader.readtensortyped::<f32>(&name).unwrap().unwrap();
                total += t.getlen();
            }
            for name in f16names() {
                let t = reader.readtensortyped::<f16>(&name).unwrap().unwrap();
                total += t.getlen();
            }
            black_box(total);
        });
    });
}

fn rawreadall(c: &mut Criterion) {
    let s = scratch::build();
    c.bench_function("load/raw_readall", |b| {
        b.iter(|| {
            let reader = rawreader::fromfile(&s.raw_path).unwrap();
            let all = reader.readall().unwrap();
            let mut total = 0usize;
            for (entry, (_, bytes)) in reader.gettensors().iter().zip(all.iter()) {
                match entry.getdtype() {
                    dtype::f32 => {
                        let t: tensor<f32> =
                            tensor::tryfrombytes(entry.getshape().clone(), bytes).unwrap();
                        total += t.getlen();
                    }
                    dtype::f16 => {
                        let t: tensor<f16> =
                            tensor::tryfrombytes(entry.getshape().clone(), bytes).unwrap();
                        total += t.getlen();
                    }
                    _ => {}
                }
            }
            black_box(total);
        });
    });
}

/// The zero-materialization path: `readtensorview` aliases the memory-mapped
/// file for every raw `f32` tensor (page-aligned data section), while
/// `readtensortyped` allocates an owned buffer per tensor. The difference is
/// the materialization cost the loading pipeline skips when it feeds the
/// borrowed view straight into a backend operand.
fn viewvsowned(c: &mut Criterion) {
    let s = scratch::build();
    let mut group = c.benchmark_group("load/view");
    group.bench_function("owned_typed", |b| {
        b.iter(|| {
            let reader = checkpointreader::openfullfile(&s.u_path).unwrap();
            let mut total = 0usize;
            for name in f32names() {
                let t = reader.readtensortyped::<f32>(&name).unwrap().unwrap();
                total += t.getlen();
            }
            black_box(total);
        });
    });
    group.bench_function("borrowed_view", |b| {
        b.iter(|| {
            let reader = checkpointreader::openfullfile(&s.u_path).unwrap();
            let mut total = 0usize;
            for name in f32names() {
                let v = reader.readtensorview::<f32>(&name).unwrap().unwrap();
                total += v.as_ref().len();
            }
            black_box(total);
        });
    });
    group.finish();
}

criterion_group!(benches, ufull, rawtyped, rawreadall, viewvsowned);
criterion_main!(benches);
