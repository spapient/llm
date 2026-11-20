//! 稠密 GEMM 的线程化扩展余量：同一份行切分跑两组内核，看「多加一个核能拿回多少」。
//!
//! 三条 f32 对照压的是同一份输入与输出缓冲：`matmul_tiled`（分块 rank-1）、
//! `fast::gemm_f32_single`（打包面板的单线程微内核）、`fast::gemm_f32`（同一微内核按行切给线程）。
//! i8 只有分块内核，所以对照的是它自己与它在 `spread` 下的线程化版本。行切分只写在 `spread` 一处。
//!
//! 报的是乘加量吞吐（`GElem/s`），因为这三条路径读入字节数相同、差别在算术与并行度，而不是访存。

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use op::{fast, matmul, matmul_i8};
use std::hint::black_box;
use std::thread;

/// 参与对照的形状 `(m, n, k)`：prefill 主链路（大 K）与近方阵各一档。
const SHAPES: [(usize, usize, usize); 2] = [(256, 256, 4096), (512, 512, 512)];

/// 可用核数（取不到时按 1 算，不猜）。
fn threads() -> usize {
    thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
}

/// 把 `m` 行切给线程：每个线程只碰自己那段输出行，`run` 收到该段的起始行与行数。
///
/// 切分粒度取 `ceil(m / 核数)`，因此线程数不超过 `m`；`out` 按 `chunk * n` 切块，
/// 每块的行数由块长反推，最后一块自然短一些。
fn spread<T: Send>(m: usize, n: usize, out: &mut [T], run: impl Fn(usize, usize, &mut [T]) + Sync) {
    let chunk = m.div_ceil(threads().min(m));
    thread::scope(|scope| {
        for (block, part) in out.chunks_mut(chunk * n).enumerate() {
            let run = &run;
            let start = block * chunk;
            scope.spawn(move || run(start, part.len() / n, part));
        }
    });
}

/// f32 三档：单线程分块、单线程快速内核、快速内核的线程化版本。
fn bench_f32(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    for (m, n, k) in SHAPES {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
        let mut out = vec![0.0f32; m * n];
        let id = BenchmarkId::new("f32", format!("{m}x{n}x{k}"));
        group.throughput(Throughput::Elements((2 * m * n * k) as u64));

        group.bench_function(id.clone(), |bencher| {
            bencher.iter(|| {
                matmul::matmul_tiled(black_box(&a), black_box(&b), m, n, k, black_box(&mut out));
            });
        });
        group.bench_function(
            BenchmarkId::new("f32_fast_single", format!("{m}x{n}x{k}")),
            |bencher| {
                bencher.iter(|| {
                    fast::gemm_f32_single(
                        black_box(&a),
                        black_box(&b),
                        m,
                        n,
                        k,
                        black_box(&mut out),
                    );
                });
            },
        );
        group.bench_function(
            BenchmarkId::new("f32_fast_threads", format!("{m}x{n}x{k}")),
            |bencher| {
                bencher.iter(|| {
                    fast::gemm_f32(black_box(&a), black_box(&b), m, n, k, black_box(&mut out));
                });
            },
        );
    }
}

/// i8 两档：单线程分块与线程化分块（快速内核只覆盖 f32）。
fn bench_i8(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    for (m, n, k) in SHAPES {
        let a: Vec<i8> = (0..m * k).map(|i| ((i % 13) as i8) - 6).collect();
        let b: Vec<i8> = (0..k * n).map(|i| ((i % 11) as i8) - 5).collect();
        let mut out = vec![0i32; m * n];
        let id = BenchmarkId::new("i8", format!("{m}x{n}x{k}"));
        group.throughput(Throughput::Elements((2 * m * n * k) as u64));

        group.bench_function(id, |bencher| {
            bencher.iter(|| {
                matmul_i8::forward_tiled(
                    black_box(&a),
                    black_box(&b),
                    m,
                    n,
                    k,
                    black_box(&mut out),
                )
                .expect("几何对齐");
            });
        });
        group.bench_function(
            BenchmarkId::new("i8_threads", format!("{m}x{n}x{k}")),
            |bencher| {
                bencher.iter(|| {
                    spread(m, n, &mut out, |start, rows, part| {
                        matmul_i8::forward_tiled(
                            &a[start * k..(start + rows) * k],
                            &b,
                            rows,
                            n,
                            k,
                            part,
                        )
                        .expect("几何对齐");
                    });
                    black_box(&out);
                });
            },
        );
    }
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_threads");
    group.sample_size(15);
    bench_f32(&mut group);
    bench_i8(&mut group);
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
