//! 大 K GEMM 基准：稠密 f32 / int8 的标量与分块对照，以及线程化扩展余量。
//!
//! 报 ns/iter 与读入字节吞吐；`gemm_threads` 报乘加量吞吐（`GElem/s`），因为那几条路径读入
//! 字节数相同、差别在算术与并行度。所有对照的输出逐位一致（对拍在 `op::matmul` 与
//! `op::matmul_i8` 的测试里）。

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use op::{fast, matmul, matmul_i8};
use std::hint::black_box;
use std::thread;

/// 输出行数 `m`。
const M: usize = 256;

/// 输出列数 `n`。
const N: usize = 256;

/// 收缩轴 `k`（prefill 主链路代表值）。
const K: usize = 4096;

/// 参与线程对照的形状 `(m, n, k)`：prefill 主链路与近方阵各一档。
const SHAPES: [(usize, usize, usize); 2] = [(256, 256, 4096), (512, 512, 512)];

/// 稠密 f32：朴素 i→j→p 与分块 rank-1 更新。
pub(crate) fn dense(c: &mut Criterion) {
    let a: Vec<f32> = (0..M * K).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
    let b: Vec<f32> = (0..K * N).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
    let mut out = vec![0.0f32; M * N];

    let mut group = c.benchmark_group("matmul_f32_256x256_k4096");
    group.throughput(Throughput::Bytes(
        ((a.len() + b.len()) * size_of::<f32>()) as u64,
    ));
    group.bench_function("scalar", |bench| {
        bench.iter(|| {
            matmul::forward_scalar(
                black_box(&a),
                black_box(&b),
                black_box(M),
                black_box(N),
                black_box(K),
                black_box(&mut out),
            )
            .expect("缓冲尺寸与 m·n·k 一致")
        })
    });
    group.bench_function("tiled", |bench| {
        bench.iter(|| {
            matmul::forward_tiled(
                black_box(&a),
                black_box(&b),
                black_box(M),
                black_box(N),
                black_box(K),
                black_box(&mut out),
            )
            .expect("缓冲尺寸与 m·n·k 一致")
        })
    });
    group.finish();

    projection(c);
}

/// 线性层布局 `[rows, k] × [n, k]`（权重不转置）：逐行 `gemv`、tile 与寄存器分块 + 常驻池对照。
///
/// 这一组量的是"权重读几遍"：逐行路径每行读一遍整块权重，`tiled` 一行权重读一遍服务全部行，
/// `packed` 再叠加 8×8 寄存器分块与多线程。三者算术相同，输出逐位一致。
fn projection(c: &mut Criterion) {
    let rows = M;
    let x: Vec<f32> = (0..rows * K)
        .map(|i| ((i % 5) as f32) * 0.25 - 0.5)
        .collect();
    let w: Vec<f32> = (0..N * K).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
    let mut out = vec![0.0f32; rows * N];

    let mut group = c.benchmark_group("matmul_tn_f32_256x256_k4096");
    group.throughput(Throughput::Bytes(
        ((x.len() + w.len()) * size_of::<f32>()) as u64,
    ));
    group.bench_function("row_by_row_gemv", |bench| {
        bench.iter(|| {
            for row in 0..rows {
                op::gemv_f32::forward(
                    black_box(&x[row * K..][..K]),
                    black_box(&w),
                    black_box(&mut out[row * N..][..N]),
                )
                .expect("缓冲尺寸与 m·n·k 一致");
            }
        })
    });
    group.bench_function("tiled", |bench| {
        bench.iter(|| {
            matmul::forward_tn_tiled(
                black_box(&x),
                black_box(&w),
                black_box(rows),
                black_box(N),
                black_box(K),
                black_box(&mut out),
            )
            .expect("缓冲尺寸与 m·n·k 一致")
        })
    });
    group.bench_function("packed", |bench| {
        bench.iter(|| {
            op::fast::gemm_f32_tn(
                black_box(&x),
                black_box(&w),
                black_box(rows),
                black_box(N),
                black_box(K),
                black_box(&mut out),
            )
        })
    });
    group.finish();
}

/// 大 K int8：标量参考与分块 tile，累加器都是 i32。
pub(crate) fn int8(c: &mut Criterion) {
    let a: Vec<i8> = (0..M * K).map(|i| ((i % 64) as i8) - 32).collect();
    let b: Vec<i8> = (0..K * N).map(|i| ((i % 64) as i8) - 32).collect();
    let mut out = vec![0i32; M * N];

    let mut group = c.benchmark_group("matmul_i8_256x256_k4096");
    group.throughput(Throughput::Bytes(
        ((a.len() + b.len()) * size_of::<i8>()) as u64,
    ));
    group.bench_function("scalar", |bench| {
        bench.iter(|| {
            matmul_i8::forward_scalar(
                black_box(&a),
                black_box(&b),
                black_box(M),
                black_box(N),
                black_box(K),
                black_box(&mut out),
            )
            .expect("缓冲尺寸与 m*n*k 一致")
        })
    });
    group.bench_function("tiled", |bench| {
        bench.iter(|| {
            matmul_i8::forward_tiled(
                black_box(&a),
                black_box(&b),
                black_box(M),
                black_box(N),
                black_box(K),
                black_box(&mut out),
            )
            .expect("缓冲尺寸与 m*n*k 一致")
        })
    });
    group.finish();
}

/// 单线程分块、单线程快速内核、快速内核线程化（i8 只有分块内核及其线程化版本）。
pub(crate) fn threads(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_threads");
    group.sample_size(15);
    f32_paths(&mut group);
    i8_paths(&mut group);
    group.finish();
}

/// 可用核数（取不到时按 1 算，不猜）。
fn cores() -> usize {
    thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
}

/// 把 `m` 行切给线程：每个线程只碰自己那段输出行，`run` 收到该段的起始行与行数。
///
/// 切分粒度取 `ceil(m / 核数)`，因此线程数不超过 `m`；`out` 按 `chunk * n` 切块，
/// 每块的行数由块长反推，最后一块自然短一些。
fn spread<T: Send>(m: usize, n: usize, out: &mut [T], run: impl Fn(usize, usize, &mut [T]) + Sync) {
    let chunk = m.div_ceil(cores().min(m));
    thread::scope(|scope| {
        for (block, part) in out.chunks_mut(chunk * n).enumerate() {
            let run = &run;
            let start = block * chunk;
            scope.spawn(move || run(start, part.len() / n, part));
        }
    });
}

/// f32 三档：单线程分块、单线程快速内核、快速内核的线程化版本。
fn f32_paths(group: &mut BenchmarkGroup<'_, WallTime>) {
    for (m, n, k) in SHAPES {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
        let mut out = vec![0.0f32; m * n];
        let shape = format!("{m}x{n}x{k}");
        group.throughput(Throughput::Elements((2 * m * n * k) as u64));

        group.bench_function(BenchmarkId::new("f32", &shape), |bencher| {
            bencher.iter(|| {
                matmul::matmul_tiled(black_box(&a), black_box(&b), m, n, k, black_box(&mut out));
            });
        });
        group.bench_function(BenchmarkId::new("f32_fast_single", &shape), |bencher| {
            bencher.iter(|| {
                fast::gemm_f32_single(black_box(&a), black_box(&b), m, n, k, black_box(&mut out));
            });
        });
        group.bench_function(BenchmarkId::new("f32_fast_threads", &shape), |bencher| {
            bencher.iter(|| {
                fast::gemm_f32(black_box(&a), black_box(&b), m, n, k, black_box(&mut out));
            });
        });
    }
}

/// i8 两档：单线程分块与线程化分块（快速内核只覆盖 f32）。
fn i8_paths(group: &mut BenchmarkGroup<'_, WallTime>) {
    for (m, n, k) in SHAPES {
        let a: Vec<i8> = (0..m * k).map(|i| ((i % 13) as i8) - 6).collect();
        let b: Vec<i8> = (0..k * n).map(|i| ((i % 11) as i8) - 5).collect();
        let mut out = vec![0i32; m * n];
        let shape = format!("{m}x{n}x{k}");
        group.throughput(Throughput::Elements((2 * m * n * k) as u64));

        group.bench_function(BenchmarkId::new("i8", &shape), |bencher| {
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
        group.bench_function(BenchmarkId::new("i8_threads", &shape), |bencher| {
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
        });
    }
}
