//! 大 K 稠密 GEMM 的微基准：prefill 典型 256×256、收缩轴 4096。
//!
//! 两条同源路径对照：`matmul_scalar`（朴素 i→j→p 参考）与 `matmul_tiled`（分块 rank-1 更新）。
//! 同一输入、同一输出缓冲，只换遍历顺序；ns/iter 越小越好，另报读入字节吞吐。
//! 基线判定（`tools::regress`）随靶机首次实测后登记，未登记前本基准只出测量、不回退判定。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use op::matmul;
use std::hint::black_box;

/// 输出行数 `m`。
const M: usize = 256;

/// 输出列数 `n`。
const N: usize = 256;

/// 收缩轴 `k`（prefill 主链路代表值）。
const K: usize = 4096;

fn bench(c: &mut Criterion) {
    let a: Vec<f32> = (0..M * K).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
    let b: Vec<f32> = (0..K * N).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
    let mut out = vec![0.0f32; M * N];

    let read_bytes = ((a.len() + b.len()) * size_of::<f32>()) as u64;
    let mut group = c.benchmark_group("matmul_f32_256x256_k4096");
    group.throughput(Throughput::Bytes(read_bytes));
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

/// 线性层布局 `[rows, k] × [n, k]`（权重不转置）的 prefill 批：逐行 `gemv` 与 tile 的对照。
///
/// 这一组量的是"权重读几遍"：逐行路径每行读一遍整块权重，tile 路径一行权重只读一遍而服务全部行，
/// 于是省掉的倍数正是行数。两者算术相同，输出逐位一致（对拍见 `op::matmul` 的 `tn_*` 用例）。
fn projection(c: &mut Criterion) {
    let rows = M;
    let x: Vec<f32> = (0..rows * K)
        .map(|i| ((i % 5) as f32) * 0.25 - 0.5)
        .collect();
    let w: Vec<f32> = (0..N * K).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
    let mut out = vec![0.0f32; rows * N];

    let read_bytes = ((x.len() + w.len()) * size_of::<f32>()) as u64;
    let mut group = c.benchmark_group("matmul_tn_f32_256x256_k4096");
    group.throughput(Throughput::Bytes(read_bytes));
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
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
