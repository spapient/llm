//! 大 K int8 GEMM 的微基准：256×256、收缩轴 4096。
//!
//! 标量参考（i→j→p）与分块 tile（rank-1 更新）对照；累加器都是 i32，结果逐位一致，
//! 本基准只比较遍历次序对墙钟的影响。ns/iter 越小越好，另报读入字节吞吐。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use op::matmul_i8;
use std::hint::black_box;

/// 输出行数 m。
const M: usize = 256;

/// 输出列数 n。
const N: usize = 256;

/// 收缩轴 k（prefill 主链路代表值）。
const K: usize = 4096;

fn bench(c: &mut Criterion) {
    let a: Vec<i8> = (0..M * K).map(|i| ((i % 64) as i8) - 32).collect();
    let b: Vec<i8> = (0..K * N).map(|i| ((i % 64) as i8) - 32).collect();
    let mut out = vec![0i32; M * N];

    let read_bytes = ((a.len() + b.len()) * size_of::<i8>()) as u64;
    let mut group = c.benchmark_group("matmul_i8_256x256_k4096");
    group.throughput(Throughput::Bytes(read_bytes));
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

criterion_group!(benches, bench);
criterion_main!(benches);
