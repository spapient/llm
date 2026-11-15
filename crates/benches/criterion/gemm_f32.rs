//! `gemm_f32::forward` 的微基准：prefill 阶段 256×256 的稠密 f32 矩阵乘（收缩轴 K=8）。
//!
//! 与 `gemv_f32` 同一口径：**ns/iter，越小越好**，另报读入字节吞吐。双重循环遍历与标量
//! 参考一致；预热、采样与测量时长由 criterion 配置决定，本文件只声明一次迭代做什么。
//!
//! 基线判定（`tool::regress`）随靶机首次实测后登记，未登记前本基准只出测量、不回退判定。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use op::gemm_f32;
use std::hint::black_box;

/// 输出行数 `m`。
const M: usize = 256;

/// 输出列数 `n`；收缩轴 `K` 由算子模板固定为 8。
const N: usize = 256;

fn bench_forward(c: &mut Criterion) {
    let k = op::gemm::K;
    let a: Vec<f32> = (0..M * k).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect();
    let b: Vec<f32> = (0..k * N).map(|i| (i % 7) as f32 * 0.2 - 0.4).collect();
    let mut out = vec![0.0f32; M * N];

    let read_bytes = ((a.len() + b.len()) * size_of::<f32>()) as u64;
    let mut group = c.benchmark_group("gemm_f32");
    group.throughput(Throughput::Bytes(read_bytes));
    group.bench_function("prefill_256x256_k8", |bench| {
        bench.iter(|| {
            gemm_f32::forward(
                black_box(&a),
                black_box(&b),
                black_box(M),
                black_box(N),
                black_box(&mut out),
            )
            .expect("缓冲尺寸与 m·k·n 一致")
        })
    });

    group.finish();
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
