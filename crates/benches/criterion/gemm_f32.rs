//! `gemm_f32::forward` 的微基准：prefill 阶段 256×256 的稠密 f32 矩阵乘（收缩轴 K=8）。
//!
//! 与 `gemv_f32` 同一口径：**ns/iter，越小越好**，另报读入字节吞吐。`prefill_256x256_k8` 是
//! 执行路径（`forward`），`scalar_256x256_k8` 是同一份算术的显式标量入口（`op::gemm::matmul`）。
//!
//! 两档在稠密 f32 上**故意相同**：`K = 8` 时一行 A 的权重就在寄存器里，分块（列块或
//! `matmul_tiled` 的三级块）实测慢 2~3 倍，故这条格式不分块——本基准把这一点留在台架上可复现，
//! 而不是只在注释里承诺。量化格式的分块收益见 `gemm_quant`。
//!
//! 基线判定（`tool::regress`）随靶机首次实测后登记，未登记前本基准只出测量、不回退判定。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use op::gemm;
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
    group.bench_function("scalar_256x256_k8", |bench| {
        bench.iter(|| {
            gemm::matmul::<gemm_f32::GemmF32>(
                black_box(&a),
                black_box(&b),
                black_box(M),
                black_box(N),
                black_box(&mut out),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
