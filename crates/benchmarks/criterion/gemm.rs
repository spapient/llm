//! 小 K 稠密 / 量化矩阵乘基准：256×256、收缩轴 `K = 8`。
//!
//! 每组都把执行路径（`forward`）与同一份算术的显式标量入口（`op::gemm::matmul`）并列，
//! 结果逐位一致，只比较遍历次序与解包次数。报 ns/iter 与读入字节吞吐。

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput};
use op::gemm::{self, Gemm};
use op::{gemm_f32, gemm_i2, gemm_i4, gemm_i8};
use std::hint::black_box;

/// 输出行数 `m`。
const M: usize = 256;

/// 输出列数 `n`。
const N: usize = 256;

/// f32：`prefill_256x256_k8` 走执行路径，`scalar_*` 走标量参考。
///
/// `K = 8` 时一行 A 就在寄存器里，分块实测慢 2~3 倍，所以这条格式不分块；本基准把这点留在
/// 台架上可复现。量化格式的分块收益见 [`quant`]。
pub(crate) fn dense(c: &mut Criterion) {
    let k = op::gemm::K;
    let a: Vec<f32> = (0..M * k).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect();
    let b: Vec<f32> = (0..k * N).map(|i| (i % 7) as f32 * 0.2 - 0.4).collect();
    let mut out = vec![0.0f32; M * N];

    let mut group = c.benchmark_group("gemm_f32");
    group.throughput(Throughput::Bytes(((a.len() + b.len()) * size_of::<f32>()) as u64));
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

/// 量化三格式：`scalar_*` 逐 `(i, j)` 重解位流，`tiled_*` 只解一次再按列块整数累加。
pub(crate) fn quant(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_quant");
    quant_pair::<gemm_i2::GemmI2>(&mut group, "i2_256x256_k8");
    quant_pair::<gemm_i4::GemmI4>(&mut group, "i4_256x256_k8");
    quant_pair::<gemm_i8::GemmI8>(&mut group, "i8_256x256_k8");
    group.finish();
}

/// 一种格式的两档：同一份 A/B 缓冲，标量参考与 tile 执行路径。
fn quant_pair<G: Gemm>(group: &mut BenchmarkGroup<'_, WallTime>, name: &str) {
    let a = G::sample_a(M);
    let b = G::sample_b(N);
    let mut out = vec![0.0f32; M * N];
    group.throughput(Throughput::Bytes((a.len() + b.len()) as u64));

    group.bench_function(format!("scalar_{name}"), |bench| {
        bench.iter(|| gemm::matmul::<G>(black_box(&a), black_box(&b), M, N, black_box(&mut out)));
    });
    group.bench_function(format!("tiled_{name}"), |bench| {
        bench.iter(|| {
            gemm::forward::<G>(black_box(&a), black_box(&b), M, N, black_box(&mut out))
                .expect("缓冲尺寸与 m·k·n 一致")
        });
    });
}
