//! 量化矩阵乘（`gemm_i2` / `gemm_i4` / `gemm_i8`）的微基准：256×256、收缩轴 `K = 8`。
//!
//! 每组两档：`scalar_*` 是逐 `(i, j)` 重解一遍位流的旧遍历（`op::gemm::matmul`，保留为与内核
//! 相互独立的标量参考），`tiled_*` 是执行路径——位流只解一次、再按列块做整数累加。两条结果
//! 逐位一致（对拍见 `op::gemm::test_support` 的形状矩阵），本基准只比较解包次数与内存次序对
//! 墙钟的影响：旧遍历的移位、掩码与零点平移是 `m·n·K` 次，新路径是 `m·K + K·n` 次。
//!
//! **ns/iter，越小越好**，另报读入字节吞吐（A、B 两条位流之和）。基线判定（`tool::regress`）
//! 随靶机首次实测后登记，未登记前本基准只出测量、不回退判定。

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use op::gemm::{self, Gemm};
use op::{gemm_i2, gemm_i4, gemm_i8};
use std::hint::black_box;

/// 输出行数 `m`。
const M: usize = 256;

/// 输出列数 `n`；收缩轴 `K` 由算子模板固定为 8。
const N: usize = 256;

/// 一种格式的两档：标量参考（逐元素重解位流）与 tile 执行路径，同一份 A/B 缓冲。
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

fn bench_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_quant");
    quant_pair::<gemm_i2::GemmI2>(&mut group, "i2_256x256_k8");
    quant_pair::<gemm_i4::GemmI4>(&mut group, "i4_256x256_k8");
    quant_pair::<gemm_i8::GemmI8>(&mut group, "i8_256x256_k8");
    group.finish();
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
