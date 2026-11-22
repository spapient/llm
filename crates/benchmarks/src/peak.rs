//! 职责：f32 FMA 端口峰值的单线程测定。

use std::hint::black_box;

use criterion::{Criterion, Throughput};

/// f32 FMA 端口峰值：寄存器常驻、4 条累加链互不依赖，用来把 GF/s 换算成峰值百分比。
///
/// 只有先测出本机峰值，「GEMM 达 90% 端口」才是结论；引用别处的时钟数只是猜。
/// 链数 × lane 必须留在 32 个 128 位寄存器内（4×16 = 16 个），多一条就溢出到栈，
/// 量到的会是访存带宽而不是 FMA 端口。
pub(crate) fn flops_peak(c: &mut Criterion) {
    let mut group = c.benchmark_group("flops_peak");
    group.throughput(Throughput::Elements((2 * LANES * CHAIN * ROUNDS) as u64));
    group.bench_function("f32", |iter| {
        iter.iter(|| {
            fma_burst(
                black_box(&mut [[0.0f32; LANES]; CHAIN]),
                black_box(&[1.000_000_1; LANES]),
            )
        });
    });
    group.finish();
}

/// 每链 16 个 f32 即 4 条 128 位寄存器。
const LANES: usize = 16;
const CHAIN: usize = 4;
const ROUNDS: usize = 4096;

/// `ROUNDS × CHAIN × LANES` 次乘加，结果求和返回以避免整段被消除。
fn fma_burst(lanes: &mut [[f32; LANES]; CHAIN], step: &[f32; LANES]) -> f32 {
    for _ in 0..ROUNDS {
        for chain in lanes.iter_mut() {
            for (value, factor) in chain.iter_mut().zip(step) {
                *value = value.mul_add(*factor, *factor);
            }
        }
    }
    lanes.iter().flatten().sum()
}
