//! 逐元素与行内算子基准：elementwise、gemv、softmax、rope、rmsnorm、silu。
//!
//! 六组都只声明"一次迭代做什么"与每迭代元素数；预热、采样与测量时长由 src 配置决定。
//! 基线判定在 `tests/regress`，本文件不写性能目标数字。

use criterion::{Criterion, Throughput};
use op::add_f32::Add;
use op::{elementwise, gemv_f32, rmsnorm_f32, rope_f32, silu_f32, softmax_f32};
use std::hint::black_box;

/// decode 典型行宽（hidden 维度）。
const ROW: usize = 4096;

/// elementwise 的向量长度（输入与输出各 256 KiB）。
const VECTOR: usize = 1 << 16;

/// 一行 decode 输入与同长输出。
fn row() -> (Vec<f32>, Vec<f32>) {
    let x = (0..ROW).map(|i| i as f32 * 0.25 - 2.0).collect();
    (x, vec![0.0f32; ROW])
}

/// `elementwise::forward::<Add>`：1<<16 个 f32 的逐元素加法。
pub(crate) fn elementwise(c: &mut Criterion) {
    let a: Vec<f32> = (0..VECTOR).map(|i| i as f32 * 0.5 - 1.0).collect();
    let b: Vec<f32> = (0..VECTOR).map(|i| i as f32 * 0.25 + 0.5).collect();
    let mut out = vec![0.0f32; VECTOR];

    let mut group = c.benchmark_group("elementwise/add_f32");
    group.throughput(Throughput::Elements(VECTOR as u64));
    group.bench_function("forward", |bench| {
        bench.iter(|| {
            elementwise::forward::<Add>(black_box(&a), black_box(&b), black_box(&mut out))
                .expect("缓冲等长")
        })
    });
    group.finish();
}

/// `gemv_f32::forward`：一次 decode 步（1×4096×4096），权重全读一遍。
pub(crate) fn gemv(c: &mut Criterion) {
    let weights: Vec<f32> = (0..ROW * ROW).map(|i| (i as f32 * 0.001).sin()).collect();
    let x: Vec<f32> = (0..ROW).map(|i| (i as f32 * 0.002).cos()).collect();
    let mut out = vec![0.0f32; ROW];

    let mut group = c.benchmark_group("gemv_f32");
    // 时间与带宽两条口径都归档：读者不必自己换算。
    group.throughput(Throughput::Bytes((ROW * ROW * size_of::<f32>()) as u64));
    group.bench_function("decode_1x4096x4096", |bench| {
        bench.iter(|| {
            gemv_f32::forward(black_box(&x), black_box(&weights), black_box(&mut out))
                .expect("形状与缓冲等长")
        })
    });
    group.finish();
}

/// `softmax_f32::forward`：decode 一行 4096 个 f32 的三趟数值稳定归一化。
pub(crate) fn softmax(c: &mut Criterion) {
    let (x, mut out) = row();

    let mut group = c.benchmark_group("softmax_f32");
    group.throughput(Throughput::Elements(ROW as u64));
    group.bench_function("decode_1x4096", |bench| {
        bench.iter(|| {
            softmax_f32::forward(black_box(&x), black_box(&mut out)).expect("输入与输出等长")
        })
    });
    group.finish();
}

/// `rope_f32::forward`：decode 一行 4096 个 f32 的二维位置旋转（位置取 7）。
pub(crate) fn rope(c: &mut Criterion) {
    let (x, mut out) = row();

    let mut group = c.benchmark_group("rope_f32");
    group.throughput(Throughput::Elements(ROW as u64));
    group.bench_function("decode_1x4096", |bench| {
        bench.iter(|| {
            rope_f32::forward(7, black_box(&x), black_box(&mut out)).expect("输入与输出等长")
        })
    });
    group.finish();
}

/// `rmsnorm_f32::forward`：decode 一行 4096 个 f32 的均方根归一化。
pub(crate) fn rmsnorm(c: &mut Criterion) {
    let (x, mut out) = row();
    let gamma: Vec<f32> = (0..ROW).map(|i| 0.5 + i as f32 * 0.01).collect();

    let mut group = c.benchmark_group("rmsnorm_f32");
    group.throughput(Throughput::Elements(ROW as u64));
    group.bench_function("decode_1x4096", |bench| {
        bench.iter(|| {
            rmsnorm_f32::forward(black_box(&x), black_box(&gamma), black_box(&mut out))
                .expect("三个缓冲等长")
        })
    });
    group.finish();
}

/// `silu_f32::forward`：decode 一行 4096 个 f32 的逐元素激活。
pub(crate) fn silu(c: &mut Criterion) {
    let x: Vec<f32> = (0..ROW).map(|i| i as f32 * 0.25 - 4.0).collect();
    let mut out = vec![0.0f32; ROW];

    let mut group = c.benchmark_group("silu_f32");
    group.throughput(Throughput::Elements(ROW as u64));
    group.bench_function("decode_1x4096", |bench| {
        bench
            .iter(|| silu_f32::forward(black_box(&x), black_box(&mut out)).expect("输入与输出等长"))
    });
    group.finish();
}
