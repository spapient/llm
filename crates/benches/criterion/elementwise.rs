//! `elementwise::forward::<Add>` 的微基准：测量 1<<16 个 f32 的逐元素加法。
//!
//! 预热轮数、测量时长与采样次数都不在这里写死：criterion 按自己的配置决定
//! （`--warm-up-time` / `--measurement-time` / `--sample-size`），本文件只声明
//! "一次迭代做什么"与"一次迭代处理多少元素"（`Throughput::Elements`，报告因此直接是 elems/s）。
//! 与 `span_parse` 同一分工：bench 只产出实测值，是否回退由 `tool::regress` 的基线判定负责。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use op::add_f32::Add;
use op::elementwise;
use std::hint::black_box;

/// 每次迭代的元素数（输入与输出各 256 KiB）。
const LEN: usize = 1 << 16;

fn bench_forward(c: &mut Criterion) {
    let a: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.5 - 1.0).collect();
    let b: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.25 + 0.5).collect();
    let mut out = vec![0.0f32; LEN];

    let mut group = c.benchmark_group("elementwise/add_f32");
    group.throughput(Throughput::Elements(LEN as u64));
    group.bench_function("forward", |bench| {
        bench.iter(|| {
            elementwise::forward::<Add>(black_box(&a), black_box(&b), black_box(&mut out))
                .expect("缓冲等长")
        })
    });

    group.finish();
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
