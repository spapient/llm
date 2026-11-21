//! `silu_f32::forward` 的微基准：decode 一行 4096 个 f32 的 SiLU 激活。
//!
//! 与 `softmax_f32` 同一口径：**ns/iter，越小越好**，另报元素吞吐（elems/s）。逐元素
//! `x / (1 + exp(-x))`，遍历顺序与标量参考一致；预热、采样与测量时长由 criterion
//! 配置决定，本文件只声明一次迭代做什么。
//!
//! 基线判定（`tools::regress`）随靶机首次实测后登记，未登记前本基准只出测量、不回退判定。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use op::silu_f32;
use std::hint::black_box;

/// decode 典型行宽（hidden 维度），与 rope 基准同量级。
const LEN: usize = 4096;

fn bench_forward(c: &mut Criterion) {
    let a: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.25 - 4.0).collect();
    let mut out = vec![0.0f32; LEN];

    let mut group = c.benchmark_group("silu_f32");
    group.throughput(Throughput::Elements(LEN as u64));
    group.bench_function("decode_1x4096", |bench| {
        bench
            .iter(|| silu_f32::forward(black_box(&a), black_box(&mut out)).expect("输入与输出等长"))
    });

    group.finish();
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
