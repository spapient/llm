//! `softmax_f32::forward` 的微基准：decode 一行 4096 个 f32 的数值稳定归一化。
//!
//! 与 `gemv_f32` 同一口径：**ns/iter，越小越好**，另报元素吞吐（elems/s）。三趟遍历
//! （求最大、指数和、归一化）的顺序与标量参考一致；预热、采样与测量时长由 criterion
//! 配置决定，本文件只声明一次迭代做什么。
//!
//! 基线判定（`tool::regress`）随靶机首次实测后登记，未登记前本基准只出测量、不回退判定。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use op::softmax_f32;
use std::hint::black_box;

/// decode 典型行宽（hidden 维度），与 gemv 基准同量级。
const LEN: usize = 4096;

fn bench_forward(c: &mut Criterion) {
    let x: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.25 - 2.0).collect();
    let mut out = vec![0.0f32; LEN];

    let mut group = c.benchmark_group("softmax_f32");
    group.throughput(Throughput::Elements(LEN as u64));
    group.bench_function("decode_1x4096", |bench| {
        bench.iter(|| {
            softmax_f32::forward(black_box(&x), black_box(&mut out)).expect("输入与输出等长")
        })
    });

    group.finish();
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
