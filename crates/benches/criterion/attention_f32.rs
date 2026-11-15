//! `attention::flash_forward` 的微基准：decode 单头 128 个 KV 行、头宽 64 的 online softmax 注意力。
//!
//! 与 `softmax_f32` 同一口径：**ns/iter，越小越好**，另报元素吞吐（elems/s）。online 路径
//! 单遍维护最大值、归一化分母与累加向量，遍历顺序与标量参考一致；预热、采样与测量时长由
//! criterion 配置决定，本文件只声明一次迭代做什么。
//!
//! 基线判定（`tool::regress`）随靶机首次实测后登记，未登记前本基准只出测量、不回退判定。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use op::attention;
use std::hint::black_box;

/// KV 序列长度（decode 代表值）。
const S: usize = 128;

/// 头宽（head_dim）。
const D: usize = 64;

fn bench_flash(c: &mut Criterion) {
    let len = S * D;
    let q: Vec<f32> = (0..len).map(|i| -0.5 + i as f32 * 0.13).collect();
    let k: Vec<f32> = (0..len).map(|i| 0.4 - i as f32 * 0.09).collect();
    let v: Vec<f32> = (0..len).map(|i| -0.2 + i as f32 * 0.11).collect();
    let mut out = vec![0.0f32; len];

    let mut group = c.benchmark_group("attention_f32");
    group.throughput(Throughput::Elements(len as u64));
    group.bench_function("flash_decode_128x64", |bench| {
        bench.iter(|| {
            attention::flash_forward(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(&mut out),
                S,
                D,
            )
            .expect("q/k/v/out 等长且为 S·D")
        })
    });

    group.finish();
}

criterion_group!(benches, bench_flash);
criterion_main!(benches);
