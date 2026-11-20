//! decode 注意力两条内核的微基准：朴素 [`attention::decode_forward`] 与 Flash
//! [`attention::decode_flash_forward`]，单查询行（`q_rows = 1`）对变长历史 K/V。
//!
//! 口径与既有算子基准一致：**ns/iter，越小越好**，另按每迭代扫到的 KV 元素数报吞吐。
//! 预热、采样与测量时长由 criterion 自身配置决定，本文件只声明一次迭代做什么；不写性能
//! 目标数字，也不与外部框架比倍数。代表历史长度取 128 / 512，头宽 64。
//!
//! 基线判定（`tool::regress`）只接 128 两条；512 两条在同一文件里出测量，不进基线表。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use op::attention;
use std::hint::black_box;

/// 头宽（head_dim）。
const D: usize = 64;

/// 按历史长度搭一块确定性输入：q 单行，k/v 各 `hist·d`；线性相位远离 0，构造期即为有限值。
fn inputs(hist: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let kv_len = hist * D;
    let q = (0..D).map(|i| -0.5 + i as f32 * 0.13).collect();
    let k = (0..kv_len).map(|i| 0.4 - i as f32 * 0.009).collect();
    let v = (0..kv_len).map(|i| -0.2 + i as f32 * 0.011).collect();
    let out = vec![0.0f32; D];
    (q, k, v, out)
}

fn bench_hist(c: &mut Criterion, hist: usize) {
    let (q, k, v, mut out) = inputs(hist);

    let mut group = c.benchmark_group("decode_attention_f32");
    group.throughput(Throughput::Elements((hist * D) as u64));

    let naive = format!("naive_hist{hist}x{D}");
    group.bench_function(&naive, |bench| {
        bench.iter(|| {
            attention::decode_forward(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(&mut out),
                1,
                hist,
                D,
            )
            .expect("q/k/v/out 几何为 1·d / hist·d")
        })
    });

    let flash = format!("flash_hist{hist}x{D}");
    group.bench_function(&flash, |bench| {
        bench.iter(|| {
            attention::decode_flash_forward(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(&mut out),
                1,
                hist,
                D,
            )
            .expect("q/k/v/out 几何为 1·d / hist·d")
        })
    });

    group.finish();
}

fn bench_hist_128(c: &mut Criterion) {
    bench_hist(c, 128);
}

fn bench_hist_512(c: &mut Criterion) {
    bench_hist(c, 512);
}

criterion_group!(benches, bench_hist_128, bench_hist_512);
criterion_main!(benches);
