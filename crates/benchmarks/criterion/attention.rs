//! attention 基准：decode 单头、prefill 的 online / dense 与各自 tiled 对照。
//!
//! 报 ns/iter（越小越好）与每迭代扫过的元素数或读入字节数；预热、采样与测量时长由 criterion
//! 配置决定。tile 与非 tile 两条路径逐位一致（对拍在 `op::attention` 的测试里），本文件只比较
//! 遍历次序对墙钟的影响。

use crate::attention_fixture::Sample;
use criterion::{Criterion, Throughput};
use op::attention;
use std::hint::black_box;

/// decode 的代表 KV 长度。
const DECODE_S: usize = 128;

/// prefill 的代表序列长度。
const PREFILL_S: usize = 1024;

/// 头宽（head_dim）。
const D: usize = 64;

/// q/k/v/out 各 `s·d` 时一次前向读入的字节数。
fn read_bytes(s: usize) -> u64 {
    (s * D * 3 * size_of::<f32>()) as u64
}

/// `flash_forward`：decode 单头 128 行 KV 的单遍 online softmax。
pub(crate) fn flash_decode(c: &mut Criterion) {
    let len = DECODE_S * D;
    let mut sample = Sample::new(len);

    let mut group = c.benchmark_group("attention_f32");
    group.throughput(Throughput::Elements(len as u64));
    group.bench_function("flash_decode_128x64", |bench| {
        bench.iter(|| {
            attention::flash_forward(
                black_box(&sample.q),
                black_box(&sample.k),
                black_box(&sample.v),
                black_box(&mut sample.out),
                DECODE_S,
                D,
            )
            .expect("q/k/v/out 等长且为 s·d")
        })
    });
    group.finish();
}

/// prefill：online 与 dense 两条路径，各自与 tiled 版本对照。
pub(crate) fn prefill(c: &mut Criterion) {
    let len = PREFILL_S * D;
    let mut sample = Sample::new(len);
    let bytes = read_bytes(PREFILL_S);

    let mut online = c.benchmark_group("attention_prefill_f32");
    online.throughput(Throughput::Bytes(bytes));
    online.bench_function("online_1024x64", |bench| {
        bench.iter(|| {
            attention::flash_forward(
                black_box(&sample.q),
                black_box(&sample.k),
                black_box(&sample.v),
                black_box(&mut sample.out),
                black_box(PREFILL_S),
                black_box(D),
            )
            .expect("缓冲尺寸与 s·d 一致")
        })
    });
    online.bench_function("tiled_1024x64", |bench| {
        bench.iter(|| {
            attention::tiled::flash_tiled_forward(
                black_box(&sample.q),
                black_box(&sample.k),
                black_box(&sample.v),
                black_box(&mut sample.out),
                black_box(PREFILL_S),
                black_box(D),
            )
            .expect("缓冲尺寸与 s·d 一致")
        })
    });
    online.finish();

    let mut dense = c.benchmark_group("attention_dense_f32");
    dense.throughput(Throughput::Bytes(bytes));
    dense.bench_function("naive_1024x64", |bench| {
        bench.iter(|| {
            attention::forward(
                black_box(&sample.q),
                black_box(&sample.k),
                black_box(&sample.v),
                black_box(&mut sample.out),
                black_box(PREFILL_S),
                black_box(D),
            )
            .expect("缓冲尺寸与 s·d 一致")
        })
    });
    dense.bench_function("tiled_1024x64", |bench| {
        bench.iter(|| {
            attention::tiled::forward(
                black_box(&sample.q),
                black_box(&sample.k),
                black_box(&sample.v),
                black_box(&mut sample.out),
                black_box(PREFILL_S),
                black_box(D),
            )
            .expect("缓冲尺寸与 s·d 一致")
        })
    });
    dense.finish();
}

/// 单查询行对变长历史的朴素与 Flash 两条 decode 内核：历史取 128（进基线）与 512（只出测量）。
pub(crate) fn decode(c: &mut Criterion) {
    for hist in [128, 512] {
        hist_pair(c, hist);
    }
}

/// 一个历史长度的一对内核：q 单行，k/v 各 `hist·d`。
fn hist_pair(c: &mut Criterion, hist: usize) {
    let kv_len = hist * D;
    let q: Vec<f32> = (0..D).map(|i| -0.5 + i as f32 * 0.13).collect();
    let k: Vec<f32> = (0..kv_len).map(|i| 0.4 - i as f32 * 0.009).collect();
    let v: Vec<f32> = (0..kv_len).map(|i| -0.2 + i as f32 * 0.011).collect();
    let mut out = vec![0.0f32; D];

    let mut group = c.benchmark_group("decode_attention_f32");
    group.throughput(Throughput::Elements(kv_len as u64));
    group.bench_function(format!("naive_hist{hist}x{D}"), |bench| {
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
    group.bench_function(format!("flash_hist{hist}x{D}"), |bench| {
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
