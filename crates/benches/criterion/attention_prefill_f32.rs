//! prefill 自注意力的微基准：大序列 1024、头宽 64，online 与分块 tile 两条路径对照。
//!
//! 两条路径结果逐位一致（对拍见 `op::attention::tests::tiled_flash_matches_online_bitwise`），
//! 本基准只比较遍历次序对墙钟的影响。ns/iter 越小越好，另报读入字节吞吐。
//! 基线判定（`tool::regress`）随靶机首次实测后登记，未登记前本基准只出测量、不回退判定。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use op::{attention, attention_tiled};
use std::hint::black_box;

/// 序列长 `s`（prefill 代表值）。
const S: usize = 1024;

/// 头宽 `d`。
const D: usize = 64;

fn bench(c: &mut Criterion) {
    let len = S * D;
    let q: Vec<f32> = (0..len).map(|i| -0.5 + i as f32 * 0.13).collect();
    let k: Vec<f32> = (0..len).map(|i| 0.4 - i as f32 * 0.09).collect();
    let v: Vec<f32> = (0..len).map(|i| -0.2 + i as f32 * 0.11).collect();
    let mut out = vec![0.0f32; len];

    let read_bytes = (len * 3 * size_of::<f32>()) as u64;
    let mut group = c.benchmark_group("attention_prefill_f32");
    group.throughput(Throughput::Bytes(read_bytes));
    group.bench_function("online_1024x64", |bench| {
        bench.iter(|| {
            attention::flash_forward(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(&mut out),
                black_box(S),
                black_box(D),
            )
            .expect("缓冲尺寸与 s·d 一致")
        })
    });
    group.bench_function("tiled_1024x64", |bench| {
        bench.iter(|| {
            attention_tiled::flash_tiled_forward(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(&mut out),
                black_box(S),
                black_box(D),
            )
            .expect("缓冲尺寸与 s·d 一致")
        })
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
