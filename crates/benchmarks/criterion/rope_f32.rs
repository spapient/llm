//! `rope_f32::forward` 的微基准：decode 一行 4096 个 f32 的二维位置旋转。
//!
//! 与 `gemv_f32` 同一口径：**ns/iter，越小越好**，另报元素吞吐（elems/s）。频率底数
//! `10000`、位置取 decode 约定的 7；预热、采样与测量时长由 criterion 配置决定，本文件只
//! 声明一次迭代做什么。
//!
//! 基线判定（`tool::regress`）随靶机首次实测后登记，未登记前本基准只出测量、不回退判定。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use op::rope_f32;
use std::hint::black_box;

/// decode 典型行宽（hidden 维度），必须为偶数以成对旋转。
const LEN: usize = 4096;

/// 位置下标：与算子样本约定一致。
const POS: usize = 7;

fn bench_forward(c: &mut Criterion) {
    let x: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.25 - 2.0).collect();
    let mut out = vec![0.0f32; LEN];

    let mut group = c.benchmark_group("rope_f32");
    group.throughput(Throughput::Elements(LEN as u64));
    group.bench_function("decode_1x4096", |bench| {
        bench.iter(|| {
            rope_f32::forward(POS, black_box(&x), black_box(&mut out)).expect("输入与输出等长")
        })
    });

    group.finish();
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
