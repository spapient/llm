//! 整模型一步前向基准：合成小栈（查表首层 + 投影）走 [`Predict::logits`] 一步 decode。
//!
//! 两个变体：单序列一步、四条序列同批逐序列走一步。口径只出 ns/iter 与每步 token 吞吐
//! （tokens/s 由 criterion 按元素吞吐换算）；内存只记可观测账目——占用页数（页池 `in_use`），
//! 不虚构峰值字节数，也不统计执行器内部绑定缓冲（那一侧没有公开读数）。本基准只出测量，
//! 不进 `tools::regress` 基线表，沿用 `train.rs` 先例。
//!
//! 测量结果归档为 `tools::bench::BenchRecord`（`EntryKind::Model`）：台账与记录构造在
//! `tests/regress.rs` 的 `DECODE_SCENARIOS`；记录只承载本机一次测量与身份，tokens/s 与页数
//! 由 Report 侧按账目与实测 ns 换算 / 登记，不进记录。
//!
//! 权重由确定性 PRNG 生成有限值；预热、采样与测量时长由 criterion 自身配置决定。

#[path = "../support/model.rs"]
mod support;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use engine::exec::Stream;
use engine::kv::{Cache, Pager};
use engine::{Batch, Sequence};
use serve::Predict;
use serve::model::CpuModel;
use std::hint::black_box;
use support::PROMPT;

/// 同批序列数（小批变体）。
const BATCH: usize = 4;

/// 装配合成栈与一批序列；模型、批与序列在测量轮次间复用，前向内不重新分配权重。
fn fixture() -> (CpuModel, Batch, Vec<Sequence>) {
    let model = support::synthetic_model();

    let pager = Pager::new(Cache::new(16, 4).expect("页池"), 2).expect("行宽");
    let mut batch = Batch::new(pager);
    let sequences: Vec<Sequence> = (0..BATCH)
        .map(|id| Sequence::new(id as u64, PROMPT, 2, 4, Stream::new(0)).expect("序列"))
        .collect();
    for sequence in &sequences {
        batch.admit(sequence.clone()).expect("准入");
    }
    (model, batch, sequences)
}

fn bench_single(c: &mut Criterion) {
    let (mut model, mut batch, sequences) = fixture();
    let sequence = sequences[0].clone();
    let pages = batch.scheduler().cache().in_use();

    let mut group = c.benchmark_group("model_decode");
    group.throughput(Throughput::Elements(1));
    group.bench_function("single_step", |bench| {
        bench.iter(|| {
            let logits = model
                .logits(black_box(&sequence), black_box(&mut batch))
                .expect("一步前向");
            black_box(logits);
        })
    });
    group.finish();
    black_box(pages);
}

fn bench_batch(c: &mut Criterion) {
    let (mut model, mut batch, sequences) = fixture();
    let pages = batch.scheduler().cache().in_use();

    let mut group = c.benchmark_group("model_decode");
    group.throughput(Throughput::Elements(BATCH as u64));
    group.bench_function("batch4_step", |bench| {
        bench.iter(|| {
            let mut produced = 0usize;
            for sequence in &sequences {
                let logits = model.logits(sequence, &mut batch).expect("一步前向");
                produced += logits.len();
            }
            black_box(produced);
        })
    });
    group.finish();
    black_box(pages);
}

criterion_group!(benches, bench_single, bench_batch);
criterion_main!(benches);
