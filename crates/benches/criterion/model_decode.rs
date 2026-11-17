//! 整模型一步前向基准：合成小栈（查表首层 + 投影）走 [`Predict::logits`] 一步 decode。
//!
//! 两个变体：单序列一步、四条序列同批逐序列走一步。口径只出 ns/iter 与每步 token 吞吐
//! （tokens/s 由 criterion 按元素吞吐换算）；内存只记可观测账目——占用页数（页池 `in_use`），
//! 不虚构峰值字节数，也不统计执行器内部绑定缓冲（那一侧没有公开读数）。本基准只出测量，
//! 不进 `tool::regress` 基线表，沿用 `train.rs` 先例。
//!
//! 权重由确定性 PRNG 生成有限值；预热、采样与测量时长由 criterion 自身配置决定。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use engine::exec::Stream;
use engine::kv::{Cache, Pager};
use engine::{Batch, Sequence};
use nn::{Embedding, Linear, Stack, WeightFormat};
use serve::Predict;
use serve::model::{CpuModel, Params};
use std::hint::black_box;
use testkit::Rng;

/// 词表大小。
const VOCAB: usize = 16;
/// 嵌入与隐藏宽度。
const DIM: usize = 8;
/// 投影输出宽度（末层 logits 宽度）。
const OUT: usize = 8;
/// 同批序列数（小批变体）。
const BATCH: usize = 4;
/// 提示长度：一步 decode，序列只带一个 token。
const PROMPT: &[u32] = &[3];

/// 确定性有限权重：±[0.5, 2]，远离 0，避免平方类差分为 0。
fn finite_values(n: usize, seed: u32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.value() as f32).collect()
}

/// 装配合成栈与一批序列；模型、批与序列在测量轮次间复用，前向内不重新分配权重。
fn fixture() -> (CpuModel, Batch, Vec<Sequence>) {
    let table = finite_values(VOCAB * DIM, 7);
    let projection = finite_values(OUT * DIM, 11);

    let stack = Stack::of(vec![
        Box::new(Embedding::new("tok_emb", VOCAB, DIM, 1).expect("层")),
        Box::new(Linear::with_format("proj", WeightFormat::F32, DIM, OUT).expect("层")),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("tok_emb.weight", table);
    params.insert("proj.weight", projection);
    let model = CpuModel::new(&stack, &params).expect("装配");

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
