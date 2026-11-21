//! 整模型一步前向与 `train` 状态层基准。
//!
//! 前半：合成小栈（查表首层 + 投影）走 `Predict::logits`，单序列一步与四条序列同批各一档；
//! 内存只记可观测账目（页池 `in_use`），不虚构峰值字节数。前半不进基线表，沿用先例。
//!
//! 后半：学习率调度求值、检查点快照编解码、激活缓冲池复用前向轮；同样只出本机测量。

use crate::fixtures::{PROMPT, synthetic_model};
use autograd::{Sizes, Store, Tape};
use criterion::{Criterion, Throughput};
use engine::exec::Stream;
use engine::kv::{Cache, Pager};
use engine::{Batch, Sequence};
use serve::Predict;
use serve::model::CpuModel;
use std::hint::black_box;
use tensor::{DType, Shape};
use train::snapshot::{Facts, Snapshot};
use train::{Anneal, Buffers, Data, Reference, Schedule};
use weights::{Entry, Header};

/// 同批序列数（小批变体）。
const BATCH: usize = 4;

/// 调度曲线：100 步 warmup，1000 步余弦衰减到 0.1 谷底。
const SCHEDULE_STEPS: u64 = 1_000;

/// 快照条目数与每条形状：基准只压条目表编解码，不带权重 blob。
const SNAPSHOT_ENTRIES: usize = 64;
const SNAPSHOT_DIM: usize = 4_096;

/// 前向链宽度与层数，与 store 测试同口径的小模型。
const WIDTH: usize = 16;
const LAYERS: usize = 4;

/// 装配合成栈与一批序列；模型、批与序列在测量轮次间复用，前向内不重新分配权重。
fn fixture() -> (CpuModel, Batch, Vec<Sequence>) {
    let model = synthetic_model();

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

/// 单序列一步与批内逐序列一步；每步 token 数由 `Throughput` 声明。
pub(crate) fn decode(c: &mut Criterion) {
    let (mut model, mut batch, sequences) = fixture();
    let pages = batch.scheduler().cache().in_use();

    let mut group = c.benchmark_group("model_decode");
    group.throughput(Throughput::Elements(1));
    group.bench_function("single_step", |bench| {
        bench.iter(|| {
            black_box(
                model
                    .logits(black_box(&sequences[0]), black_box(&mut batch))
                    .expect("一步前向"),
            );
        })
    });

    group.throughput(Throughput::Elements(BATCH as u64));
    group.bench_function("batch4_step", |bench| {
        bench.iter(|| {
            let mut produced = 0usize;
            for sequence in &sequences {
                produced += model.logits(sequence, &mut batch).expect("一步前向").len();
            }
            black_box(produced);
        })
    });
    group.finish();
    black_box(pages);
}

/// 三组训练状态基准：调度、快照、缓冲池。
pub(crate) fn train(c: &mut Criterion) {
    schedule(c);
    snapshot(c);
    buffers(c);
}

/// `Schedule::lr`：1000 步全曲线求和，避免单点求值被常量折叠。
fn schedule(c: &mut Criterion) {
    let schedule = Schedule::with(100, SCHEDULE_STEPS, Anneal::Cosine, 0.1).expect("调度合法");
    let mut group = c.benchmark_group("train/schedule");
    group.throughput(Throughput::Elements(SCHEDULE_STEPS));
    group.bench_function("lr_warmup_to_cosine", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for step in 0..SCHEDULE_STEPS {
                acc += schedule.lr(black_box(1.0), black_box(step));
            }
            black_box(acc);
        })
    });
    group.finish();
}

/// 快照条目表：64 条、每条 4096 维，只压头部编解码。
fn snapshot(c: &mut Criterion) {
    let entries = (0..SNAPSHOT_ENTRIES)
        .map(|index| {
            Entry::new(
                format!("blk.{index}.attn_q.weight"),
                DType::F32,
                Shape::new([SNAPSHOT_DIM]),
            )
            .at(index * SNAPSHOT_DIM * 4)
        })
        .collect();
    let facts = Facts::new(7, 1, SNAPSHOT_ENTRIES, 32, 32).expect("事实");
    let snapshot = Snapshot::new(facts, Header::new(entries, 0).expect("头部自洽"));
    let bytes = snapshot.encode().expect("编码");

    let mut group = c.benchmark_group("train/snapshot");
    group.throughput(Throughput::Elements(SNAPSHOT_ENTRIES as u64));
    group.bench_function("encode", |b| {
        b.iter(|| black_box(snapshot.encode().expect("编码")))
    });
    group.bench_function("decode", |b| {
        b.iter(|| black_box(Snapshot::decode(black_box(&bytes)).expect("解码")))
    });
    group.finish();
}

/// 一条 `gemv_i4` 链的形状表与常驻值；激活块在轮次间走空闲表复用。
struct Chain {
    tape: Tape,
    sizes: Sizes,
    x: autograd::Var,
    w: autograd::Var,
    scales: autograd::Var,
}

impl Chain {
    fn new() -> Self {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let w = tape.leaf();
        let scales = tape.leaf();
        let mut loss = x;
        for _ in 0..LAYERS {
            loss = tape.record("gemv_i4", &[loss, w, scales]).expect("记录");
        }

        let mut sizes = Sizes::new(tape.vars());
        sizes.set(x, Shape::new([WIDTH]), DType::F32);
        sizes.set(w, Shape::new([WIDTH, WIDTH]), DType::U4);
        sizes.set(scales, Shape::new([WIDTH]), DType::F32);
        for node in tape.nodes() {
            sizes.set(node.output(), Shape::new([WIDTH]), DType::F32);
        }
        Self {
            tape,
            sizes,
            x,
            w,
            scales,
        }
    }

    fn f32s(&self, n: usize) -> Data {
        Data::F32((0..n).map(|i| ((i * 7) % 13) as f32 * 0.25 - 1.0).collect())
    }

    fn packed_i4(&self) -> Data {
        Data::U8(
            (0..WIDTH * (WIDTH / 2))
                .map(|i| ((i * 37) % 251) as u8)
                .collect(),
        )
    }

    fn store(&self) -> Buffers<Reference> {
        let mut store = Buffers::new(self.sizes.clone(), Reference);
        store.load(self.x, self.f32s(WIDTH)).expect("输入");
        store.load(self.w, self.packed_i4()).expect("权重");
        store.load(self.scales, self.f32s(WIDTH)).expect("缩放");
        store
    }
}

/// 缓冲池：一轮前向后释放全部中间块，下一轮复用同一批槽位。
fn buffers(c: &mut Criterion) {
    let chain = Chain::new();
    let mut store = chain.store();
    let nodes: Vec<_> = chain
        .tape
        .nodes()
        .iter()
        .map(|node| (node.op().to_string(), node.inputs().to_vec(), node.output()))
        .collect();

    let mut group = c.benchmark_group("train/buffers");
    group.throughput(Throughput::Elements(LAYERS as u64));
    group.bench_function("forward_round_pool_reuse", |b| {
        b.iter(|| {
            for (op, inputs, output) in &nodes {
                store.compute(op, inputs, *output).expect("前向可算");
            }
            for (_, _, output) in &nodes {
                store.free(*output);
            }
            black_box(store.live());
        })
    });
    group.finish();
}
