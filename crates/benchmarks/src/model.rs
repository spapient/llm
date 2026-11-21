//! 整模型一步前向与 `train` 状态层基准。
//!
//! 前半：合成小栈（查表首层 + 投影）走 `Predict::logits`，单序列一步与四条序列同批各一档；
//! 内存只记可观测账目（页池 `in_use`），不虚构峰值字节数。
//!
//! 后半：学习率调度求值、检查点快照编解码、激活缓冲池复用一轮前向。三组都只压 CPU 上的状态
//! 计算，不碰设备；夹具失败一律走 [`checked`]，本文件没有 `unwrap` / `expect` / `panic!`。

use crate::fixtures::{PROMPT, checked, synthetic_model};
use autograd::{Sizes, Store, Tape, Var};
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
    let cache = checked(Cache::new(16, 4), "页池");
    let mut batch = Batch::new(checked(Pager::new(cache, 2), "行宽"));
    let sequences: Vec<Sequence> = (0..BATCH)
        .map(|id| checked(Sequence::new(id as u64, PROMPT, 2, 4, Stream::new(0)), "序列"))
        .collect();
    for sequence in &sequences {
        checked(batch.admit(sequence.clone()), "准入");
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
            black_box(checked(
                model.logits(black_box(&sequences[0]), black_box(&mut batch)),
                "一步前向",
            ))
        })
    });

    group.throughput(Throughput::Elements(BATCH as u64));
    group.bench_function("batch4_step", |bench| {
        bench.iter(|| black_box(steps(&mut model, &mut batch, &sequences)))
    });
    group.finish();
    black_box(pages);
}

/// 批内逐序列一步，返回产出的 logits 总数（防止编译器消掉）。
fn steps(model: &mut CpuModel, batch: &mut Batch, sequences: &[Sequence]) -> usize {
    let mut produced = 0usize;
    for sequence in sequences {
        produced += checked(model.logits(sequence, batch), "一步前向").len();
    }
    black_box(produced)
}

/// 三组训练状态基准：调度、快照、缓冲池。
pub(crate) fn train(c: &mut Criterion) {
    schedule(c);
    snapshot(c);
    buffers(c);
}

/// `Schedule::lr`：1000 步全曲线求和，避免单点求值被常量折叠。
fn schedule(c: &mut Criterion) {
    let curve = checked(
        Schedule::with(100, SCHEDULE_STEPS, Anneal::Cosine, 0.1),
        "调度合法",
    );
    let mut group = c.benchmark_group("train/schedule");
    group.throughput(Throughput::Elements(SCHEDULE_STEPS));
    group.bench_function("lr_warmup_to_cosine", |bench| {
        bench.iter(|| {
            let mut acc = 0.0f32;
            for step in 0..SCHEDULE_STEPS {
                acc += curve.lr(black_box(1.0), black_box(step));
            }
            black_box(acc)
        })
    });
    group.finish();
}

/// 快照条目表：64 条、每条 4096 维，只压头部编解码。
fn snapshot(c: &mut Criterion) {
    let entries = (0..SNAPSHOT_ENTRIES)
        .map(|index| {
            let shape = Shape::new([SNAPSHOT_DIM]);
            let offset = index * SNAPSHOT_DIM * size_of::<f32>();
            Entry::new(format!("blk.{index}.attn_q.weight"), DType::F32, shape).at(offset)
        })
        .collect();
    let facts = checked(Facts::new(7, 1, SNAPSHOT_ENTRIES, 32, 32), "事实");
    let header = checked(Header::new(entries, 0), "头部自洽");
    let table = Snapshot::new(facts, header);
    let bytes = checked(table.encode(), "编码");

    let mut group = c.benchmark_group("train/snapshot");
    group.throughput(Throughput::Elements(SNAPSHOT_ENTRIES as u64));
    group.bench_function("encode", |bench| {
        bench.iter(|| black_box(checked(table.encode(), "编码")))
    });
    group.bench_function("decode", |bench| {
        bench.iter(|| black_box(checked(Snapshot::decode(black_box(&bytes)), "解码")))
    });
    group.finish();
}

/// 一条 `gemv_i4` 链的形状表与常驻值；激活块在轮次间走空闲表复用。
struct Chain {
    tape: Tape,
    sizes: Sizes,
    x: Var,
    w: Var,
    scales: Var,
}

impl Chain {
    fn new() -> Self {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let w = tape.leaf();
        let scales = tape.leaf();
        let mut loss = x;
        for _ in 0..LAYERS {
            loss = checked(tape.record("gemv_i4", &[loss, w, scales]), "记录");
        }
        let sizes = chain_sizes(&tape, [x, w, scales]);
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
        checked(store.load(self.x, self.f32s(WIDTH)), "输入");
        checked(store.load(self.w, self.packed_i4()), "权重");
        checked(store.load(self.scales, self.f32s(WIDTH)), "缩放");
        store
    }
}

/// 形状表：输入与缩放按行宽、权重按 `[WIDTH, WIDTH]` 的 u4、每个节点输出按行宽。
fn chain_sizes(tape: &Tape, [x, w, scales]: [Var; 3]) -> Sizes {
    let mut sizes = Sizes::new(tape.vars());
    sizes.set(x, Shape::new([WIDTH]), DType::F32);
    sizes.set(w, Shape::new([WIDTH, WIDTH]), DType::U4);
    sizes.set(scales, Shape::new([WIDTH]), DType::F32);
    for node in tape.nodes() {
        sizes.set(node.output(), Shape::new([WIDTH]), DType::F32);
    }
    sizes
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
    group.bench_function("forward_round_pool_reuse", |bench| {
        bench.iter(|| black_box(round(&mut store, &nodes)))
    });
    group.finish();
}

/// 一轮：先按表算全部节点，再一次性释放，最后取存活块数。
fn round(store: &mut Buffers<Reference>, nodes: &[(String, Vec<Var>, Var)]) -> usize {
    for (op, inputs, output) in nodes {
        checked(store.compute(op, inputs, *output), "前向可算");
    }
    for (_, _, output) in nodes {
        store.free(*output);
    }
    black_box(store.live())
}
