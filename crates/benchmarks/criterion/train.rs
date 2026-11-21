//! `train` 状态层的微基准：学习率调度求值、检查点快照编解码、激活缓冲池复用前向轮。
//!
//! 只产出本机实测耗时；是否回退由 `tools::regress` 的基线判定负责，bench 本身不写性能目标数字，
//! 也不与任何外部框架比倍数。预热、测量时长与采样沿用 criterion 自身配置。

use autograd::{Sizes, Store, Tape};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tensor::{DType, Shape};
use train::snapshot::{Facts, Snapshot};
use train::{Anneal, Buffers, Data, Reference, Schedule};
use weights::{Entry, Header};

/// 调度曲线：100 步 warmup，1000 步余弦衰减到 0.1 谷底。
const SCHEDULE_STEPS: u64 = 1_000;

/// 快照条目数与每条形状：基准只压条目表编解码，不带权重 blob。
const SNAPSHOT_ENTRIES: usize = 64;
const SNAPSHOT_DIM: usize = 4_096;

/// 前向链宽度与层数，与 store 测试同口径的小模型。
const WIDTH: usize = 16;
const LAYERS: usize = 4;

fn bench_schedule(c: &mut Criterion) {
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

fn snapshot_header() -> Header {
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
    Header::new(entries, 0).expect("头部自洽")
}

fn bench_snapshot(c: &mut Criterion) {
    let snapshot = Snapshot::new(
        Facts::new(7, 1, SNAPSHOT_ENTRIES, 32, 32).expect("事实"),
        snapshot_header(),
    );
    let bytes = snapshot.encode().expect("编码");

    let mut encode = c.benchmark_group("train/snapshot");
    encode.throughput(Throughput::Elements(SNAPSHOT_ENTRIES as u64));
    encode.bench_function("encode", |b| {
        b.iter(|| black_box(snapshot.encode().expect("编码")))
    });
    encode.finish();

    let mut decode = c.benchmark_group("train/snapshot");
    decode.throughput(Throughput::Elements(SNAPSHOT_ENTRIES as u64));
    decode.bench_function("decode", |b| {
        b.iter(|| black_box(Snapshot::decode(black_box(&bytes)).expect("解码")))
    });
    decode.finish();
}

/// 一条 `gemv_i4` 链的形状表与常驻值；激活块在轮次间走空闲表复用。
struct Fixture {
    tape: Tape,
    sizes: Sizes,
    x: autograd::Var,
    w: autograd::Var,
    scales: autograd::Var,
}

impl Fixture {
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

fn bench_buffers(c: &mut Criterion) {
    let fixture = Fixture::new();
    let mut store = fixture.store();
    let nodes: Vec<_> = fixture
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

criterion_group!(benches, bench_schedule, bench_snapshot, bench_buffers);
criterion_main!(benches);
