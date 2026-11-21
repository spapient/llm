//! attention 基准：decode 单头、prefill 的 online / dense 与各自 tiled 对照。
//!
//! 每条用例只在调用点声明"哪个名字配哪条内核"；一次迭代做什么、`black_box` 落在哪一层，
//! 由 [`one`] 统一给出——九条基准共用那一处，遍历次序之外没有第二份说法。预热、采样与
//! 测量时长由 criterion 配置决定。tile 与非 tile 两条路径逐位一致（对拍在 `op::attention`
//! 的测试里），本文件只比较遍历次序对墙钟的影响。

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput};
use op::attention;
use std::fmt;
use std::hint::black_box;

use crate::fixtures::Sample;

/// decode 的代表 KV 长度。
const DECODE_S: usize = 128;

/// prefill 的代表序列长度。
const PREFILL_S: usize = 1024;

/// 头宽（head_dim）。
const D: usize = 64;

/// decode 的历史长度两档：128 进基线，512 只出测量。
const HISTS: [usize; 2] = [128, 512];

/// 缓冲长度对不上时的失败文案（`s·d` 是四条切片的公共长度）。
const SIZED: &str = "缓冲尺寸与 s·d 一致";

/// q/k/v/out 各 `s·d` 时一次前向读入的字节数。
fn read_bytes(s: usize) -> u64 {
    (s * D * 3 * size_of::<f32>()) as u64
}

/// 一条内核基准：`run` 拿四条切片做一次前向。
///
/// `black_box` 落在四条切片入口，不落在样本上：内核只看得见切片，挡住这四层就挡住了全部
/// 读数。失败文案由调用点给出——`op` 的错误类型不跨 crate 命名，这里只判成败。
fn one<F, E: fmt::Debug>(
    (q, k, v, out): (&[f32], &[f32], &[f32], &mut [f32]),
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    run: F,
) where
    F: Fn(&[f32], &[f32], &[f32], &mut [f32]) -> Result<(), E>,
{
    group.bench_function(name, move |bench| {
        bench.iter(|| {
            let out = black_box(&mut out[..]);
            run(black_box(q), black_box(k), black_box(v), out).expect(SIZED)
        })
    });
}

/// decode 路径的签名：`q`、`k`、`v`、`out` 加 `(head, hist, d)`。
///
/// 参数生命周期由省略写法各自独立推出——共用一个会让 `out` 的可变借用被要求活到 `q/k/v` 那么久。
type Decode<E> = fn(&[f32], &[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<(), E>;

/// 把 fn item 坍成同一张函数指针类型，好让签名相同的两条路径放进同一张表。
///
/// 直接写 `[("naive", f), ("flash", g)]` 编译不过：两个 fn item 是不同的零尺寸类型，没有共同
/// 类型。显式标注那个七参数签名又会触发 `clippy::type_complexity`；这里的 `E` 由每条路径的真实
/// 返回类型推出，调用点因此不必命名错误类型（bench 不依赖 `diagnostic`）。
fn uniform<E: fmt::Debug>(run: Decode<E>) -> Decode<E> {
    run
}

/// `flash_forward`：decode 单头 128 行 KV 的单遍 online softmax。
pub(crate) fn flash_decode(c: &mut Criterion) {
    let mut sample = Sample::new(DECODE_S * D);
    let mut group = c.benchmark_group("attention_f32");
    group.throughput(Throughput::Elements((DECODE_S * D) as u64));
    let lanes = (
        &sample.q[..],
        &sample.k[..],
        &sample.v[..],
        &mut sample.out[..],
    );
    one(lanes, &mut group, "flash_decode_128x64", |q, k, v, out| {
        attention::flash_forward(q, k, v, out, DECODE_S, D)
    });
    group.finish();
}

/// prefill：online 与 dense 两条路径，各自与 tiled 版本对照。
pub(crate) fn prefill(c: &mut Criterion) {
    let mut sample = Sample::new(PREFILL_S * D);
    let bytes = read_bytes(PREFILL_S);

    let mut online = c.benchmark_group("attention_prefill_f32");
    online.throughput(Throughput::Bytes(bytes));
    one(
        lanes(&mut sample),
        &mut online,
        "online_1024x64",
        |q, k, v, out| attention::flash_forward(q, k, v, out, PREFILL_S, D),
    );
    one(
        lanes(&mut sample),
        &mut online,
        "tiled_1024x64",
        |q, k, v, out| attention::tiled::flash_tiled_forward(q, k, v, out, PREFILL_S, D),
    );
    online.finish();

    let mut dense = c.benchmark_group("attention_dense_f32");
    dense.throughput(Throughput::Bytes(bytes));
    one(
        lanes(&mut sample),
        &mut dense,
        "naive_1024x64",
        |q, k, v, out| attention::forward(q, k, v, out, PREFILL_S, D),
    );
    one(
        lanes(&mut sample),
        &mut dense,
        "tiled_1024x64",
        |q, k, v, out| attention::tiled::forward(q, k, v, out, PREFILL_S, D),
    );
    dense.finish();
}

/// 样本的四条切片：每次调用点重新借一次，`q`/`k`/`v` 只读、`out` 独占。
fn lanes(sample: &mut Sample) -> (&[f32], &[f32], &[f32], &mut [f32]) {
    (&sample.q, &sample.k, &sample.v, &mut sample.out)
}

/// 单查询行对变长历史的朴素与 Flash 两条 decode 内核。
pub(crate) fn decode(c: &mut Criterion) {
    for hist in HISTS {
        hist_pair(c, hist);
    }
}

/// 一个历史长度的一对内核：q 单行 `d`，k/v 按 `hist·d`，out 回到单行。
fn hist_pair(c: &mut Criterion, hist: usize) {
    let kv_len = hist * D;
    let q: Vec<f32> = (0..D).map(|index| -0.5 + index as f32 * 0.13).collect();
    let k: Vec<f32> = (0..kv_len)
        .map(|index| 0.4 - index as f32 * 0.009)
        .collect();
    let v: Vec<f32> = (0..kv_len)
        .map(|index| -0.2 + index as f32 * 0.011)
        .collect();
    let mut out = vec![0.0f32; D];

    let mut group = c.benchmark_group("decode_attention_f32");
    group.throughput(Throughput::Elements(kv_len as u64));
    let pair = [
        ("naive", uniform(attention::decode_forward)),
        ("flash", uniform(attention::decode_flash_forward)),
    ];
    for (label, run) in pair {
        let name = format!("{label}_hist{hist}x{D}");
        let lanes = (&q[..], &k[..], &v[..], &mut out[..]);
        one(lanes, &mut group, &name, |q, k, v, out| {
            run(q, k, v, out, 1, hist, D)
        });
    }
    group.finish();
}
