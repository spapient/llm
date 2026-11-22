//! attention 基准：decode 单头、prefill 的 online / dense 与各自 tiled 对照。

use crate::fixtures::{Sample, checked};
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput};
use op::attention;
use std::fmt;
use std::hint::black_box;

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

/// 一条内核基准：`run` 拿四条等长切片做一次前向。
fn one<E: fmt::Display>(
    sample: &mut Sample,
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    run: impl Fn(&[f32], &[f32], &[f32], &mut [f32]) -> Result<(), E>,
) {
    checked(run(&sample.q, &sample.k, &sample.v, &mut sample.out), SIZED);
    bench(
        group,
        name,
        (&sample.q, &sample.k, &sample.v, &mut sample.out),
        run,
    );
}

/// decode 的一条内核：`q` 单行 `d`，`k`/`v` 按 `hist·d`，`out` 回到单行。
fn decode_one<E: fmt::Display>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    (q, k, v, out): (&[f32], &[f32], &[f32], &mut [f32]),
    hist: usize,
    run: impl Fn(&[f32], &[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<(), E>,
) {
    checked(run(q, k, v, out, 1, hist, D), "decode 几何");
    bench(group, name, (q, k, v, out), |q, k, v, out| {
        run(q, k, v, out, 1, hist, D)
    });
}

/// 四条切片上一次前向的计时循环：`run` 只认切片，几何在调用前已校验。
fn bench<E: fmt::Display>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    (q, k, v, out): (&[f32], &[f32], &[f32], &mut [f32]),
    run: impl Fn(&[f32], &[f32], &[f32], &mut [f32]) -> Result<(), E>,
) {
    group.bench_function(name, |iter| {
        iter.iter(|| {
            let out = black_box(&mut *out);
            black_box(run(black_box(q), black_box(k), black_box(v), out))
        })
    });
}

/// `flash_forward`：decode 单头 128 行 KV 的单遍 online softmax。
pub(crate) fn flash_decode(c: &mut Criterion) {
    let mut sample = Sample::new(DECODE_S * D);
    let mut group = c.benchmark_group("attention_f32");
    group.throughput(Throughput::Elements((DECODE_S * D) as u64));
    one(
        &mut sample,
        &mut group,
        "flash_decode_128x64",
        |q, k, v, out| attention::flash_forward(q, k, v, out, DECODE_S, D),
    );
    group.finish();
}

/// prefill：online 与 dense 两条路径，各自与 tiled 版本对照。
pub(crate) fn prefill(c: &mut Criterion) {
    let mut sample = Sample::new(PREFILL_S * D);
    let bytes = read_bytes(PREFILL_S);

    let mut online = c.benchmark_group("attention_prefill_f32");
    online.throughput(Throughput::Bytes(bytes));
    one(
        &mut sample,
        &mut online,
        "online_1024x64",
        |q, k, v, out| attention::flash_forward(q, k, v, out, PREFILL_S, D),
    );
    one(&mut sample, &mut online, "tiled_1024x64", |q, k, v, out| {
        attention::tiled::flash_tiled_forward(q, k, v, out, PREFILL_S, D)
    });
    online.finish();

    let mut dense = c.benchmark_group("attention_dense_f32");
    dense.throughput(Throughput::Bytes(bytes));
    one(&mut sample, &mut dense, "naive_1024x64", |q, k, v, out| {
        attention::forward(q, k, v, out, PREFILL_S, D)
    });
    one(&mut sample, &mut dense, "tiled_1024x64", |q, k, v, out| {
        attention::tiled::forward(q, k, v, out, PREFILL_S, D)
    });
    dense.finish();
}

/// 单查询行对变长历史的朴素与 Flash 两条 decode 内核。
pub(crate) fn decode(c: &mut Criterion) {
    for hist in HISTS {
        hist_pair(c, hist);
    }
}

/// 一个历史长度的一对内核。
fn hist_pair(c: &mut Criterion, hist: usize) {
    let kv_len = hist * D;
    let q: Vec<f32> = (0..D).map(|idx| -0.5 + idx as f32 * 0.13).collect();
    let k: Vec<f32> = (0..kv_len).map(|idx| 0.4 - idx as f32 * 0.009).collect();
    let v: Vec<f32> = (0..kv_len).map(|idx| -0.2 + idx as f32 * 0.011).collect();
    let mut out = vec![0.0f32; D];

    let mut group = c.benchmark_group("decode_attention_f32");
    group.throughput(Throughput::Elements(kv_len as u64));
    let name = format!("naive_hist{hist}x{D}");
    let lanes: (&[f32], &[f32], &[f32], &mut [f32]) = (&q, &k, &v, &mut out);
    decode_one(&mut group, &name, lanes, hist, attention::decode_forward);
    let name = format!("flash_hist{hist}x{D}");
    let lanes: (&[f32], &[f32], &[f32], &mut [f32]) = (&q, &k, &v, &mut out);
    decode_one(
        &mut group,
        &name,
        lanes,
        hist,
        attention::decode_flash_forward,
    );
    group.finish();
}
