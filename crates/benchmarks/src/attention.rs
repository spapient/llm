//! attention 基准：decode 单头、prefill 的 online 与 dense 两条路径。

use crate::fixtures::{Sample, checked_or_exit};
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput};
use nn::conv::attention;
use std::fmt;
use std::hint::black_box;
use tensor::View;

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

/// 一条 prefill 内核基准：三条输入是 `s×d` 二维视图，做一次前向。
fn one<E: fmt::Display>(
    sample: &mut Sample,
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    s: usize,
    run: impl Fn(View<'_, f32>, View<'_, f32>, View<'_, f32>, &mut [f32]) -> Result<(), E>,
) {
    let q = View::matrix(s, D, &sample.q).expect(SIZED);
    let k = View::matrix(s, D, &sample.k).expect(SIZED);
    let v = View::matrix(s, D, &sample.v).expect(SIZED);
    checked_or_exit(run(q, k, v, &mut sample.out), SIZED);
    bench(group, name, sample, s, run);
}

/// 四条切片上一次前向的计时循环：几何在调用前已校验，视图每次迭代重新装配。
fn bench<E: fmt::Display>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    sample: &mut Sample,
    s: usize,
    run: impl Fn(View<'_, f32>, View<'_, f32>, View<'_, f32>, &mut [f32]) -> Result<(), E>,
) {
    group.bench_function(name, |iter| {
        iter.iter(|| {
            let q = View::matrix(s, D, black_box(&sample.q)).expect(SIZED);
            let k = View::matrix(s, D, black_box(&sample.k)).expect(SIZED);
            let v = View::matrix(s, D, black_box(&sample.v)).expect(SIZED);
            let out = black_box(&mut *sample.out);
            black_box(run(q, k, v, out))
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
        DECODE_S,
        |q, k, v, out| attention::flash_forward(q, k, v, out),
    );
    group.finish();
}

/// prefill：online 与 dense 两条路径。
pub(crate) fn prefill(c: &mut Criterion) {
    let mut sample = Sample::new(PREFILL_S * D);
    let bytes = read_bytes(PREFILL_S);

    let mut online = c.benchmark_group("attention_prefill_f32");
    online.throughput(Throughput::Bytes(bytes));
    one(
        &mut sample,
        &mut online,
        "online_1024x64",
        PREFILL_S,
        |q, k, v, out| attention::flash_forward(q, k, v, out),
    );
    online.finish();

    let mut dense = c.benchmark_group("attention_dense_f32");
    dense.throughput(Throughput::Bytes(bytes));
    one(
        &mut sample,
        &mut dense,
        "naive_1024x64",
        PREFILL_S,
        |q, k, v, out| attention::forward(q, k, v, out),
    );
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
    decode_one(&mut group, &name, &q, &k, &v, &mut out, hist, attention::decode_forward);
    let name = format!("flash_hist{hist}x{D}");
    decode_one(
        &mut group,
        &name,
        &q,
        &k,
        &v,
        &mut out,
        hist,
        attention::decode_flash_forward,
    );
    group.finish();
}

/// decode 的一条内核：`q` 单行 `d`，`k`/`v` 按 `hist·d`，`out` 回到单行。
fn decode_one<E: fmt::Display>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    hist: usize,
    run: impl Fn(View<'_, f32>, View<'_, f32>, View<'_, f32>, &mut [f32]) -> Result<(), E>,
) {
    let qv = View::matrix(1, D, q).expect("decode 几何");
    let kv = View::matrix(hist, D, k).expect("decode 几何");
    let vv = View::matrix(hist, D, v).expect("decode 几何");
    checked_or_exit(run(qv, kv, vv, out), "decode 几何");
    group.bench_function(name, |iter| {
        iter.iter(|| {
            let qv = View::matrix(1, D, black_box(q)).expect("decode 几何");
            let kv = View::matrix(hist, D, black_box(k)).expect("decode 几何");
            let vv = View::matrix(hist, D, black_box(v)).expect("decode 几何");
            black_box(run(qv, kv, vv, black_box(out)))
        })
    });
}
