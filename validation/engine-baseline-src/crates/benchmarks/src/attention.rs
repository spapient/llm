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

/// prefill 内核入口：三条只读视图做一次前向。
type Run3<E> = fn(View<'_, f32>, View<'_, f32>, View<'_, f32>, &mut [f32]) -> Result<(), E>;

/// 一条 prefill 基准：预检装配三条 `s×d` 视图，闭包内复用。
fn one<E: fmt::Display>(
    sample: &mut Sample,
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    s: usize,
    run: Run3<E>,
) {
    let qv = checked_or_exit(View::matrix(s, D, &sample.q), SIZED);
    let kv = checked_or_exit(View::matrix(s, D, &sample.k), SIZED);
    let vv = checked_or_exit(View::matrix(s, D, &sample.v), SIZED);
    checked_or_exit(
        run(qv.clone(), kv.clone(), vv.clone(), &mut sample.out),
        SIZED,
    );
    bench3(group, name, run, qv, kv, vv, &mut sample.out);
}

/// 三条视图上的计时循环：几何在预检阶段已校验，视图按迭代复用。
fn bench3<E: fmt::Display>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    run: Run3<E>,
    qv: View<'_, f32>,
    kv: View<'_, f32>,
    vv: View<'_, f32>,
    out: &mut [f32],
) {
    group.bench_function(name, |iter| {
        iter.iter(|| black_box(run(qv.clone(), kv.clone(), vv.clone(), black_box(out))))
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
        attention::flash_forward,
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
        attention::flash_forward,
    );
    online.finish();

    let mut dense = c.benchmark_group("attention_dense_f32");
    dense.throughput(Throughput::Bytes(bytes));
    one(
        &mut sample,
        &mut dense,
        "naive_1024x64",
        PREFILL_S,
        attention::forward,
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
    decode_one(
        &mut group,
        &name,
        &q,
        &k,
        &v,
        &mut out,
        attention::decode_forward,
    );
    let name = format!("flash_hist{hist}x{D}");
    decode_one(
        &mut group,
        &name,
        &q,
        &k,
        &v,
        &mut out,
        attention::decode_flash_forward,
    );
    group.finish();
}

/// decode 的一条内核：预检装配 `q` 单行、`k/v` 历史两行视图，闭包内复用。
fn decode_one<E: fmt::Display>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    run: Run3<E>,
) {
    let hist = k.len() / D;
    let qv = checked_or_exit(View::matrix(1, D, q), "decode 几何");
    let kv = checked_or_exit(View::matrix(hist, D, k), "decode 几何");
    let vv = checked_or_exit(View::matrix(hist, D, v), "decode 几何");
    checked_or_exit(run(qv.clone(), kv.clone(), vv.clone(), out), "decode 几何");
    group.bench_function(name, |iter| {
        iter.iter(|| black_box(run(qv.clone(), kv.clone(), vv.clone(), black_box(out))))
    });
}
