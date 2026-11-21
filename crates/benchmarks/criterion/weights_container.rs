//! `weights` 自有 `.u` 容器 v2 的微基准：解析、Merkle 根/证明/校验、按名查找与写出。
//!
//! 只产出实测耗时与吞吐；是否回退由 `tools::regress` 的基线判定负责，bench 本身不写性能目标数字。
//! 预热、测量时长、采样次数沿用 criterion 自身配置。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use weights::{Archive, Segment, Wire, Writer};

const N_SEGS: usize = 64;
const ELEMS: usize = 256;
const BYTES_PER_SEG: usize = ELEMS * 4;

fn build_writer() -> Writer {
    let mut writer = Writer::new();
    for index in 0..N_SEGS {
        let data = vec![(index % 256) as u8; BYTES_PER_SEG];
        let segment = Segment::new(
            format!("seg_{index}"),
            Wire::F32,
            vec![ELEMS as u64],
            None,
            data,
        )
        .expect("段自洽");
        writer.push(segment).expect("追加");
    }
    writer
}

fn bench_parse(c: &mut Criterion) {
    let bytes = build_writer().finish(6).expect("字节");
    let mut group = c.benchmark_group("weights/archive");
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("parse", |b| {
        b.iter(|| Archive::parse(black_box(&bytes)).expect("解析"))
    });
    group.finish();
}

fn bench_root_prove_verify(c: &mut Criterion) {
    let bytes = build_writer().finish(6).expect("字节");
    let archive = Archive::parse(&bytes).expect("解析");
    let root = archive.merkle_root().expect("根");
    let proof = archive.prove(0).expect("证明");

    let mut group = c.benchmark_group("weights/archive");
    group.bench_function("merkle_root", |b| {
        b.iter(|| archive.merkle_root().expect("根"))
    });
    group.bench_function("prove", |b| {
        b.iter(|| archive.prove(black_box(0)).expect("证明"))
    });
    group.bench_function("verify", |b| {
        b.iter(|| archive.verify(0, &proof, &root).expect("校验"))
    });
    group.finish();
}

fn bench_get(c: &mut Criterion) {
    let bytes = build_writer().finish(6).expect("字节");
    let archive = Archive::parse(&bytes).expect("解析");
    let mut group = c.benchmark_group("weights/archive");
    group.bench_function("get_by_name", |b| {
        b.iter(|| archive.get(black_box("seg_32")).expect("段存在"))
    });
    group.finish();
}

fn bench_finish(c: &mut Criterion) {
    let writer = build_writer();
    let mut group = c.benchmark_group("weights/writer");
    group.throughput(Throughput::Bytes((N_SEGS * BYTES_PER_SEG) as u64));
    group.bench_function("finish", |b| b.iter(|| writer.finish(6).expect("字节")));
    group.finish();
}

criterion_group!(
    benches,
    bench_parse,
    bench_root_prove_verify,
    bench_get,
    bench_finish
);
criterion_main!(benches);
