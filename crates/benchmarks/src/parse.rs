//! 头部解析基准：safetensors 头部 JSON 与整档解析，是权重加载的关键路径。

use crate::fixtures::checked_or_exit;
use criterion::{Criterion, Throughput};
use std::hint::black_box;
use weights::json;
use weights::safetensors::Archive;

/// 条目数。
const N_ENTRIES: usize = 8;

/// 每条目 F32 元素数。
const ELEMS: usize = 256;

/// 每条目字节数。
const BYTES: usize = ELEMS * 4;

/// 一段有代表性的 safetensors 头部 JSON（N 个 F32 条目）。
fn header_json() -> String {
    let mut entries = Vec::with_capacity(N_ENTRIES);
    for idx in 0..N_ENTRIES {
        let start = idx * BYTES;
        let end = start + BYTES;
        entries.push(format!(
            "\"w.{idx}\": {{\"dtype\": \"F32\", \"shape\": [{ELEMS}], \
             \"data_offsets\": [{start}, {end}]}}"
        ));
    }
    format!("{{{}}}", entries.join(","))
}

/// 完整 safetensors 字节：8 字节长度前缀 + JSON + 递增数据段。
fn safetensors_bytes() -> Vec<u8> {
    let json = header_json();
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(json.as_bytes());
    file.extend((0..N_ENTRIES * BYTES).map(|i| (i % 251) as u8));
    file
}

pub(crate) fn parse(c: &mut Criterion) {
    json_header(c);
    safetensors_archive(c);
}

fn json_header(c: &mut Criterion) {
    let json = header_json();
    let mut group = c.benchmark_group("parse/json");
    group.throughput(Throughput::Bytes(json.len() as u64));
    group.bench_function("header", |bench| {
        bench.iter(|| black_box(checked_or_exit(json::parse(black_box(&json)), "JSON 头部")))
    });
    group.finish();
}

fn safetensors_archive(c: &mut Criterion) {
    let bytes = safetensors_bytes();
    let mut group = c.benchmark_group("parse/safetensors");
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("archive", |bench| {
        bench.iter(|| black_box(checked_or_exit(Archive::parse(black_box(&bytes)), "归档")))
    });
    group.finish();
}
