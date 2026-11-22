//! 头部解析基准：safetensors 头部 JSON 与整档解析，是权重加载的关键路径。

use crate::fixtures::checked_or_exit;
use criterion::{Criterion, Throughput};
use std::hint::black_box;
use weights::gguf::Gguf;
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
    gguf_floats(c);
}

/// 拼一个最小 GGUF：单张量 `w`（Q4_K 或 Q6_K，256 元素）+ `general.alignment`。
fn gguf(kind: u32, block: &[u8]) -> Vec<u8> {
    let mut head = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&1u64.to_le_bytes());
    head.extend_from_slice(&1u64.to_le_bytes());
    let key = "general.alignment";
    head.extend_from_slice(&(key.len() as u64).to_le_bytes());
    head.extend_from_slice(key.as_bytes());
    head.extend_from_slice(&4u32.to_le_bytes());
    head.extend_from_slice(&32u32.to_le_bytes());
    head.extend_from_slice(&1u64.to_le_bytes());
    head.extend_from_slice(b"w");
    head.extend_from_slice(&1u32.to_le_bytes());
    head.extend_from_slice(&256u64.to_le_bytes());
    head.extend_from_slice(&kind.to_le_bytes());
    head.extend_from_slice(&0u64.to_le_bytes());
    let fill = (32 - head.len() % 32) % 32;
    head.extend(std::iter::repeat_n(0u8, fill));
    head.extend_from_slice(block);
    head
}

/// Q4_K 块：d=1、dmin=0、scales 全 1、量化点全 0x21。
fn q4k_block() -> Vec<u8> {
    let mut block = 0x3C00u16.to_le_bytes().to_vec();
    block.extend(0u16.to_le_bytes());
    block.extend([1u8; 12]);
    block.extend([0x21u8; 128]);
    block
}

/// Q6_K 块：量化点 0x21、高位 0、scales 全 1、d=1。
fn q6k_block() -> Vec<u8> {
    let mut block = vec![0x21u8; 128];
    block.extend([0u8; 64]);
    block.extend([1u8; 16]);
    block.extend(0x3C00u16.to_le_bytes());
    block
}

fn gguf_floats(c: &mut Criterion) {
    let q4k_bytes = gguf(12, &q4k_block());
    let q6k_bytes = gguf(14, &q6k_block());
    let q4k = checked_or_exit(Gguf::parse(&q4k_bytes), "GGUF 解析");
    let q6k = checked_or_exit(Gguf::parse(&q6k_bytes), "GGUF 解析");
    let mut group = c.benchmark_group("parse/gguf");
    group.throughput(Throughput::Elements(256));
    group.bench_function("q4_k_floats", |bench| {
        bench.iter(|| black_box(checked_or_exit(q4k.floats(black_box("w")), "反量化")))
    });
    group.bench_function("q6_k_floats", |bench| {
        bench.iter(|| black_box(checked_or_exit(q6k.floats(black_box("w")), "反量化")))
    });
    group.finish();
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
