//! `Bytes` 基线分词器与合成 BPE 的微基准：编码、计数、解码、详细对齐与流式增量解码。
//!
//! 只产出实测耗时与吞吐；bench 本身不写性能目标数字，是否回退由 `tools::regress` 的基线判定负责。
//! 基线是"最坏情况 1 token/byte"的对照面：真词表落地后，同口径对比压缩率与逐字节成本。
//! BPE 词表是基准内合成的（256 个单字节 + 少量 ASCII 二元组合），只用于跑通合并路径，
//! 不代表任何发布模型的词表规模。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tokenizer::{Bpe, Bytes, DecoderStream, Merge, Tokenizer, VocabFamily, VocabSource};

/// 混合语料：ASCII 单词、CJK 与 emoji，循环填充到 4 KiB 的字符边界。
const TARGET_BYTES: usize = 4096;
const CHUNK: usize = 32;

fn sample() -> String {
    let mut out = String::new();
    for ch in "hello 世界 🚀 tokenizer ".chars().cycle() {
        if out.len() + ch.len_utf8() > TARGET_BYTES {
            break;
        }
        out.push(ch);
    }
    out
}

/// 基准内合成的 BPE：256 个单字节 + 三个 ASCII 二元组合；合并秩按表内顺序。
fn bpe_encoder() -> Bpe {
    let mut pieces: Vec<Vec<u8>> = (0..=255u8).map(|byte| vec![byte]).collect();
    pieces.extend([b"he".to_vec(), b"ll".to_vec(), b"  ".to_vec()]);
    let merges = vec![
        Merge {
            left: vec![b'h'],
            right: vec![b'e'],
        },
        Merge {
            left: vec![b'l'],
            right: vec![b'l'],
        },
        Merge {
            left: vec![b' '],
            right: vec![b' '],
        },
    ];
    let source = VocabSource {
        family: VocabFamily::Bpe,
        pieces,
        merges,
        ..VocabSource::default()
    };
    let vocab = tokenizer::Vocab::try_new(source).expect("基准词表构造");
    Bpe::new(vocab).expect("基准 BPE 实例")
}

fn bench_encode_bpe(c: &mut Criterion) {
    let text = sample();
    let encoder = bpe_encoder();
    let mut group = c.benchmark_group("tokenizer/encode");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("bpe_synth_4kib", |b| {
        b.iter(|| encoder.encode(black_box(&text)).expect("编码"))
    });
    group.finish();
}

fn bench_decode_bpe(c: &mut Criterion) {
    let text = sample();
    let encoder = bpe_encoder();
    let tokens = encoder.encode(&text).expect("预置编码");
    let mut group = c.benchmark_group("tokenizer/decode");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("bpe_synth_4kib", |b| {
        b.iter(|| encoder.decode(black_box(&tokens)).expect("解码"))
    });
    group.finish();
}

fn bench_encode(c: &mut Criterion) {
    let text = sample();
    let mut group = c.benchmark_group("tokenizer/encode");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("bytes_4kib", |b| {
        b.iter(|| Bytes.encode(black_box(&text)).expect("编码"))
    });
    group.finish();
}

fn bench_count(c: &mut Criterion) {
    let text = sample();
    let mut group = c.benchmark_group("tokenizer/count");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("bytes_4kib", |b| {
        b.iter(|| Bytes.count(black_box(&text)).expect("计数"))
    });
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let text = sample();
    let tokens = Bytes.encode(&text).expect("预置编码");
    let mut group = c.benchmark_group("tokenizer/decode");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("bytes_4kib", |b| {
        b.iter(|| Bytes.decode(black_box(&tokens)).expect("解码"))
    });
    group.finish();
}

fn bench_detailed(c: &mut Criterion) {
    let text = sample();
    let mut group = c.benchmark_group("tokenizer/encode_detailed");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("bytes_4kib", |b| {
        b.iter(|| Bytes.encode_detailed(black_box(&text)).expect("详细编码"))
    });
    group.finish();
}

fn bench_stream(c: &mut Criterion) {
    let text = sample();
    let tokens = Bytes.encode(&text).expect("预置编码");
    let bytes: Vec<u8> = tokens.into_iter().map(token_byte).collect();
    let mut group = c.benchmark_group("tokenizer/stream");
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("push_32b_chunks", |b| {
        b.iter(|| {
            let mut stream = DecoderStream::new();
            for chunk in bytes.chunks(CHUNK) {
                black_box(stream.push(black_box(chunk)).expect("增量解码"));
            }
            black_box(stream.finish().expect("收尾"));
        })
    });
    group.finish();
}

/// 把编号压回单字节：`Bytes` 词表内编号必为 `0..=255`。
fn token_byte(token: tokenizer::Token) -> u8 {
    token.id() as u8
}

criterion_group!(
    benches,
    bench_encode,
    bench_encode_bpe,
    bench_count,
    bench_decode,
    bench_decode_bpe,
    bench_detailed,
    bench_stream
);
criterion_main!(benches);
