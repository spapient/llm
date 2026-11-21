//! 文本与字节编码基准：分词器（Bytes 基线与合成 BPE）与 `wire` 字节码闭集。
//!
//! 分词器共用一份混合语料（ASCII / CJK / emoji，循环填到 4 KiB 的字符边界）与一个合成 BPE，
//! 五个分组只声明各自测的动作。BPE 词表是基准内合成的（256 个单字节 + 三个 ASCII 二元组合），
//! 只用于跑通合并路径，不代表任何发布模型的词表规模。
//!
//! 只出实测耗时与吞吐；是否回退由 `tests/regress` 的基线判定负责，本文件不写性能目标数字。

use criterion::{Criterion, Throughput};
use std::hint::black_box;
use tokenizer::{Bpe, Bytes, DecoderStream, Merge, Tokenizer, VocabFamily, VocabSource};
use wire::crc::update;
use wire::leb::{Cursor, write};
use wire::sink::{Count, Sealed, Sink};

/// 语料目标字节数。
const TARGET_BYTES: usize = 4096;

/// 流式解码的分块字节数。
const CHUNK: usize = 32;

/// 混合语料，按字符边界截到 [`TARGET_BYTES`]。
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

/// 分词器五组：编码（基线 / BPE）、计数、解码（基线 / BPE）、详细对齐、流式增量。
pub(crate) fn tokenizer(c: &mut Criterion) {
    let text = sample();
    let encoder = bpe_encoder();
    let bpe_tokens = encoder.encode(&text).expect("预置编码");
    let tokens = Bytes.encode(&text).expect("预置编码");
    let bytes: Vec<u8> = tokens.iter().copied().map(token_byte).collect();

    let mut encode = c.benchmark_group("tokenizer/encode");
    encode.throughput(Throughput::Bytes(text.len() as u64));
    encode.bench_function("bpe_synth_4kib", |b| {
        b.iter(|| encoder.encode(black_box(&text)).expect("编码"))
    });
    encode.bench_function("bytes_4kib", |b| {
        b.iter(|| Bytes.encode(black_box(&text)).expect("编码"))
    });
    encode.finish();

    let mut count = c.benchmark_group("tokenizer/count");
    count.throughput(Throughput::Bytes(text.len() as u64));
    count.bench_function("bytes_4kib", |b| {
        b.iter(|| Bytes.count(black_box(&text)).expect("计数"))
    });
    count.finish();

    let mut decode = c.benchmark_group("tokenizer/decode");
    decode.throughput(Throughput::Bytes(text.len() as u64));
    decode.bench_function("bpe_synth_4kib", |b| {
        b.iter(|| encoder.decode(black_box(&bpe_tokens)).expect("解码"))
    });
    decode.bench_function("bytes_4kib", |b| {
        b.iter(|| Bytes.decode(black_box(&tokens)).expect("解码"))
    });
    decode.finish();

    let mut detailed = c.benchmark_group("tokenizer/encode_detailed");
    detailed.throughput(Throughput::Bytes(text.len() as u64));
    detailed.bench_function("bytes_4kib", |b| {
        b.iter(|| Bytes.encode_detailed(black_box(&text)).expect("详细编码"))
    });
    detailed.finish();

    let mut stream = c.benchmark_group("tokenizer/stream");
    stream.throughput(Throughput::Bytes(bytes.len() as u64));
    stream.bench_function("push_32b_chunks", |b| {
        b.iter(|| {
            let mut decoder = DecoderStream::new();
            for chunk in bytes.chunks(CHUNK) {
                black_box(decoder.push(black_box(chunk)).expect("增量解码"));
            }
            black_box(decoder.finish().expect("收尾"));
        })
    });
    stream.finish();
}

/// 把编号压回单字节：`Bytes` 词表内编号必为 `0..=255`。
fn token_byte(token: tokenizer::Token) -> u8 {
    token.id() as u8
}

/// LEB128 往返用的取值集合：覆盖 1..10 字节各档长度。
const VALUES: [u64; 8] = [
    0,
    127,
    128,
    16_383,
    16_384,
    u32::MAX as u64,
    1 << 40,
    u64::MAX,
];

/// CRC 输入字节数与密封 Sink 输入字节数。
const CRC_BYTES: usize = 4096;
const SEALED_BYTES: usize = 1024;

/// `wire` 三组：CRC 更新、LEB128 编解码、密封 Sink。
pub(crate) fn wire(c: &mut Criterion) {
    let data: Vec<u8> = (0..=255).cycle().take(CRC_BYTES).collect();
    let mut group = c.benchmark_group("wire/crc");
    group.throughput(Throughput::Bytes(CRC_BYTES as u64));
    group.bench_function("update_4kib", |b| {
        b.iter(|| {
            let mut state = !0u32;
            for chunk in data.chunks(64) {
                state = update(state, black_box(chunk));
            }
            black_box(state);
        })
    });
    group.finish();

    let mut encode = c.benchmark_group("wire/leb");
    encode.throughput(Throughput::Elements(VALUES.len() as u64));
    encode.bench_function("encode_values", |b| {
        b.iter(|| {
            let mut count = Count(0);
            for value in VALUES {
                write(&mut count, black_box(value));
            }
            black_box(count.0);
        })
    });
    encode.finish();

    let mut buffer = Vec::new();
    for value in VALUES {
        write(&mut buffer, value);
    }
    let mut decode = c.benchmark_group("wire/leb");
    decode.throughput(Throughput::Elements(VALUES.len() as u64));
    decode.bench_function("decode_values", |b| {
        b.iter(|| {
            let mut cursor = Cursor::new(black_box(&buffer));
            for _ in VALUES {
                black_box(cursor.read::<u64>().expect("规范编码"));
            }
        })
    });
    decode.finish();

    let sealed_data: Vec<u8> = (0..=255).cycle().take(SEALED_BYTES).collect();
    let mut sealed = c.benchmark_group("wire/sink");
    sealed.throughput(Throughput::Bytes(SEALED_BYTES as u64));
    sealed.bench_function("sealed_1kib", |b| {
        b.iter(|| {
            let mut out = Vec::new();
            let mut sink = Sealed::new(&mut out);
            for chunk in sealed_data.chunks(32) {
                sink.put(black_box(chunk));
            }
            black_box(sink.checksum());
            black_box(&out);
        })
    });
    sealed.finish();
}
