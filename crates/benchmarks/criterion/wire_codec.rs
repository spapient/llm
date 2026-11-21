//! `wire` 字节码闭集的微基准：CRC 更新、LEB128 编码/解码、密封 Sink。
//!
//! 只产出实测耗时与吞吐；是否回退由 `tools::regress` 的基线判定负责，bench 本身不写性能目标数字。
//! 预热、测量时长、采样次数沿用 criterion 自身配置；吞吐口径用 [`Throughput`] 声明，报告直接出 bytes/s 或 elems/s。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use wire::crc::update;
use wire::leb::{Cursor, write};
use wire::sink::{Count, Sealed, Sink};

/// 编解码往返用的取值集合：覆盖 1..10 字节各档 LEB128 长度。
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

const CRC_BYTES: usize = 4096;
const SEALED_BYTES: usize = 1024;

fn bench_crc(c: &mut Criterion) {
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
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("wire/leb");
    group.throughput(Throughput::Elements(VALUES.len() as u64));
    group.bench_function("encode_values", |b| {
        b.iter(|| {
            let mut count = Count(0);
            for value in VALUES {
                write(&mut count, black_box(value));
            }
            black_box(count.0);
        })
    });
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut buffer = Vec::new();
    for value in VALUES {
        write(&mut buffer, value);
    }
    let mut group = c.benchmark_group("wire/leb");
    group.throughput(Throughput::Elements(VALUES.len() as u64));
    group.bench_function("decode_values", |b| {
        b.iter(|| {
            let mut cursor = Cursor::new(black_box(&buffer));
            for _ in VALUES {
                black_box(cursor.read::<u64>().expect("规范编码"));
            }
        })
    });
    group.finish();
}

fn bench_sealed(c: &mut Criterion) {
    let data: Vec<u8> = (0..=255).cycle().take(SEALED_BYTES).collect();
    let mut group = c.benchmark_group("wire/sink");
    group.throughput(Throughput::Bytes(SEALED_BYTES as u64));
    group.bench_function("sealed_1kib", |b| {
        b.iter(|| {
            let mut out = Vec::new();
            let mut sealed = Sealed::new(&mut out);
            for chunk in data.chunks(32) {
                sealed.put(black_box(chunk));
            }
            black_box(sealed.checksum());
            black_box(&out);
        })
    });
    group.finish();
}

criterion_group!(benches, bench_crc, bench_encode, bench_decode, bench_sealed);
criterion_main!(benches);
