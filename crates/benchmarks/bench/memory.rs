//! 内存搬运参照：给 roofline 一个**本机实测上界**，而不是厂商峰值规格。
//!
//! 三档工作集对应三级存储（16 KiB / 1 MiB / 64 MiB），每档读、写分开测：只流式读权重的
//! GEMV 可以跑过拷贝上限，把两个方向合成一个数，「这层卡在哪一侧」就再也判断不了
//! （`nn::Bounds` 的分账口径与此相同）。小档在迭代内多跑几趟，让计时噪声不盖过信号。
//!
//! 读用无符号整数异或归约而不是浮点求和：浮点加法默认不可重排，编译器会把归约串行化，测到的
//! 是加法延迟而不是搬运速率。搬运带宽与元素类型无关，整数读同样代表这一档的读上界。

use criterion::{Criterion, Throughput};
use std::hint::black_box;

/// 三档：`(标签, 规模后缀, 字数, 迭代内趟数)`，字数均按 `u64` 计。
const TIERS: [(&str, &str, usize, usize); 3] = [
    ("l1", "16kib", 2048, 64),
    ("l2", "1mib", 128 * 1024, 8),
    ("dram", "64mib", 8 * 1024 * 1024, 1),
];

/// 每档的填充样式：不同值避免编译器把三档的读写折叠成同一份常量。
const FILLS: [u64; 3] = [
    0x0101_0101_0101_0101,
    0x0202_0202_0202_0202,
    0x0303_0303_0303_0303,
];

/// 读一遍并做异或归约。
fn xor_read(words: &[u64]) -> u64 {
    let mut acc = 0u64;
    for word in words {
        acc ^= *word;
    }
    acc
}

/// 读方向的三档上界。
pub(crate) fn read(c: &mut Criterion) {
    let mut group = c.benchmark_group("mem_stream");
    for (index, (label, size, words, rounds)) in TIERS.into_iter().enumerate() {
        let data = vec![FILLS[index]; words];
        group.throughput(Throughput::Bytes(
            (rounds * words * size_of::<u64>()) as u64,
        ));
        group.bench_function(format!("{label}_read_{size}"), |bench| {
            bench.iter(|| {
                let mut acc = 0u64;
                for _ in 0..rounds {
                    acc ^= xor_read(black_box(&data));
                }
                black_box(acc)
            })
        });
    }
    group.finish();
}

/// 写方向的三档上界：`fill` 直接落到本机的写路径（大块时通常是流式存储指令）。
pub(crate) fn write(c: &mut Criterion) {
    let mut group = c.benchmark_group("mem_stream");
    for (index, (label, size, words, rounds)) in TIERS.into_iter().enumerate() {
        let mut data = vec![0u64; words];
        group.throughput(Throughput::Bytes(
            (rounds * words * size_of::<u64>()) as u64,
        ));
        group.bench_function(format!("{label}_write_{size}"), |bench| {
            bench.iter(|| {
                for _ in 0..rounds {
                    data.fill(black_box(FILLS[index]));
                }
                black_box(&data);
            })
        });
    }
    group.finish();
}
