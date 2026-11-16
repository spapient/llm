//! 内存搬运参照：给 roofline 一个**本机实测上界**，而不是厂商峰值规格。
//!
//! 为什么不用厂商峰值：`docs/lld.md` 的 E6 把「仅有未验证的峰值规格」判为不可计算（该行 OPEN）。
//! roofline 的分母必须是同机器、同工作集规模下测出来的搬运速率，否则达成率只是换了张纸的估算。
//!
//! 三档工作集对应三级存储，而不是靠猜硬件缓存容量：
//!
//! - `l1`：16 KiB，与四条 decode 行宽算子基准（4096 个 f32）同规模；
//! - `l2`：1 MiB，覆盖 `gemm` / `attention` 那类几百 KiB 的工作集；
//! - `dram`：64 MiB，远超本机末级缓存，测的是主存。
//!
//! 每档读、写**分开测**：只流式读权重的 GEMV 可以跑过拷贝上限，把两个方向合成一个数，
//! 「这层卡在哪一侧」就再也判断不了（`nn::Bounds` 的分账口径与此相同）。
//!
//! 口径仍是 **ns/iter，越小越好**，`Throughput::Bytes` 报的是每迭代搬运的字节数；判定走
//! `tool::regress`，与算子基准同一套规则（见 `tests/regress.rs`）。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

/// L1 档字数：2048 个 `u64` = 16 KiB。
const L1_WORDS: usize = 2048;

/// L1 档一轮只有 16 KiB，计时噪声会盖过信号，所以迭代内跑 64 趟。
const L1_ROUNDS: usize = 64;

/// L2 档字数：131072 个 `u64` = 1 MiB。
const L2_WORDS: usize = 128 * 1024;

/// L2 档迭代内跑 8 趟。
const L2_ROUNDS: usize = 8;

/// DRAM 档字数：8388608 个 `u64` = 64 MiB。
const DRAM_WORDS: usize = 8 * 1024 * 1024;

/// 读一遍并做异或归约。
///
/// 用无符号整数而不是 f32 求和：浮点加法默认不可重排，编译器会把归约串行化，测到的是加法
/// 延迟而不是搬运速率。搬运带宽与元素类型无关，整数读同样代表这一档的读上界。
fn xor_read(words: &[u64]) -> u64 {
    let mut acc = 0u64;
    for word in words {
        acc ^= *word;
    }
    black_box(acc)
}

/// 读方向的三档上界。
fn bench_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("mem_stream");

    let l1 = vec![0x0101_0101_0101_0101u64; L1_WORDS];
    group.throughput(Throughput::Bytes(
        (L1_ROUNDS * L1_WORDS * size_of::<u64>()) as u64,
    ));
    group.bench_function("l1_read_16kib", |bench| {
        bench.iter(|| {
            let mut acc = 0u64;
            for _ in 0..L1_ROUNDS {
                acc ^= xor_read(black_box(&l1));
            }
            black_box(acc)
        })
    });

    let l2 = vec![0x0202_0202_0202_0202u64; L2_WORDS];
    group.throughput(Throughput::Bytes(
        (L2_ROUNDS * L2_WORDS * size_of::<u64>()) as u64,
    ));
    group.bench_function("l2_read_1mib", |bench| {
        bench.iter(|| {
            let mut acc = 0u64;
            for _ in 0..L2_ROUNDS {
                acc ^= xor_read(black_box(&l2));
            }
            black_box(acc)
        })
    });

    let dram = vec![0x0303_0303_0303_0303u64; DRAM_WORDS];
    group.throughput(Throughput::Bytes((DRAM_WORDS * size_of::<u64>()) as u64));
    group.bench_function("dram_read_64mib", |bench| {
        bench.iter(|| xor_read(black_box(&dram)))
    });

    group.finish();
}

/// 写方向的三档上界：`fill` 直接落到本机的写路径（大块时通常是流式存储指令）。
fn bench_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("mem_stream");

    let mut l1 = vec![0u64; L1_WORDS];
    group.throughput(Throughput::Bytes(
        (L1_ROUNDS * L1_WORDS * size_of::<u64>()) as u64,
    ));
    group.bench_function("l1_write_16kib", |bench| {
        bench.iter(|| {
            for _ in 0..L1_ROUNDS {
                l1.fill(black_box(0x1111_1111_1111_1111));
            }
            black_box(&l1);
        })
    });

    let mut l2 = vec![0u64; L2_WORDS];
    group.throughput(Throughput::Bytes(
        (L2_ROUNDS * L2_WORDS * size_of::<u64>()) as u64,
    ));
    group.bench_function("l2_write_1mib", |bench| {
        bench.iter(|| {
            for _ in 0..L2_ROUNDS {
                l2.fill(black_box(0x2222_2222_2222_2222));
            }
            black_box(&l2);
        })
    });

    let mut dram = vec![0u64; DRAM_WORDS];
    group.throughput(Throughput::Bytes((DRAM_WORDS * size_of::<u64>()) as u64));
    group.bench_function("dram_write_64mib", |bench| {
        bench.iter(|| {
            dram.fill(black_box(0x3333_3333_3333_3333));
            black_box(&dram);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_read, bench_write);
criterion_main!(benches);
