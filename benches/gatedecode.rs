//! gate 解码吞吐诊断基准。
//!
//! 量化 gate 解码的两项关键优化，用同一数据剖面（16 MiB 相关 f32 权重，
//! 随机游走）对比：
//!
//! 1. `gate::decompress`：当前实现——8 状态交错 rANS + slot→symbol 查找表。
//! 2. 单状态查找表解码：交错化**之前**的实现（1 个 rANS 状态 + 查找表），
//!    量化交错化带来的指令级并行（ILP）增益——8 条独立状态链让 CPU 乱序
//!    执行引擎并行推进，打破单状态的串行依赖。
//!
//! 两个实现都逐字节还原原始数据（断言）；单状态流由本地编码器生成，与
//! `gate::compress` 格式独立。无第三方压缩依赖，`forbid(unsafe_code)` 保持。

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;

use ::checkpoint::gate;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

const SIZE: usize = 16 * 1024 * 1024;
const NSYMBOLS: usize = 256;
const LOG2_M: u32 = 16;
const M: u32 = 1 << LOG2_M;
const L: u32 = 1 << 23;

fn xorshift64(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn gen_weights_real(seed: &mut u64, n: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n);
    let mut val: f32 = 0.0;
    while v.len() < n {
        let r = xorshift64(seed) % 100;
        let delta: f32 = if r < 70 {
            (xorshift64(seed) % 2000) as f32 / 1e6 - 1e-3
        } else if r < 90 {
            (xorshift64(seed) % 200_000) as f32 / 1e6 - 0.1
        } else {
            0.0
        };
        val += delta;
        if val > 1.0 {
            val -= 2.0;
        }
        if val < -1.0 {
            val += 2.0;
        }
        v.extend_from_slice(&val.to_le_bytes());
    }
    v
}

/// 频率表 + 累积频率表 + slot→symbol 查找表（与 `gate.rs` 的解码表同构）。
struct TableLut {
    freq: [u32; NSYMBOLS],
    cum: [u32; NSYMBOLS + 1],
    lut: Vec<u16>,
}

impl TableLut {
    fn build(counts: &[u32; NSYMBOLS]) -> Self {
        let freq = normalize(counts);
        let mut cum = [0u32; NSYMBOLS + 1];
        let mut acc = 0u32;
        for s in 0..NSYMBOLS {
            cum[s] = acc;
            acc += freq[s];
        }
        cum[NSYMBOLS] = acc;
        let mut lut = vec![0u16; M as usize];
        let mut s = 0usize;
        for (slot, e) in lut.iter_mut().enumerate() {
            while cum[s + 1] <= slot as u32 {
                s += 1;
            }
            *e = s as u16;
        }
        Self { freq, cum, lut }
    }
}

/// 归一化 256 符号频率到总和 `M`（与 `gate.rs` 同规则）。
fn normalize(counts: &[u32; NSYMBOLS]) -> [u32; NSYMBOLS] {
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    let mut freq = [0u32; NSYMBOLS];
    if total == 0 {
        freq[0] = M;
        return freq;
    }
    let mut sum = 0i64;
    let mut max_s = 0usize;
    for (s, &c) in counts.iter().enumerate() {
        if c > 0 {
            let f = ((c as u64 * M as u64 + total / 2) / total).max(1) as u32;
            freq[s] = f;
            sum += f as i64;
            if f > freq[max_s] {
                max_s = s;
            }
        }
    }
    let mut diff = M as i64 - sum;
    while diff > 0 {
        freq[max_s] += 1;
        diff -= 1;
    }
    while diff < 0 && freq[max_s] > 1 {
        freq[max_s] -= 1;
        diff += 1;
    }
    freq
}

/// 单状态 rANS 编码（交错化前的 `encode_plane`），仅用于生成基准流。
fn encode_plane_single(data: &[u8], t: &TableLut) -> Vec<u8> {
    let mut x = L;
    let mut renorm = Vec::new();
    for &b in data {
        let s = b as usize;
        let f = t.freq[s];
        let x_max = ((L >> LOG2_M) << 8) * f;
        while x >= x_max {
            renorm.push((x & 0xFF) as u8);
            x >>= 8;
        }
        x = (x / f) * M + (x % f) + t.cum[s];
    }
    let mut out = Vec::new();
    for shift in [0, 8, 16, 24] {
        out.push(((x >> shift) & 0xFF) as u8);
    }
    for &b in renorm.iter().rev() {
        out.push(b);
    }
    out
}

/// 单状态查找表解码（交错化前的 `decode_plane`）。rANS 逆序弹符号，故反序写入。
fn decode_plane_single(stream: &[u8], t: &TableLut, n: usize) -> Vec<u8> {
    let mut data = stream;
    let mut x = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    data = &data[4..];
    let mut buf = vec![0u8; n];
    for i in (0..n).rev() {
        let slot = (x & (M - 1)) as usize;
        let s = t.lut[slot] as usize;
        buf[i] = s as u8;
        x = t.freq[s] * (x >> LOG2_M) + slot as u32 - t.cum[s];
        while x < L {
            x = (x << 8) | u32::from(data[0]);
            data = &data[1..];
        }
    }
    buf
}

fn gatedecode(c: &mut Criterion) {
    let mut seed = 0x1234_5678_9abc_def0u64;
    let data = gen_weights_real(&mut seed, SIZE);

    // 当前实现：8 状态交错 + 查找表。
    let stored = gate::compress(&data, 0);
    assert_eq!(
        gate::decompress(&stored, 0).unwrap(),
        data,
        "gate::decompress roundtrip"
    );

    // 对照：单状态 + 查找表（交错化前）。
    let mut counts = [0u32; NSYMBOLS];
    for &b in &data {
        counts[b as usize] += 1;
    }
    let t = TableLut::build(&counts);
    let single_stream = encode_plane_single(&data, &t);
    assert_eq!(
        decode_plane_single(&single_stream, &t, SIZE),
        data,
        "single-state roundtrip"
    );

    println!(
        "gate: input {} B, interleaved {} B ({:.2}%), single-state {} B ({:.2}%)",
        data.len(),
        stored.len(),
        100.0 * stored.len() as f64 / data.len() as f64,
        single_stream.len(),
        100.0 * single_stream.len() as f64 / data.len() as f64
    );

    let mut group = c.benchmark_group("gatedecode");
    group.throughput(Throughput::Bytes(SIZE as u64));
    group.bench_function("interleaved (gate::decompress)", |b| {
        b.iter(|| black_box(gate::decompress(black_box(&stored), 0).unwrap()))
    });
    group.bench_function("single-state + lut", |b| {
        b.iter(|| black_box(decode_plane_single(black_box(&single_stream), &t, SIZE)))
    });
    group.finish();
}

criterion_group!(benches, gatedecode);
criterion_main!(benches);
