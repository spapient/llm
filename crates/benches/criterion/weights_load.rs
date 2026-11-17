//! 装载基准：文件流后端（[`File`] / [`Loader`]）与映射后端（[`MmapSource`] / [`Reader`]）的
//! 首条与全量装载对照。
//!
//! 只出 ns/iter 与字节吞吐；是否回退由 `tool::regress` 的基线判定负责，本基准**不登记基线
//! 数值**——跨机绝对值不可比，沿用 `model_decode.rs` 先例。RSS 进程读数没有公开接口，本基准
//! 只记可观测字节账目，不虚构峰值内存；进程级采样留给后续刀。
//!
//! 测量结果归档为 `tool::bench::BenchRecord`（`EntryKind::Op`）：台账与记录构造在
//! `tests/regress.rs` 的 `LOAD_SCENARIOS`；记录只承载本机一次测量与身份，字节吞吐由 Report 侧
//! 按字节账目与实测 ns 换算，不进记录。
//!
//! 冷热口径：
//!
//! - `*_reopen`：每迭代重新打开文件（打开 + 头部校验 + 条目读取），贴近冷装载；
//! - `*_open`：复用已打开的源，只测条目读取，贴近热路径。
//!
//! mmap 用例仅在 Unix 目标编译（weights 的 `mmap` feature 由本成员 dev-dep 固定开启）；其它
//! 平台这一组用例整块剔除，不提供空实现。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::PathBuf;
#[cfg(unix)]
use weights::MmapSource;
use weights::safetensors::File;
use weights::{Loader, Reader};

/// 条目数。
const N_ENTRIES: usize = 8;
/// 每条 F32 元素数。
const ELEMS: usize = 256;
/// 每条字节数（F32 = 4 字节）。
const BYTES_PER_ENTRY: usize = ELEMS * 4;
/// 数据段总字节数。
const TOTAL_BYTES: usize = N_ENTRIES * BYTES_PER_ENTRY;

/// 构造一份确定性 safetensors 临时文件；返回路径与条目名（按偏移顺序）。
fn fixture() -> (PathBuf, Vec<String>) {
    let mut entries = Vec::with_capacity(N_ENTRIES);
    let mut names = Vec::with_capacity(N_ENTRIES);
    for index in 0..N_ENTRIES {
        let start = index * BYTES_PER_ENTRY;
        let end = start + BYTES_PER_ENTRY;
        names.push(format!("w.{index}"));
        entries.push(format!(
            "\"w.{index}\": {{\"dtype\": \"F32\", \"shape\": [{ELEMS}], \
             \"data_offsets\": [{start}, {end}]}}"
        ));
    }
    let json = format!("{{{}}}", entries.join(","));
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(json.as_bytes());
    // 数据段：递增字节，避免全零在压缩/页缓存上抹掉后端差异。
    let payload: Vec<u8> = (0..TOTAL_BYTES).map(|i| (i % 251) as u8).collect();
    file.extend_from_slice(&payload);

    let path = std::env::temp_dir().join(format!(
        "llm_weights_load_{}_{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&path, &file).expect("写基准临时文件");
    (path, names)
}

/// 全量拷回：遍历头部条目逐条 `load`，累加字节数防止编译器消掉读取。
fn load_all(file: &File, names: &[String]) -> usize {
    let mut produced = 0usize;
    for name in names {
        let entry = file.header().get(name.as_str()).expect("条目在位");
        let bytes = file.load(entry).expect("装载");
        produced += bytes.len();
    }
    black_box(produced)
}

/// 首条拷回：只读第一个条目。
fn load_first(file: &File, names: &[String]) -> usize {
    let entry = file.header().get(names[0].as_str()).expect("条目在位");
    let bytes = file.load(entry).expect("装载");
    black_box(bytes.len())
}

fn bench_file_reopen(c: &mut Criterion) {
    let (path, names) = fixture();
    let mut group = c.benchmark_group("weights/load");
    group.throughput(Throughput::Bytes(TOTAL_BYTES as u64));
    group.bench_function("read_all_reopen", |bench| {
        bench.iter(|| {
            let file = File::open(black_box(&path)).expect("打开");
            load_all(&file, &names);
        });
    });
    group.bench_function("read_first_reopen", |bench| {
        bench.iter(|| {
            let file = File::open(black_box(&path)).expect("打开");
            load_first(&file, &names);
        });
    });
    group.finish();
}

fn bench_file_open(c: &mut Criterion) {
    let (path, names) = fixture();
    let file = File::open(&path).expect("打开");
    let mut group = c.benchmark_group("weights/load");
    group.throughput(Throughput::Bytes(TOTAL_BYTES as u64));
    group.bench_function("read_all_open", |bench| {
        bench.iter(|| load_all(&file, &names));
    });
    group.bench_function("read_first_open", |bench| {
        bench.iter(|| load_first(&file, &names));
    });
    group.finish();
}

#[cfg(unix)]
fn mapped_all(source: &MmapSource, names: &[String]) -> usize {
    let mut produced = 0usize;
    for name in names {
        let entry = source.header().get(name.as_str()).expect("条目在位");
        let bytes = source.read(entry).expect("借用读取");
        produced += bytes.len();
    }
    black_box(produced)
}

#[cfg(unix)]
fn mapped_first(source: &MmapSource, names: &[String]) -> usize {
    let entry = source.header().get(names[0].as_str()).expect("条目在位");
    let bytes = source.read(entry).expect("借用读取");
    black_box(bytes.len())
}

#[cfg(unix)]
fn bench_mmap_reopen(c: &mut Criterion) {
    let (path, names) = fixture();
    let mut group = c.benchmark_group("weights/load");
    group.throughput(Throughput::Bytes(TOTAL_BYTES as u64));
    group.bench_function("mmap_all_reopen", |bench| {
        bench.iter(|| {
            let source = MmapSource::open(black_box(&path)).expect("映射");
            mapped_all(&source, &names);
        });
    });
    group.bench_function("mmap_first_reopen", |bench| {
        bench.iter(|| {
            let source = MmapSource::open(black_box(&path)).expect("映射");
            mapped_first(&source, &names);
        });
    });
    group.finish();
}

#[cfg(unix)]
fn bench_mmap_open(c: &mut Criterion) {
    let (path, names) = fixture();
    let source = MmapSource::open(&path).expect("映射");
    let mut group = c.benchmark_group("weights/load");
    group.throughput(Throughput::Bytes(TOTAL_BYTES as u64));
    group.bench_function("mmap_all_open", |bench| {
        bench.iter(|| mapped_all(&source, &names));
    });
    group.bench_function("mmap_first_open", |bench| {
        bench.iter(|| mapped_first(&source, &names));
    });
    group.finish();
}

#[cfg(unix)]
criterion_group!(
    benches,
    bench_file_reopen,
    bench_file_open,
    bench_mmap_reopen,
    bench_mmap_open
);
#[cfg(not(unix))]
criterion_group!(benches, bench_file_reopen, bench_file_open);
criterion_main!(benches);
