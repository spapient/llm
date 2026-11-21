//! weights 的两组基准：自有 `.u` 容器（解析 / Merkle / 查找 / 写出）与 safetensors 装载。
//!
//! 容器组共用一份字节与一个 `Archive`：解析、根、证明、校验、按名查找都是只读操作，没有理由
//! 每档重建一遍。装载组分文件流与映射两个后端、冷（每次重开）与热（复用已打开源）两种口径；
//! 临时文件整个 bench 只造一份，结束时删掉。
//!
//! 只出 ns/iter 与字节吞吐；是否回退由 `tests/regress` 的基线判定负责，本文件不登记基线数值。

use criterion::{Criterion, Throughput};
use std::hint::black_box;
use std::path::PathBuf;
use weights::Loader;
#[cfg(unix)]
use weights::MmapSource;
#[cfg(unix)]
use weights::Reader;
use weights::safetensors::File;
use weights::{Archive, Segment, Wire, Writer};

/// 容器段数。
const SEGMENTS: usize = 64;

/// 每段 F32 元素数。
const ELEMS: usize = 256;

/// 每段字节数（F32 = 4 字节）。
const BYTES_PER_SEGMENT: usize = ELEMS * 4;

/// 装载条目数。
const N_ENTRIES: usize = 8;

/// 装载每条字节数（F32 = 4 字节）。
const BYTES_PER_ENTRY: usize = ELEMS * 4;

/// 装载数据段总字节数。
const TOTAL_BYTES: usize = N_ENTRIES * BYTES_PER_ENTRY;

/// 造一份确定性容器：64 段、每段 256 个 F32。
fn container() -> Writer {
    let mut writer = Writer::new();
    for index in 0..SEGMENTS {
        let data = vec![(index % 256) as u8; BYTES_PER_SEGMENT];
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

/// 容器组：一份字节喂全部只读路径，写出单独一组（`finish` 可重复调用）。
pub(crate) fn archive(c: &mut Criterion) {
    let bytes = container().finish(6).expect("字节");
    let parsed = Archive::parse(&bytes).expect("解析");
    let root = parsed.merkle_root().expect("根");
    let proof = parsed.prove(0).expect("证明");

    let mut group = c.benchmark_group("weights/archive");
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("parse", |b| {
        b.iter(|| Archive::parse(black_box(&bytes)).expect("解析"))
    });
    group.bench_function("merkle_root", |b| {
        b.iter(|| parsed.merkle_root().expect("根"))
    });
    group.bench_function("prove", |b| {
        b.iter(|| parsed.prove(black_box(0)).expect("证明"))
    });
    group.bench_function("verify", |b| {
        b.iter(|| parsed.verify(0, &proof, &root).expect("校验"))
    });
    group.bench_function("get_by_name", |b| {
        b.iter(|| parsed.get(black_box("seg_32")).expect("段存在"))
    });
    group.finish();

    let writer = container();
    let mut writer_group = c.benchmark_group("weights/writer");
    writer_group.throughput(Throughput::Bytes((SEGMENTS * BYTES_PER_SEGMENT) as u64));
    writer_group.bench_function("finish", |b| b.iter(|| writer.finish(6).expect("字节")));
    writer_group.finish();
}

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
        produced += file.load(entry).expect("装载").len();
    }
    black_box(produced)
}

/// 首条拷回：只读第一个条目。
fn load_first(file: &File, names: &[String]) -> usize {
    let entry = file.header().get(names[0].as_str()).expect("条目在位");
    let bytes = file.load(entry).expect("装载");
    black_box(bytes.len())
}

#[cfg(unix)]
fn mapped_all(source: &MmapSource, names: &[String]) -> usize {
    let mut produced = 0usize;
    for name in names {
        let entry = source.header().get(name.as_str()).expect("条目在位");
        produced += source.read(entry).expect("借用读取").len();
    }
    black_box(produced)
}

#[cfg(unix)]
fn mapped_first(source: &MmapSource, names: &[String]) -> usize {
    let entry = source.header().get(names[0].as_str()).expect("条目在位");
    let bytes = source.read(entry).expect("借用读取");
    black_box(bytes.len())
}

/// 装载组：`*_reopen` 每迭代重开（冷），`*_open` 复用已打开源（热）；`mmap_*` 仅 Unix。
pub(crate) fn load(c: &mut Criterion) {
    let (path, names) = fixture();
    let file = File::open(&path).expect("打开");
    #[cfg(unix)]
    let source = MmapSource::open(&path).expect("映射");

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
    group.bench_function("read_all_open", |bench| {
        bench.iter(|| load_all(&file, &names));
    });
    group.bench_function("read_first_open", |bench| {
        bench.iter(|| load_first(&file, &names));
    });
    #[cfg(unix)]
    {
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
        group.bench_function("mmap_all_open", |bench| {
            bench.iter(|| mapped_all(&source, &names));
        });
        group.bench_function("mmap_first_open", |bench| {
            bench.iter(|| mapped_first(&source, &names));
        });
    }
    group.finish();

    drop(file);
    #[cfg(unix)]
    drop(source);
    let _ = std::fs::remove_file(&path);
}
