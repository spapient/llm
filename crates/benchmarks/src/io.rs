//! 外部格式与字节编码：自有 `.u` 容器 / safetensors 装载，分词器与 `wire` 字节码闭集。
//!
//! 两组都是"字节进、字节出"的边界层，不起编译器、不碰设备。共同的纪律：夹具在装配阶段造好并
//! 校验，计时闭包里只留被测动作；唯一的失败出口是 [`checked`]，所以本文件没有
//! `unwrap` / `expect` / `panic!`。
//!
//! 吞吐按**每条用例自己的字节数**登记（全量 8 条 vs 首条 1 条差 8 倍），与 `tests/table.rs`
//! 的 `LOAD_SCENARIOS` 逐条对齐——组级一个总数会让台账与基准各说一套。

pub(crate) mod weights {
    use crate::fixtures::{checked, found};
    use criterion::{BenchmarkGroup, Criterion, Throughput, measurement::WallTime};
    use std::hint::black_box;
    use std::path::PathBuf;
    use weights::safetensors::File;
    use weights::{Archive, Loader, Segment, Wire, Writer};
    #[cfg(unix)]
    use weights::{MmapSource, Reader};

    /// 容器段数。
    const SEGMENTS: usize = 64;

    /// 每段 F32 元素数。
    const ELEMS: usize = 256;

    /// 每段字节数（F32 = 4 字节）。
    const BYTES_PER_SEGMENT: usize = ELEMS * 4;

    /// 装载条目数。
    const N_ENTRIES: usize = 8;

    /// 装载数据段总字节数。
    const TOTAL_BYTES: usize = N_ENTRIES * BYTES_PER_SEGMENT;

    /// 造一份确定性容器：64 段、每段 256 个 F32。
    fn container() -> Writer {
        let mut writer = Writer::new();
        for index in 0..SEGMENTS {
            let data = vec![(index % 256) as u8; BYTES_PER_SEGMENT];
            let shape = vec![ELEMS as u64];
            let name = format!("seg_{index}");
            let segment = checked(Segment::new(name, Wire::F32, shape, None, data), "段自洽");
            checked(writer.push(segment), "追加段");
        }
        writer
    }

    /// 容器组：一份字节喂全部只读路径，写出单独一组（`finish` 可重复调用）。
    pub(crate) fn archive(c: &mut Criterion) {
        let bytes = checked(container().finish(6), "容器字节");
        let parsed = checked(Archive::parse(&bytes), "解析");
        let root = checked(parsed.merkle_root(), "根");
        let proof = checked(parsed.prove(0), "证明");
        let mut group = c.benchmark_group("weights/archive");
        group.throughput(Throughput::Bytes(bytes.len() as u64));

        group.bench_function("parse", |bench| {
            bench.iter(|| black_box(checked(Archive::parse(black_box(&bytes)), "解析")))
        });
        group.bench_function("merkle_root", |bench| {
            bench.iter(|| black_box(checked(parsed.merkle_root(), "根")))
        });
        group.bench_function("prove", |bench| {
            bench.iter(|| black_box(checked(parsed.prove(black_box(0)), "证明")))
        });
        group.bench_function("verify", |bench| {
            bench.iter(|| black_box(checked(parsed.verify(0, &proof, &root), "校验")))
        });
        group.finish();
        lookup(c, &parsed);
        write_group(c);
    }

    /// 按名取段：证明索引可用，`finish` 之后仍可随机访问。
    fn lookup(c: &mut Criterion, parsed: &Archive<'_>) {
        let mut group = c.benchmark_group("weights/archive");
        group.throughput(Throughput::Bytes(BYTES_PER_SEGMENT as u64));
        group.bench_function("get_by_name", |bench| {
            bench.iter(|| black_box(found(parsed.get(black_box("seg_32")), "段存在")))
        });
        group.finish();
    }

    /// 写出组：`finish` 是唯一写路径，64 段各 1 KiB。
    fn write_group(c: &mut Criterion) {
        let writer = container();
        let mut group = c.benchmark_group("weights/writer");
        group.throughput(Throughput::Bytes((SEGMENTS * BYTES_PER_SEGMENT) as u64));
        group.bench_function("finish", |bench| {
            bench.iter(|| black_box(checked(writer.finish(6), "写出字节")))
        });
        group.finish();
    }

    /// 构造一份确定性 safetensors 临时文件；返回路径与条目名（按偏移顺序）。
    fn fixture() -> (PathBuf, Vec<String>) {
        let mut entries = Vec::with_capacity(N_ENTRIES);
        let mut names = Vec::with_capacity(N_ENTRIES);
        for index in 0..N_ENTRIES {
            let start = index * BYTES_PER_SEGMENT;
            let end = start + BYTES_PER_SEGMENT;
            names.push(format!("w.{index}"));
            entries.push(format!(
                "\"w.{index}\": {{\"dtype\": \"F32\", \"shape\": [{ELEMS}], \
                 \"data_offsets\": [{start}, {end}]}}"
            ));
        }
        let json = format!("{{{}}}", entries.join(","));
        let mut file = (json.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(json.as_bytes());
        // 数据段递增字节：全零会被页缓存与压缩抹平后端差异。
        file.extend((0..TOTAL_BYTES).map(|i| (i % 251) as u8));
        let path = temp_path();
        checked(std::fs::write(&path, &file), "写基准临时文件");
        (path, names)
    }

    /// 本进程独享的临时文件路径。
    fn temp_path() -> PathBuf {
        let nanos = checked(
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH),
            "系统时钟",
        )
        .as_nanos();
        std::env::temp_dir().join(format!(
            "llm_weights_load_{}_{nanos}.bin",
            std::process::id()
        ))
    }

    /// 装载组：`*_reopen` 每迭代重开（冷），`*_open` 复用已打开源（热）；`mmap_*` 仅 Unix。
    ///
    /// 四条 `read_*` 与四条 `mmap_*` 共用同一个 `plan`：只换"怎么开"与"怎么读"，
    /// 条目名、首条 / 全量与吞吐口径只有一份。
    pub(crate) fn load(c: &mut Criterion) {
        let (path, names) = fixture();
        let file = checked(File::open(&path), "打开");
        let mut group = c.benchmark_group("weights/load");
        let all = Throughput::Bytes(TOTAL_BYTES as u64);
        let first = Throughput::Bytes(BYTES_PER_SEGMENT as u64);

        group.throughput(all);
        group.bench_function("read_all_reopen", |bench| {
            bench.iter(|| {
                let file = checked(File::open(black_box(&path)), "冷开");
                loads(&file, &names, false)
            })
        });
        group.bench_function("read_all_open", |bench| {
            bench.iter(|| loads(&file, &names, false))
        });

        group.throughput(first);
        group.bench_function("read_first_reopen", |bench| {
            bench.iter(|| {
                let file = checked(File::open(black_box(&path)), "冷开");
                loads(&file, &names, true)
            })
        });
        group.bench_function("read_first_open", |bench| {
            bench.iter(|| loads(&file, &names, true))
        });
        mapped(&mut group, &path, &names);
        group.finish();
        checked(std::fs::remove_file(&path), "删除临时文件");
    }

    /// 文件流后端的一次装载：`first` 为真时只读首条。
    fn loads(file: &File, names: &[String], first: bool) -> usize {
        let lanes = if first { &names[..1] } else { names };
        let mut produced = 0usize;
        for name in lanes {
            let entry = found(file.header().get(name.as_str()), "条目在位");
            produced += checked(file.load(entry), "装载").len();
        }
        black_box(produced)
    }

    /// 映射后端的四条用例：共享 `loads` 的条目选择与吞吐口径，只换开与读。
    #[cfg(unix)]
    fn mapped(group: &mut BenchmarkGroup<'_, WallTime>, path: &PathBuf, names: &[String]) {
        let source = checked(MmapSource::open(path), "映射");
        let all = Throughput::Bytes(TOTAL_BYTES as u64);
        let first = Throughput::Bytes(BYTES_PER_SEGMENT as u64);
        group.throughput(all);
        group.bench_function("mmap_all_reopen", |bench| {
            bench.iter(|| {
                let source = checked(MmapSource::open(black_box(path)), "冷映射");
                views(&source, names, false)
            })
        });
        group.bench_function("mmap_all_open", |bench| {
            bench.iter(|| views(&source, names, false))
        });
        group.throughput(first);
        group.bench_function("mmap_first_reopen", |bench| {
            bench.iter(|| {
                let source = checked(MmapSource::open(black_box(path)), "冷映射");
                views(&source, names, true)
            })
        });
        group.bench_function("mmap_first_open", |bench| {
            bench.iter(|| views(&source, names, true))
        });
    }

    /// 映射后端一次借用读取。
    #[cfg(unix)]
    fn views(source: &MmapSource, names: &[String], first: bool) -> usize {
        let lanes = if first { &names[..1] } else { names };
        let mut produced = 0usize;
        for name in lanes {
            let entry = found(source.header().get(name.as_str()), "条目在位");
            produced += checked(source.read(entry), "借用读取").len();
        }
        black_box(produced)
    }

    /// 非 Unix 目标上没有映射后端，装载组的骨架保持一致。
    #[cfg(not(unix))]
    fn mapped(_: &mut BenchmarkGroup<'_, WallTime>, _: &PathBuf, _: &[String]) {}
}

pub(crate) mod text {
    use crate::fixtures::checked;
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
    ///
    /// 只用于跑通合并路径，不代表任何发布模型的词表规模。
    fn bpe_encoder() -> Bpe {
        let mut pieces: Vec<Vec<u8>> = (0..=255u8).map(|byte| vec![byte]).collect();
        pieces.extend([b"he".to_vec(), b"ll".to_vec(), b"  ".to_vec()]);
        let merge = |left: u8, right: u8| Merge {
            left: vec![left],
            right: vec![right],
        };
        let merges = vec![merge(b'h', b'e'), merge(b'l', b'l'), merge(b' ', b' ')];
        let source = VocabSource {
            family: VocabFamily::Bpe,
            pieces,
            merges,
            ..VocabSource::default()
        };
        let vocab = checked(tokenizer::Vocab::try_new(source), "基准词表构造");
        checked(Bpe::new(vocab), "基准 BPE 实例")
    }

    /// 分词器五组：编码（基线 / BPE）、计数、解码（基线 / BPE）、详细对齐、流式增量。
    ///
    /// 每组一个函数：`criterion` 的 `iter` 要一个空闭包，而空闭包在复杂度口径里算一个条件
    /// 分支，所以一组一函数既是语义边界，也让每组停在 `Ccn` 上限以内。
    pub(crate) fn tokenizer(c: &mut Criterion) {
        let text = sample();
        let encoder = bpe_encoder();
        let plain = plain(&text);
        encode(c, &text, &encoder);
        count(c, &text);
        decode(c, &text, &encoder);
        detailed(c, &text);
        stream(c, &plain);
    }

    /// 编码：合成 BPE 与单字节基线，同一份 4 KiB 语料。
    fn encode(c: &mut Criterion, text: &str, encoder: &Bpe) {
        let mut group = c.benchmark_group("tokenizer/encode");
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_function("bpe_synth_4kib", |bench| {
            bench.iter(|| black_box(checked(encoder.encode(black_box(text)), "编码")))
        });
        group.bench_function("bytes_4kib", |bench| {
            bench.iter(|| black_box(checked(Bytes.encode(black_box(text)), "编码")))
        });
        group.finish();
    }

    /// 计数：只数不分配。
    fn count(c: &mut Criterion, text: &str) {
        let mut group = c.benchmark_group("tokenizer/count");
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_function("bytes_4kib", |bench| {
            bench.iter(|| black_box(checked(Bytes.count(black_box(text)), "计数")))
        });
        group.finish();
    }

    /// 解码：先按各自方式编码一次再回解，两边输入同一份 4 KiB 语料。
    fn decode(c: &mut Criterion, text: &str, encoder: &Bpe) {
        let bpe_tokens = checked(encoder.encode(text), "预置编码");
        let tokens = checked(Bytes.encode(text), "预置编码");
        let mut group = c.benchmark_group("tokenizer/decode");
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_function("bpe_synth_4kib", |bench| {
            bench.iter(|| black_box(checked(encoder.decode(black_box(&bpe_tokens)), "解码")))
        });
        group.bench_function("bytes_4kib", |bench| {
            bench.iter(|| black_box(checked(Bytes.decode(black_box(&tokens)), "解码")))
        });
        group.finish();
    }

    /// 详细编码：每 token 带字节区间，供对齐与调试。
    fn detailed(c: &mut Criterion, text: &str) {
        let mut group = c.benchmark_group("tokenizer/encode_detailed");
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_function("bytes_4kib", |bench| {
            bench.iter(|| black_box(checked(Bytes.encode_detailed(black_box(text)), "详细编码")))
        });
        group.finish();
    }

    /// 流式增量解码：32 字节一块推进，`finish` 收尾。
    fn stream(c: &mut Criterion, plain: &[u8]) {
        let mut group = c.benchmark_group("tokenizer/stream");
        group.throughput(Throughput::Bytes(plain.len() as u64));
        group.bench_function("push_32b_chunks", |bench| {
            bench.iter(|| {
                let mut decoder = DecoderStream::new();
                for chunk in plain.chunks(CHUNK) {
                    black_box(checked(decoder.push(black_box(chunk)), "增量解码"));
                }
                black_box(checked(decoder.finish(), "收尾"));
            })
        });
        group.finish();
    }

    /// 单字节基线的 token 位模式，供流式解码按 32 字节切块。
    fn plain(text: &str) -> Vec<u8> {
        let tokens = checked(Bytes.encode(text), "预置编码");
        tokens.iter().map(|token| token.id() as u8).collect()
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

    /// CRC 输入字节数。
    const CRC_BYTES: usize = 4096;

    /// 密封 Sink 输入字节数。
    const SEALED_BYTES: usize = 1024;

    /// `wire` 三组：CRC 更新、LEB128 编解码、密封 Sink。
    pub(crate) fn wire(c: &mut Criterion) {
        crc(c);
        leb(c);
        sealed(c);
    }

    /// CRC32 按 64 字节分块推进。
    fn crc(c: &mut Criterion) {
        let data: Vec<u8> = (0..=255).cycle().take(CRC_BYTES).collect();
        let mut group = c.benchmark_group("wire/crc");
        group.throughput(Throughput::Bytes(CRC_BYTES as u64));
        group.bench_function("update_4kib", |bench| {
            bench.iter(|| {
                let mut state = !0u32;
                for chunk in data.chunks(64) {
                    state = update(state, black_box(chunk));
                }
                black_box(state)
            })
        });
        group.finish();
    }

    /// LEB128 编解码：两档各一个函数，共用同一份八档取值。
    fn leb(c: &mut Criterion) {
        leb_encode(c);
        leb_decode(c);
    }

    /// 编码：八档长度（1..10 字节）各写一遍。
    fn leb_encode(c: &mut Criterion) {
        let mut group = c.benchmark_group("wire/leb");
        group.throughput(Throughput::Elements(VALUES.len() as u64));
        group.bench_function("encode_values", |bench| {
            bench.iter(|| {
                let mut sink = Count(0);
                for value in VALUES {
                    write(&mut sink, black_box(value));
                }
                black_box(sink.0)
            })
        });
        group.finish();
    }

    /// 解码：同一份取值先编码成缓冲，再逐条读回。
    fn leb_decode(c: &mut Criterion) {
        let mut buffer = Vec::new();
        for value in VALUES {
            write(&mut buffer, value);
        }
        let mut group = c.benchmark_group("wire/leb");
        group.throughput(Throughput::Elements(VALUES.len() as u64));
        group.bench_function("decode_values", |bench| {
            bench.iter(|| {
                let mut cursor = Cursor::new(black_box(&buffer));
                for _ in VALUES {
                    black_box(checked(cursor.read::<u64>(), "规范编码"));
                }
            })
        });
        group.finish();
    }

    /// 密封 Sink：32 字节一块推进，最后取校验和。
    fn sealed(c: &mut Criterion) {
        let data: Vec<u8> = (0..=255).cycle().take(SEALED_BYTES).collect();
        let mut group = c.benchmark_group("wire/sink");
        group.throughput(Throughput::Bytes(SEALED_BYTES as u64));
        group.bench_function("sealed_1kib", |bench| {
            bench.iter(|| {
                let mut out = Vec::new();
                let mut sink = Sealed::new(&mut out);
                for chunk in data.chunks(32) {
                    sink.put(black_box(chunk));
                }
                black_box(sink.checksum());
                black_box(&out);
            })
        });
        group.finish();
    }
}
