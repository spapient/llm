//! 端到端契约：全 wire 类型往返、量化段往返、对齐布局，以及两级 CRC 对损坏/截断的拦截；
//! 另覆盖 safetensors 字节 → 中性段流 → `.u` 容器的导入往返。
use std::time::SystemTime;

use diagnostic::Code;
use llm_weights::import::{
    segments_from_path, segments_from_safetensors_archive, segments_from_safetensors_file,
};
use llm_weights::safetensors::{Archive as SfArchive, File as SfFile};
use llm_weights::{Archive, Loader, Quant, Reader, Segment, Wire, Writer, patch};

/// 按 safetensors 布局拼文件：8 字节小端长度前缀 + 头部 JSON + 数据段。
fn safetensors_file(json: &str, data: &[u8]) -> Vec<u8> {
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json.as_bytes());
    bytes.extend_from_slice(data);
    bytes
}

/// 给每个 wire 类型造一个 `n` 元素的段，数据填递增字节，长度正好是形状推出的字节数。
fn segment_of(wire: Wire, index: usize) -> Segment {
    let numel = 11u64; // 奇数，顺带钉亚字节类型的向上取整
    let bytes = wire.bytes_for(numel) as usize;
    let data: Vec<u8> = (0..bytes)
        .map(|b| (b as u8).wrapping_add(index as u8))
        .collect();
    Segment::new(format!("seg_{index}"), wire, vec![numel], None, data).expect("段自洽")
}

/// 写一个 F32 + F16 两段归档到带 `prefix` 的唯一临时路径，返回该路径。
fn write_pair(prefix: &str) -> std::path::PathBuf {
    let mut writer = Writer::new();
    writer.push(segment_of(Wire::F32, 0)).expect("追加");
    writer.push(segment_of(Wire::F16, 1)).expect("追加");
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("时间")
        .as_nanos();
    let target = std::env::temp_dir().join(format!("{prefix}_{}_{}.u", std::process::id(), nanos));
    writer.write_to_path(6, &target).expect("原子写");
    target
}

#[test]
fn every_wire_type_round_trips_byte_for_byte() {
    let mut writer = Writer::new();
    for (index, wire) in Wire::ALL.iter().enumerate() {
        writer.push(segment_of(*wire, index)).expect("追加");
    }
    let bytes = writer.finish(6).expect("64 B 对齐");
    let archive = Archive::parse(&bytes).expect("自证合法");
    assert_eq!(archive.segments().len(), Wire::ALL.len());

    for (index, wire) in Wire::ALL.iter().enumerate() {
        let expected = segment_of(*wire, index);
        let stored = archive.get(&format!("seg_{index}")).expect("按名查");
        assert_eq!(stored.dtype(), *wire);
        assert_eq!(stored.shape(), &[11]);
        assert_eq!(
            archive.load(stored).expect("加载逻辑字节").as_ref(),
            expected.data.as_slice()
        );
    }
}

#[test]
fn a_quantized_segment_keeps_its_scheme() {
    let quant = Quant::new(1, 0, 32, false, Wire::F16, None).expect("per-block");
    let segment = Segment::new(
        "blk.0.q.weight",
        Wire::U8,
        vec![8, 64],
        Some(quant),
        vec![3u8; 512],
    )
    .expect("量化段");

    let mut writer = Writer::new();
    writer.push(segment).expect("追加");
    let bytes = writer.finish(8).expect("256 B 对齐");
    let archive = Archive::parse(&bytes).expect("解析");
    let stored = archive.get("blk.0.q.weight").expect("段存在");
    assert_eq!(stored.quant(), Some(&quant));
    // 全相同字节高度可压，应自动选 rans 且严格变小。
    assert_eq!(stored.codec(), llm_weights::Codec::Rans);
    assert!(stored.len() < 512);
    assert_eq!(
        archive.load(stored).expect("解压").as_ref(),
        vec![3u8; 512].as_slice()
    );
}

#[test]
fn a_flipped_bit_is_rejected_by_the_global_crc() {
    let bytes = Writer::new()
        .push(segment_of(Wire::F32, 0))
        .expect("追加")
        .finish(6)
        .expect("产出");

    let mut damaged = bytes.clone();
    let flip = damaged.len() / 2;
    damaged[flip] ^= 0x01;
    let error = Archive::parse(&damaged).expect_err("比特翻转必须被拦");
    assert_eq!(error.code().as_str(), "invalid");
}

#[test]
fn a_truncated_file_is_rejected() {
    let bytes = Writer::new()
        .push(segment_of(Wire::F16, 0))
        .expect("追加")
        .finish(6)
        .expect("产出");
    let cut = &bytes[..bytes.len() - 7];
    assert!(Archive::parse(cut).is_err());
}

#[test]
fn checksums_can_be_disabled_and_then_a_flip_survives() {
    let mut writer = Writer::new();
    writer.checksum(false);
    writer.push(segment_of(Wire::F32, 0)).expect("追加");
    let bytes = writer.finish(6).expect("产出");

    let source = Archive::parse(&bytes).expect("原文件合法");
    let start = source.segments()[0].start();
    let mut damaged = bytes.clone();
    damaged[start + 1] ^= 0x01; // 精确翻数据段逻辑字节，不碰头部与对齐填充
    assert!(Archive::parse(&damaged).is_ok(), "关闭校验后读端不验 CRC");
}

#[test]
fn merkle_proofs_round_trip_and_bind_names() {
    let mut writer = Writer::new();
    for index in 0..5 {
        writer.push(segment_of(Wire::F32, index)).expect("追加");
    }
    let writer_root = writer.merkle_root();
    let bytes = writer.finish(6).expect("产出");
    let archive = Archive::parse(&bytes).expect("解析");
    let archive_root = archive.merkle_root().expect("根");
    // 写端与读端在同一逻辑内容上得到同一根。
    assert_eq!(writer_root, archive_root);

    for index in 0..5 {
        let proof = archive.prove(index).expect("证明");
        assert!(
            archive.verify(index, &proof, &archive_root).expect("验证"),
            "下标 {index} 应包含于根"
        );
    }
    // 越界下标取不到证明。
    assert!(archive.prove(99).is_err());
    // 错误根不能让证明通过。
    let proof0 = archive.prove(0).expect("证明");
    assert!(!archive.verify(0, &proof0, &[0u8; 32]).expect("验证"));
}

#[test]
fn atomic_write_produces_a_valid_file_without_temp_residue() {
    let target = write_pair("llm_atomic");
    let on_disk = std::fs::read(&target).expect("读回");
    let archive = Archive::parse(&on_disk).expect("落盘文件合法");
    assert_eq!(archive.segments().len(), 2);

    // 同目录不应留下本次的 .tmp 临时文件。
    let prefix = format!(".{}", target.file_name().expect("文件名").to_string_lossy());
    let mut residue = std::fs::read_dir(std::env::temp_dir())
        .expect("读目录")
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(&prefix));
    assert!(residue.next().is_none(), "存在临时文件残留");

    std::fs::remove_file(&target).expect("清理目标文件");
}

// `storage::Mmap` 只在 Unix 目标上存在（`mmap` feature 的 `cfg(all(feature = "mmap", unix))`），
// 因此这条用例也只在 Unix 上编译——非 Unix 目标不静默跳过，而是根本不存在这条断言。
#[cfg(unix)]
#[test]
fn a_mapped_file_opens_without_copying_into_memory() {
    let target = write_pair("llm_mmap_open");

    // 映射文件，直接在映射字节上解析，数据段不复制进内存。
    let map = storage::Mmap::open(&target).expect("映射");
    let archive = Archive::parse(map.as_slice()).expect("映射字节合法");
    assert_eq!(archive.segments().len(), 2);
    for index in 0..2 {
        let stored = &archive.segments()[index];
        let expected = segment_of(if index == 0 { Wire::F32 } else { Wire::F16 }, index);
        assert_eq!(
            archive.load(stored).expect("加载").as_ref(),
            expected.data.as_slice()
        );
    }

    std::fs::remove_file(&target).expect("清理");
}

#[test]
fn incremental_patch_recovers_cur_byte_for_byte() {
    // 模拟训练微调：base 是 8192 个 f32，cur 只改其中 3 个值，形状完全不变。
    let base_vals = vec![0.5f32; 8192];
    let mut cur_vals = base_vals.clone();
    cur_vals[10] = 0.5001;
    cur_vals[4000] = 0.4999;
    cur_vals[8191] = 0.5002;

    let f32_bytes =
        |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let pack = |vals: &[f32]| {
        let segment = Segment::new(
            "w",
            Wire::F32,
            vec![vals.len() as u64],
            None,
            f32_bytes(vals),
        )
        .expect("段");
        let mut writer = Writer::new();
        writer.compress(true);
        writer.push(segment).expect("push");
        writer.finish(6).expect("字节")
    };

    let base_bytes = pack(&base_vals);
    let base = Archive::parse(&base_bytes).expect("base");
    let cur_bytes = pack(&cur_vals);
    let cur = Archive::parse(&cur_bytes).expect("cur");

    // 补丁落盘（XOR 大量零，再经 rans 压缩），应显著小于全量 cur。
    let mut patch_writer = Writer::new();
    patch_writer.compress(true);
    for segment in patch::diff(&base, &cur).expect("diff") {
        patch_writer.push(segment).expect("push");
    }
    let patch_bytes = patch_writer.finish(6).expect("patch");
    assert!(patch_bytes.len() < cur_bytes.len() / 2, "补丁应远小于全量");

    // 读端按 base 合成，逐字节还原 cur。
    let patch_archive = Archive::parse(&patch_bytes).expect("patch 解析");
    let joined = patch::join(&base, &patch_archive).expect("join");
    let cur_logical = cur.load(&cur.segments()[0]).expect("cur 字节");
    assert_eq!(joined[0].data, cur_logical.as_ref());
}

#[test]
fn safetensors_entries_round_trip_byte_for_byte() {
    let json = r#"{"a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
                   "b": {"dtype": "F16", "shape": [4], "data_offsets": [24, 32]}}"#;
    let data: Vec<u8> = (0..32).collect();
    let bytes = safetensors_file(json, &data);
    let archive = SfArchive::parse(&bytes).expect("头部自证合法");

    let a = archive.header().get("a").expect("条目 a");
    assert_eq!(Reader::read(&archive, a).expect("可读"), &data[..24]);
    let b = archive.header().get("b").expect("条目 b");
    assert_eq!(Reader::read(&archive, b).expect("可读"), &data[24..]);
}

#[test]
fn safetensors_file_backend_reads_back_the_same_bytes() {
    let json = r#"{"a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
                   "b": {"dtype": "I8", "shape": [4], "data_offsets": [24, 28]}}"#;
    let data: Vec<u8> = (0..28).collect();
    let bytes = safetensors_file(json, &data);

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("时间")
        .as_nanos();
    let target = std::env::temp_dir().join(format!(
        "llm_st_roundtrip_{}_{}.safetensors",
        std::process::id(),
        nanos
    ));
    std::fs::write(&target, &bytes).expect("写临时文件");

    let on_disk = SfFile::open(&target).expect("头部自证合法");
    let archive = SfArchive::parse(&bytes).expect("内存副本合法");
    for entry in archive.header().entries() {
        assert_eq!(
            Loader::load(&on_disk, entry).expect("磁盘可读"),
            Loader::load(&archive, entry).expect("内存可读"),
            "条目 `{}` 两个后端字节一致",
            entry.name()
        );
    }
    std::fs::remove_file(&target).expect("清理");
}

#[test]
fn quantized_grouping_survives_a_safetensors_round_trip() {
    // 真实 4 位打包：权重按 U8 存两个 nibble 一个字节（第二维是字节数），标度每分组一个 F16。
    // 分组由标度形状推出，不经打包条目猜；这条往返守护推断结果不退化。
    let json = r#"{"w.weight": {"dtype": "U8", "shape": [8, 32], "data_offsets": [0, 256]},
                   "w.scales": {"dtype": "F16", "shape": [8, 2], "data_offsets": [256, 288]}}"#;
    let data = vec![0u8; 288];
    let bytes = safetensors_file(json, &data);
    let archive = SfArchive::parse(&bytes).expect("头部自证合法");

    let quantized = archive
        .header()
        .quantization("w.weight")
        .expect("成对条目自洽")
        .expect("是量化权重");
    assert_eq!(quantized.group(), 32, "64 个 nibble / 2 个分组");
    assert_eq!(quantized.in_features(), 64, "宽度是元素数，不是字节数");
    assert_eq!(quantized.out_features(), 8);
    assert_eq!(quantized.bits(), 4);
    assert_eq!(quantized.scales_name(), "w.scales");
    assert_eq!(
        archive.header().scales_names().expect("自洽"),
        ["w.scales"],
        "标度伙伴被同一规则排除"
    );
}

#[test]
fn axes_codec_round_trips_low_cardinality_f32() {
    let vals = [0.25f32, -0.5, 0.75, 1.5];
    let w: Vec<f32> = (0..64 * 64).map(|i| vals[i % vals.len()]).collect();
    let data: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let segment = Segment::new(
        "blk.0.w.weight",
        Wire::F32,
        vec![64, 64],
        None,
        data.clone(),
    )
    .expect("自洽");
    let mut writer = Writer::new();
    writer.push(segment).expect("追加");
    let bytes = writer.finish(6).expect("64 B 对齐");
    let archive = Archive::parse(&bytes).expect("解析");
    let stored = archive.get("blk.0.w.weight").expect("段存在");
    // 低基数 f32 矩阵应命中五轴 dict 轴，且帧严格小于原始字节。
    assert_eq!(stored.codec(), llm_weights::Codec::Axes);
    assert!(stored.len() < data.len());
    assert_eq!(
        archive.load(stored).expect("解压").as_ref(),
        data.as_slice()
    );
}

// ---------------------------------------------------------------------------
// 中性条目流导入往返
// ---------------------------------------------------------------------------
//
// 中性条目流导入往返：safetensors 字节 → 中性段流 → Writer 落 `.u` 容器 → Archive 回读，
// 逐字段、逐字节相等；文件后端与内存归档后端同口径。

fn fixture() -> (String, Vec<u8>) {
    let json = r#"{"a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
                   "b": {"dtype": "F16", "shape": [4], "data_offsets": [24, 32]}}"#;
    let data: Vec<u8> = (0..32).collect();
    (json.to_string(), data)
}

#[test]
fn archive_imports_entries_then_round_trips_through_u_container() {
    let (json, data) = fixture();
    let bytes = safetensors_file(&json, &data);
    let source = llm_weights::safetensors::Archive::parse(&bytes).expect("头部自证合法");

    let segments = segments_from_safetensors_archive(&source).expect("内存归档导入");
    assert_eq!(segments.len(), 2);

    // 字段逐一对拍：名字、类型、形状、字节。
    for (segment, entry) in segments.iter().zip(source.header().entries()) {
        assert_eq!(segment.name(), entry.name());
        assert_eq!(segment.dtype(), entry.dtype());
        assert_eq!(
            segment.shape(),
            &entry
                .shape()
                .dims()
                .iter()
                .map(|&d| d as u64)
                .collect::<Vec<_>>()[..]
        );
        let borrowed = Reader::read(&source, entry).expect("条目字节");
        assert_eq!(segment.data, borrowed);
    }

    // 中性段流经既有 Writer 落 `.u`，回读逻辑字节逐位一致。
    let mut writer = Writer::new();
    for segment in segments {
        writer.push(segment).expect("追加段不重名");
    }
    let artifact = writer.finish(6).expect("落 .u 容器");
    let archive = Archive::parse(&artifact).expect("容器自证合法");
    assert_eq!(archive.segments().len(), 2);

    let back_a = archive.get("a").expect("段 a 存在");
    assert_eq!(back_a.dtype(), Wire::F32);
    assert_eq!(back_a.shape(), &[2, 3]);
    assert_eq!(
        archive.load(back_a).expect("a 逻辑字节").as_ref(),
        &data[..24]
    );

    let back_b = archive.get("b").expect("段 b 存在");
    assert_eq!(back_b.dtype(), Wire::F16);
    assert_eq!(back_b.shape(), &[4]);
    assert_eq!(
        archive.load(back_b).expect("b 逻辑字节").as_ref(),
        &data[24..32]
    );
}

#[test]
fn file_backend_imports_the_same_entries_as_the_archive() {
    let (json, data) = fixture();
    let bytes = safetensors_file(&json, &data);

    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("时间")
        .as_nanos();
    let target = std::env::temp_dir().join(format!(
        "llm_import_st_{}_{}.safetensors",
        std::process::id(),
        nanos
    ));
    std::fs::write(&target, &bytes).expect("写临时文件");

    let file_segments = segments_from_safetensors_file(&target).expect("磁盘导入");
    let archive = llm_weights::safetensors::Archive::parse(&bytes).expect("内存副本合法");
    let archive_segments = segments_from_safetensors_archive(&archive).expect("内存导入");

    assert_eq!(file_segments.len(), archive_segments.len());
    for (file, archived) in file_segments.iter().zip(&archive_segments) {
        assert_eq!(file.name(), archived.name());
        assert_eq!(file.dtype(), archived.dtype());
        assert_eq!(file.shape(), archived.shape());
        assert_eq!(file.data, archived.data);
    }

    std::fs::remove_file(&target).expect("清理");
}

#[test]
fn path_dispatch_rejects_an_unregistered_extension() {
    let target = std::env::temp_dir().join(format!("llm_import_{}.binx", std::process::id()));
    std::fs::write(&target, b"nope").expect("写临时文件");

    let error = segments_from_path(&target).expect_err("未登记后缀必须显式失败");
    assert_eq!(error.code(), Code::Unsupported);
    std::fs::remove_file(&target).expect("清理");
}
