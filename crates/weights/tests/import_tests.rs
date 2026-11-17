//! 中性条目流导入往返：safetensors 字节 → 中性段流 → Writer 落 `.u` 容器 → Archive 回读，
//! 逐字段、逐字节相等；文件后端与内存归档后端同口径。

use std::time::SystemTime;

use diagnostic::Code;
use llm_weights::import::{
    segments_from_path, segments_from_safetensors_archive, segments_from_safetensors_file,
};
use llm_weights::{Archive, Reader, Wire, Writer};

/// 按 safetensors 布局拼文件：8 字节小端长度前缀 + 头部 JSON + 数据段。
fn safetensors_file(json: &str, data: &[u8]) -> Vec<u8> {
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json.as_bytes());
    bytes.extend_from_slice(data);
    bytes
}

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
