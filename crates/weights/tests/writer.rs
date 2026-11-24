//! `writer` 模块的单元测试。
use super::*;

use crate::Wire;

use crate::archive::Archive;

fn seg(name: &str, data: Vec<u8>) -> Segment {
    Segment::new(name, Wire::F32, vec![data.len() as u64 / 4], None, data).expect("自洽")
}

#[test]
fn an_empty_container_round_trips_with_a_footer() {
    let bytes = Writer::new().finish(6).expect("空容器");

    assert!(Archive::parse(&bytes).is_ok());
}

#[test]
fn encoded_len_matches_finish() {
    let mut writer = Writer::new();

    writer.push(seg("a", vec![1u8; 400])).expect("追加");

    writer.push(seg("b", vec![2u8; 80])).expect("追加");

    assert_eq!(
        writer.encoded_len(6).expect("长度"),
        writer.finish(6).expect("字节").len()
    );
}

#[test]
fn duplicate_names_are_rejected() {
    let mut writer = Writer::new();

    writer.push(seg("a", vec![0u8; 4])).expect("首次");

    assert!(writer.push(seg("a", vec![0u8; 4])).is_err());
}

#[test]
fn a_bad_align_log_is_rejected() {
    assert!(Writer::new().finish(17).is_err());
}

#[test]
fn encode_into_writes_exact_bytes() {
    let writer = Writer::new();

    let bytes = writer.finish(6).expect("参考");

    let mut buf = vec![0u8; bytes.len()];

    let n = writer.encode_into(6, &mut buf).expect("写入");

    assert_eq!(n, bytes.len());

    assert_eq!(buf, bytes);
}

#[test]
fn encode_into_rejects_small_buf() {
    let writer = Writer::new();

    let mut buf = [0u8; 2];

    assert!(writer.encode_into(6, &mut buf).is_err());
}

#[test]
fn default_matches_new() {
    let mut plain = Writer::default();
    plain.push(seg("a", vec![1u8; 64])).expect("追加");

    let mut fresh = Writer::new();
    fresh.push(seg("a", vec![1u8; 64])).expect("追加");

    assert_eq!(
        plain.finish(6).expect("默认"),
        fresh.finish(6).expect("新建")
    );
}

#[test]
fn with_config_matches_builders() {
    let mut configured = Writer::with(Config {
        checksum: false,
        compress: false,
        ..Config::default()
    });
    configured.push(seg("a", vec![1u8; 64])).expect("追加");

    let mut built = Writer::new();
    built.checksum(false).compress(false);
    built.push(seg("a", vec![1u8; 64])).expect("追加");

    assert_eq!(
        configured.finish(6).expect("配置"),
        built.finish(6).expect("builder")
    );
}

#[test]
fn encoded_len_rejects_bad_align() {
    assert!(Writer::new().encoded_len(17).is_err());
}

#[test]
fn write_to_path_without_checksums_round_trips() {
    let target = std::env::temp_dir().join(format!(
        "llm_weights_writer_{}_{}.u",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("时钟")
            .as_nanos()
    ));

    let mut writer = Writer::with(Config {
        checksum: false,
        compress: false,
        ..Config::default()
    });
    writer.push(seg("a", vec![7u8; 64])).expect("追加");
    writer.write_to_path(6, &target).expect("落盘");

    let bytes = std::fs::read(&target).expect("读回");
    let archive = Archive::parse(&bytes).expect("校验关闭仍可解析");
    assert_eq!(archive.segments().len(), 1);

    let _ = std::fs::remove_file(&target);
}

#[test]
fn parallel_encoding_is_byte_for_byte_deterministic() {
    let matrix: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
    let bytes: Vec<u8> = matrix.iter().flat_map(|v| v.to_le_bytes()).collect();
    let duplicate = bytes.clone();
    let grid: Vec<u8> = (0..64)
        .map(|i| 1.0 + (i % 4) as f32 * 0.25)
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let build = |workers: usize| {
        let mut writer = Writer::with(Config::default().with_workers(workers));
        writer
            .push(Segment::new("a", Wire::F32, vec![8, 8], None, bytes.clone()).expect("段"))
            .expect("追加");
        writer
            .push(Segment::new("b", Wire::F32, vec![8, 8], None, duplicate.clone()).expect("段"))
            .expect("追加");
        writer
            .push(Segment::new("c", Wire::F32, vec![8, 8], None, grid.clone()).expect("段"))
            .expect("追加");
        writer.push(seg("d", vec![0u8; 400])).expect("追加");
        writer.finish(6).expect("编码")
    };

    let serial = build(1);
    for workers in [0, 2, 3, 16] {
        assert_eq!(build(workers), serial, "workers = {workers}");
    }
    let archive = Archive::parse(&serial).expect("解析");
    assert!(archive.get("b").is_some_and(|seg| seg.is_ref()));
}

#[test]
fn axis_and_worker_builders_match_config() {
    let axes = crate::axes::AxisSet::WITHOUT_RANK;
    let mut configured = Writer::with(Config::default().with_axes(axes).with_workers(3));
    configured.push(seg("a", vec![1u8; 64])).expect("追加");

    let mut built = Writer::new();
    built.axes(axes).workers(3);
    built.push(seg("a", vec![1u8; 64])).expect("追加");

    assert_eq!(
        configured.finish(6).expect("配置"),
        built.finish(6).expect("builder")
    );
}
