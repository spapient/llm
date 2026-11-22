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
    });
    writer.push(seg("a", vec![7u8; 64])).expect("追加");
    writer.write_to_path(6, &target).expect("落盘");

    let bytes = std::fs::read(&target).expect("读回");
    let archive = Archive::parse(&bytes).expect("校验关闭仍可解析");
    assert_eq!(archive.segments().len(), 1);

    let _ = std::fs::remove_file(&target);
}
