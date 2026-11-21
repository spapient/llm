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

fn encode_into_rejects_small_buffer() {
    let writer = Writer::new();

    let mut buf = [0u8; 2];

    assert!(writer.encode_into(6, &mut buf).is_err());
}
