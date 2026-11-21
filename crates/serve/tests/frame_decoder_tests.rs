//! `serve::frame::decoder` 的用例：头、校验和与分帧。
//!
//! 生产文件只留执行与判据，用例集中于此。

use super::*;
use crate::protocol::{Envelope, Kind, Version};

/// 一条混着四种类型、两种版本、若干载荷长度的流。
fn stream() -> (Vec<u8>, Vec<Envelope>) {
    let mut bytes = Vec::new();
    let mut frames = Vec::new();
    for (index, kind) in Kind::ALL.iter().copied().enumerate() {
        let version = if index % 2 == 0 {
            Version::FIRST
        } else {
            Version::current()
        };
        let envelope = Envelope::new(version, kind, vec![index as u8; index * 3]);
        bytes.extend_from_slice(&envelope.frame().expect("编码"));
        frames.push(envelope);
    }
    (bytes, frames)
}

/// 把当前能取出来的帧全部取走（复制成拥有的信封，缓冲的借用随之结束）。
fn drain(decoder: &mut Decoder) -> Vec<Envelope> {
    let mut out = Vec::new();
    while let Some(frame) = decoder.next_frame().expect("解析") {
        out.push(frame.envelope());
    }
    out
}

#[test]
fn a_stream_read_one_byte_at_a_time_delivers_every_frame() {
    let (bytes, frames) = stream();
    let mut decoder = Decoder::new();
    let mut got = Vec::new();
    for byte in &bytes {
        decoder.push(std::slice::from_ref(byte)).expect("追加");
        got.extend(drain(&mut decoder));
    }
    assert_eq!(got, frames);
    assert_eq!(decoder.buffered(), 0, "每帧都被取走了，不该留下字节");
    decoder.finish().expect("正好停在帧边界");
}

#[test]
fn frames_arriving_together_are_all_delivered() {
    let (bytes, frames) = stream();
    let mut decoder = Decoder::new();
    decoder.push(&bytes).expect("一次全给");
    assert_eq!(drain(&mut decoder), frames);
    decoder.finish().expect("正好停在帧边界");
}

/// 每一个切法都必须得到同一条帧序列：`start` / 压缩 / 半帧这些状态机错误只在某个切法下出现。
#[test]
fn every_way_of_splitting_a_stream_delivers_the_same_frames() {
    let (bytes, frames) = stream();
    let mut generator = 0x2545_f491_4f6c_dd1du64;
    let mut roll = move || {
        generator ^= generator << 13;
        generator ^= generator >> 7;
        generator ^= generator << 17;
        generator
    };
    for _ in 0..64 {
        let mut decoder = Decoder::new();
        let mut got = Vec::new();
        let mut at = 0;
        while at < bytes.len() {
            let end = (at + 1 + (roll() as usize) % 7).min(bytes.len());
            decoder.push(&bytes[at..end]).expect("追加");
            at = end;
            got.extend(drain(&mut decoder));
        }
        decoder.finish().expect("正好停在帧边界");
        assert_eq!(got, frames);
    }
}

#[test]
fn a_partial_frame_is_not_an_error_but_finish_is() {
    let (bytes, frames) = stream();
    let mut decoder = Decoder::new();
    decoder
        .push(&bytes[..bytes.len() - 1])
        .expect("半帧不是错误");
    // 前面三帧是完整的，必须照常交付。
    assert_eq!(drain(&mut decoder), frames[..frames.len() - 1]);
    assert!(decoder.next_frame().expect("半帧不是错误").is_none());
    assert!(decoder.buffered() > 0, "最后那一帧还差一个字节");
    let error = decoder.finish().expect_err("断在帧中间");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("帧中间"), "{error}");
}

#[test]
fn a_broken_header_poisons_the_stream() {
    let mut bytes = Envelope::new(Version::current(), Kind::Chunk, b"ok".to_vec())
        .frame()
        .expect("编码");
    bytes[0] = 0xff;
    let mut decoder = Decoder::new();
    assert_eq!(
        decoder.push(&bytes).expect_err("魔数不符").code(),
        Code::Invalid
    );

    // 再喂一帧好帧也不复活：帧头坏了就没有可靠的重同步点。
    let good = Envelope::new(Version::current(), Kind::Done, Vec::new())
        .frame()
        .expect("编码");
    assert!(decoder.push(&good).is_err(), "坏流不再接受字节");
    assert!(decoder.next_frame().is_err(), "坏了就不再交付帧");

    decoder.reset();
    decoder.push(&good).expect("reset 之后是另一条连接");
    assert_eq!(drain(&mut decoder).len(), 1);
}

#[test]
fn an_oversized_declaration_is_rejected_before_the_payload_arrives() {
    let mut header = Envelope::new(Version::current(), Kind::Chunk, Vec::new())
        .frame()
        .expect("编码");
    header[6..10].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut decoder = Decoder::new();
    // 只给帧头：正常情况这里应该安心等载荷，但声明已经超限，必须此刻拒绝。
    assert_eq!(
        decoder
            .push(&header[..HEADER_LEN])
            .expect_err("超限")
            .code(),
        Code::Invalid
    );
    assert!(decoder.next_frame().is_err());
}

#[test]
fn a_corrupted_payload_is_caught_by_the_checksum() {
    let mut bytes = Envelope::new(Version::current(), Kind::Chunk, vec![1, 2, 3])
        .frame()
        .expect("编码");
    bytes[HEADER_LEN + 1] ^= 0x01;
    let mut decoder = Decoder::new();
    // 帧头本身是好的，所以问题要到"整帧到齐"才暴露。
    decoder.push(&bytes).expect("帧头合法");
    let error = decoder.next_frame().expect_err("校验和不符");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("校验和"), "{error}");
    assert!(decoder.next_frame().is_err(), "坏帧之后不再重同步");
}

/// 一帧一帧地反复走同一条缓冲：已经交付的字节必须被丢掉，占用不能随帧数增长。
#[test]
fn draining_leaves_no_bytes_behind() {
    let (bytes, frames) = stream();
    let mut decoder = Decoder::with_capacity(bytes.len());
    for _ in 0..100 {
        decoder.push(&bytes).expect("追加");
        assert_eq!(drain(&mut decoder), frames);
        assert_eq!(decoder.buffered(), 0);
    }
    decoder.finish().expect("正好停在帧边界");
}
