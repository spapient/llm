use super::*;
use diagnostic::Code;

fn envelope() -> Envelope {
    Envelope::new(Version::current(), Kind::Chunk, b"payload".to_vec())
}

#[test]
fn framing_round_trips() {
    let frame = envelope().frame().expect("编码");
    assert_eq!(frame.len(), HEADER_LEN + 7 + TRAILER_LEN);
    assert_eq!(Envelope::parse(&frame).expect("解析"), envelope());
}

#[test]
fn an_empty_payload_is_legal() {
    let empty = Envelope::new(Version::current(), Kind::Done, Vec::new());
    let frame = empty.frame().expect("编码");
    assert_eq!(frame.len(), HEADER_LEN + TRAILER_LEN);
    assert_eq!(Envelope::parse(&frame).expect("解析").payload(), b"");
}

/// 字节布局是对端契约：帧头一个字段都不能挪，校验尾只在第 2 版及以后存在。
#[test]
fn the_wire_layout_is_pinned() {
    let first = Envelope::new(Version::FIRST, Kind::Chunk, vec![1, 2, 3])
        .frame()
        .expect("编码");
    assert_eq!(
        first,
        [
            0x4c, 0x4c, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00, 1, 2, 3
        ]
    );

    let current = Envelope::new(Version::current(), Kind::Chunk, vec![1, 2, 3])
        .frame()
        .expect("编码");
    assert_eq!(
        &current[4..first.len()],
        &first[4..],
        "版本之后逐字节相同：类型、长度与载荷都没有挪位置"
    );
    assert_eq!(
        &current[first.len()..],
        &crate::crc::crc32(&current[..first.len()]).to_le_bytes()
    );
}

#[test]
fn a_frame_keeps_the_version_it_was_written_with() {
    for version in [Version::FIRST, Version::current()] {
        let envelope = Envelope::new(version, Kind::Done, vec![9]);
        let frame = envelope.frame().expect("编码");
        let trailer = if version == Version::FIRST {
            0
        } else {
            TRAILER_LEN
        };
        assert_eq!(frame.len(), HEADER_LEN + 1 + trailer);
        let parsed = Envelope::parse(&frame).expect("解析");
        assert_eq!(parsed.version(), version, "版本是往返的一部分");
        assert_eq!(parsed, envelope);
    }
}

/// 帧头里的长度来自对端：16 进制的极限值不能变成算术溢出，也不能变成一次大分配。
#[test]
fn an_absurd_declared_length_is_an_error_not_an_overflow() {
    let mut frame = envelope().frame().expect("编码");
    frame[6..10].copy_from_slice(&u32::MAX.to_le_bytes());
    let error = Envelope::parse(&frame).expect_err("必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("上限"), "{error}");
}

/// 一个缓冲里有多帧时只解析第一帧，剩下的交给调用方。
#[test]
fn trailing_bytes_belong_to_the_next_frame() {
    let first = envelope().frame().expect("编码");
    let second = Envelope::new(Version::current(), Kind::Done, b"bye".to_vec())
        .frame()
        .expect("编码");
    let mut stream = first.clone();
    stream.extend_from_slice(&second);

    assert_eq!(Envelope::parse(&stream).expect("只解析第一帧"), envelope());
    let (parsed, rest) = Envelope::parse_rest(&stream).expect("第一帧");
    assert_eq!(parsed, envelope());
    assert_eq!(rest, &second[..]);

    // 借用的那条路必须与拥有的那条路给出同一个结论，连"剩下多少"都一致。
    let (view, view_rest) = Envelope::view(&stream).expect("第一帧");
    assert_eq!(view.envelope(), parsed);
    assert_eq!(view_rest, rest);
    assert_eq!(view.payload(), b"payload");
}

/// 一次编码失败不该在流里留下半帧：调用方的缓冲原样不动。
#[test]
fn a_failed_encode_leaves_the_buffer_untouched() {
    let mut out = b"prefix".to_vec();
    let huge = Envelope::new(Version::current(), Kind::Chunk, vec![0; MAX_PAYLOAD + 1]);
    assert_eq!(huge.frame().expect_err("超限").code(), Code::Invalid);
    assert!(huge.encode_into(&mut out).is_err());
    assert_eq!(out, b"prefix");
}

#[test]
fn encode_into_appends_without_disturbing_earlier_frames() {
    let first = envelope();
    let second = Envelope::new(Version::FIRST, Kind::Done, b"bye".to_vec());
    let mut stream = Vec::new();
    first.encode_into(&mut stream).expect("编码");
    second.encode_into(&mut stream).expect("编码");
    assert_eq!(
        stream.len(),
        first.encoded_len().expect("长度") + second.encoded_len().expect("长度")
    );
    assert_eq!(
        &stream[first.encoded_len().expect("长度")..],
        &second.frame().expect("编码")[..]
    );
}

#[test]
fn a_wrong_magic_is_rejected() {
    let mut frame = envelope().frame().expect("编码");
    frame[0] = 0xff;
    assert_eq!(Envelope::parse(&frame).unwrap_err().code(), Code::Invalid);
}

#[test]
fn a_future_version_is_unsupported_not_invalid() {
    let mut frame = Envelope::new(Version::FIRST, Kind::Request, Vec::new())
        .frame()
        .expect("编码");
    let future = Version::current().get() + 1;
    frame[2..4].copy_from_slice(&future.to_le_bytes());
    assert_eq!(
        Envelope::parse(&frame).unwrap_err().code(),
        Code::Unsupported
    );
}

/// 第 0 版不是"将来"，是本端或对端的 bug：类别必须与"版本过新"分开。
#[test]
fn version_zero_is_invalid_and_cannot_be_written() {
    assert_eq!(
        Envelope::new(Version::new(0), Kind::Request, Vec::new())
            .frame()
            .expect_err("本端写不出版本 0")
            .code(),
        Code::Invalid
    );

    let mut frame = Envelope::new(Version::FIRST, Kind::Request, Vec::new())
        .frame()
        .expect("编码");
    frame[2..4].copy_from_slice(&0u16.to_le_bytes());
    assert_eq!(Envelope::parse(&frame).unwrap_err().code(), Code::Invalid);
}

/// 载荷的每一位都被校验和覆盖：长度对得上不等于内容对得上。
#[test]
fn a_flipped_payload_bit_fails_the_checksum() {
    for index in [HEADER_LEN, HEADER_LEN + 3] {
        let mut frame = envelope().frame().expect("编码");
        frame[index] ^= 0x01;
        let error = Envelope::parse(&frame).expect_err("必须拒绝");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("校验和"), "{error}");
    }
}

/// 帧头也在校验范围内，而帧头里唯一“改成另一个值也合法”的字段是类型：
/// 把 `chunk` 改成 `error` 之后帧头检查全部通过，只有校验和能发现它。
#[test]
fn a_tampered_type_is_caught_only_by_the_checksum() {
    let mut frame = envelope().frame().expect("编码");
    assert_eq!(frame[4], 2, "第 5 字节是 `Kind::Chunk` 的线上编码");
    frame[4] = 3;
    assert_eq!(
        Kind::from_code(3),
        Some(Kind::Error),
        "改后的值本身是合法类型"
    );
    let error = Envelope::parse(&frame).expect_err("必须拒绝");
    assert!(error.to_string().contains("校验和"), "{error}");
}

/// 第 1 版没有校验尾：同一个载荷的第 1 版帧必须照旧能读，不能因为"本端升级了"就拒收。
#[test]
fn a_first_version_frame_still_parses_after_the_format_grew_a_trailer() {
    let old = Envelope::new(Version::FIRST, Kind::Error, b"gone".to_vec())
        .frame()
        .expect("编码");
    let parsed = Envelope::parse(&old).expect("旧版本照旧可读");
    assert_eq!(parsed.version(), Version::FIRST);
    assert_eq!(parsed.payload(), b"gone");
}

#[test]
fn truncated_frames_and_short_payloads_are_rejected() {
    assert_eq!(
        Envelope::parse(&[0u8; 4]).unwrap_err().code(),
        Code::Invalid
    );
    let frame = envelope().frame().expect("编码");
    for cut in [HEADER_LEN, frame.len() - 1] {
        assert_eq!(
            Envelope::parse(&frame[..cut]).unwrap_err().code(),
            Code::Invalid,
            "切到 {cut} 字节必须被拒"
        );
    }
}

#[test]
fn kind_codes_are_a_closed_set() {
    // 字面锁定线上编码：`ALL` 顺序是唯一会被手改的地方，改了这里必须显式改期望。
    assert_eq!(
        Kind::ALL.iter().map(|kind| kind.code()).collect::<Vec<_>>(),
        [1, 2, 3, 4],
        "线上编码是对端契约，不允许随枚举顺序漂移"
    );
    for &kind in Kind::ALL {
        assert_eq!(Kind::from_code(kind.code()), Some(kind));
        assert!(!kind.as_str().is_empty());
        assert_eq!(kind.to_string(), kind.as_str());
    }
    for code in 0..=u16::MAX {
        assert_eq!(
            Kind::from_code(code).is_some(),
            Kind::ALL.iter().any(|kind| kind.code() == code),
            "`from_code({code})` 与线上编码表不一致"
        );
    }
}

#[test]
fn a_version_knows_which_side_of_the_range_it_is_on() {
    assert!(!Version::new(0).is_supported());
    assert!(Version::FIRST.is_supported());
    assert!(Version::current().is_supported());
    assert!(!Version::new(Version::current().get() + 1).is_supported());
    assert_eq!(Version::CURRENT.to_string(), "v2");
}
