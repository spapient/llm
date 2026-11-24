use super::*;
use proptest::prelude::*;
use testkit::prop;

#[test]
fn integers_round_trip_across_boundaries() {
    for value in [
        0u64,
        1,
        127,
        128,
        16_383,
        16_384,
        u64::from(u32::MAX),
        u64::MAX,
    ] {
        let mut buf = Vec::new();
        write(&mut buf, value);
        assert_eq!(Cursor::new(&buf).read::<u64>().expect("规范"), value);
    }
}

#[test]
fn every_unsigned_width_encodes() {
    let mut buf = Vec::new();
    write(&mut buf, 7u8);
    write(&mut buf, 300u16);
    write(&mut buf, 40_000u32);
    write(&mut buf, 9_000_000_000u64);
    write(&mut buf, 5usize);
    let mut cur = Cursor::new(&buf);
    assert_eq!(cur.read::<u64>().expect("u8"), 7);
    assert_eq!(cur.read::<u64>().expect("u16"), 300);
    assert_eq!(cur.read::<u32>().expect("u32"), 40_000);
    assert_eq!(cur.read::<u64>().expect("u64"), 9_000_000_000);
    assert_eq!(cur.read::<u64>().expect("usize"), 5);
}

#[test]
fn a_padded_zero_is_rejected() {
    let error = Cursor::new(&[0x80, 0x00])
        .read::<u64>()
        .expect_err("非规范");
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn an_overflowing_sequence_is_rejected() {
    let mut seq = vec![0x80u8; 9];
    seq.push(0x02); // 第 10 字节用了第 2 位，超出 u64
    assert!(Cursor::new(&seq).read::<u64>().is_err());
    assert!(Cursor::new(&[0x80u8; 11]).read::<u64>().is_err());
}

#[test]
fn taking_past_the_end_is_an_error() {
    let mut cursor = Cursor::new(&[1, 2, 3]);
    assert_eq!(cursor.take(2).expect("可取"), &[1, 2]);
    assert_eq!(cursor.pos(), 2);
    assert!(cursor.take(2).is_err());
}

/// 属性骨架：随机 `u64` 编码后必须规范解码回原值。
///
/// 用例串行生成——下一个输入依赖上一次 PRNG 状态，无法预生成批量；不变量是往返一致与
/// 字节数上界，边界值（0、127、128、u32::MAX、u64::MAX）由 `integers_round_trip_across_boundaries`
/// 单独锁定，这里只覆盖中间分布。
#[test]
fn random_u64_values_round_trip_through_canonical_encoding() {
    let mut rng = testkit::Rng::new(0x1E_B12E);
    for _ in 0..64 {
        let value = (u64::from(rng.below(u32::MAX)) << 32) | u64::from(rng.below(u32::MAX));
        let mut buf = Vec::new();
        write(&mut buf, value);
        assert!(buf.len() <= 10, "u64 的 LEB128 最多 10 字节");
        let decoded: u64 = Cursor::new(&buf).read().expect("随机值必为规范编码");
        assert_eq!(decoded, value);
    }
}

/// 最小 LEB128 字节数：`7` 位一位，零值占一字节。
fn minimal_len(value: u64) -> usize {
    (64 - value.leading_zeros()).max(1).div_ceil(7) as usize
}

#[test]
fn fixed_width_le_reads_and_random_access() {
    // 顺序：word(4) + wide(8) + uint(2) + text(3)，小端拼装与位置纪律。
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0x0102_0304u32.to_le_bytes());
    bytes.extend_from_slice(&0x0102_0304_0506_0708u64.to_le_bytes());
    bytes.extend_from_slice(&0xABCDu16.to_le_bytes());
    bytes.extend_from_slice(b"abc");
    let mut cur = Cursor::new(&bytes);
    assert_eq!(cur.total(), bytes.len());
    assert_eq!(cur.word().unwrap(), 0x0102_0304);
    assert_eq!(cur.wide().unwrap(), 0x0102_0304_0506_0708);
    assert_eq!(cur.uint(2).unwrap(), 0xABCD);
    assert_eq!(cur.text(3).unwrap(), "abc");
    assert!(cur.remaining().is_empty());
    assert!(Cursor::new(&bytes).text(99).is_err(), "越界文本必须失败");

    // 随机访问：偏移 + 宽度判据，溢出与越界都显式失败。
    let raw = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    assert_eq!(u16_at(&raw, 0).unwrap(), 0x2211);
    assert_eq!(u32_at(&raw, 0).unwrap(), 0x4433_2211);
    assert_eq!(u64_at(&raw, 0).unwrap(), 0x8877_6655_4433_2211);
    assert_eq!(slice(&raw, 2, 3).unwrap(), &[0x33, 0x44, 0x55]);
    assert!(slice(&raw, usize::MAX, 4).is_err(), "偏移溢出必须失败");
    assert!(u32_at(&raw, 6).is_err(), "越界必须失败");
}

proptest! {
    #![proptest_config(prop::config())]

    /// 编码→解码是恒等，且长度是最小值：末尾补 `0x80` 续位字节或整字节零宽都会在这里露馅。
    #[test]
    fn every_u64_round_trips_at_its_minimal_length(value in any::<u64>()) {
        let mut bytes = Vec::new();
        write(&mut bytes, value);
        prop_assert_eq!(bytes.len(), minimal_len(value), "{} 的编码不是最小长度", value);
        let mut cursor = Cursor::new(&bytes);
        let decoded: u64 = cursor.read().expect("自己写出的编码必须可读");
        prop_assert_eq!(decoded, value);
        prop_assert!(cursor.remaining().is_empty(), "解码后应当正好读空");
    }

    /// 游标位置纪律：读完前一段，`remaining` 恰是后一段——既不吞一个字节，也不回退。
    #[test]
    fn a_cursor_hands_the_tail_back_untouched(
        head in any::<u64>(),
        tail in proptest::collection::vec(any::<u8>(), 0..=8),
    ) {
        let mut bytes = Vec::new();
        write(&mut bytes, head);
        bytes.extend_from_slice(&tail);
        let mut cursor = Cursor::new(&bytes);
        let _: u64 = cursor.read().expect("首段可读");
        prop_assert_eq!(cursor.remaining(), tail.as_slice());
        prop_assert_eq!(cursor.pos(), bytes.len() - tail.len());
    }
}
