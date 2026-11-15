use super::*;
use testkit::crc32_bitwise;

#[test]
fn the_table_agrees_with_the_bitwise_definition() {
    let data: Vec<u8> = (0u8..=200).collect();
    assert_eq!(crc32(&data), crc32_bitwise(&data));
    assert_eq!(crc32(b""), crc32_bitwise(b""));
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926, "标准 check 值");
}

#[test]
fn chunked_feeds_match_one_shot() {
    let data: Vec<u8> = (0u8..=250).collect();
    let whole = crc32(&data);
    let mut state = !0u32;
    for chunk in data.chunks(7) {
        state = update(state, chunk);
    }
    assert_eq!(!state, whole);
}
