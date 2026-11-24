use super::*;
use proptest::prelude::*;
use testkit::prop;

#[test]
fn the_table_agrees_with_the_bitwise_definition() {
    testkit::assert_table_matches_bitwise(crc32);
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

proptest! {
    #![proptest_config(prop::config())]

    /// 任意单比特翻转都必须改变 CRC（标准多项式的突发检错覆盖单比特）；失败意味着校验和对某类损坏是瞎的。
    #[test]
    fn a_single_flipped_bit_always_changes_the_checksum(
        data in proptest::collection::vec(any::<u8>(), 1..=32),
        at in any::<proptest::sample::Index>(),
        bit in 0u8..8,
    ) {
        let idx = at.index(data.len());
        let mut flipped = data.clone();
        flipped[idx] ^= 1 << bit;
        prop_assert_ne!(
            crc32(&flipped),
            crc32(&data),
            "第 {} 字节第 {} 位翻转后 CRC 未变（{:?}）",
            idx,
            bit,
            data,
        );
    }

    /// 分块顺序无关：`update` 的状态是串行链，先左后右与一次喂完必须同值；对任意原始状态成立，不依赖从 `!0` 起步。
    #[test]
    fn chunking_a_buf_does_not_change_the_state(
        state in any::<u32>(),
        data in proptest::collection::vec(any::<u8>(), 0..=32),
        at in any::<proptest::sample::Index>(),
    ) {
        let (head, tail) = data.split_at(at.index(data.len() + 1));
        prop_assert_eq!(update(update(state, head), tail), update(state, &data));
    }

    /// 封口约定：`crc32` 就是"原始状态 `!0` 起步、末尾取反"；续封一段时取反两次还原，可从前缀校验和接着算。
    #[test]
    fn sealing_resumes_from_a_sealed_prefix(
        data in proptest::collection::vec(any::<u8>(), 0..=32),
        at in any::<proptest::sample::Index>(),
    ) {
        let (head, tail) = data.split_at(at.index(data.len() + 1));
        prop_assert_eq!(crc32(head), !update(!0, head));
        prop_assert_eq!(crc32(&data), !update(!crc32(head), tail));
    }
}
