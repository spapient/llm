//! 字节码闭集的性质：变长整数往返、最小长度、游标位置纪律、CRC 检错、接收器溢出。
//!
//! 这几条都是**等号或不等号**，不掺容差：编码少写一个续位字节就成了非规范码，
//! 游标多吞一个字节后段就全错位，CRC 没能抓住单比特翻转则封口形同虚设。

use llm_wire::sink::{Sink, SliceSink};
use llm_wire::{crc, leb};
use proptest::prelude::*;
use testkit::prop;

/// 最小 LEB128 字节数：`7` 位一位，零值占一字节。
fn minimal_len(value: u64) -> usize {
    (64 - value.leading_zeros()).max(1).div_ceil(7) as usize
}

proptest! {
    #![proptest_config(prop::config())]

    /// 编码→解码是恒等，且长度是最小值：末尾补 `0x80` 续位字节或整字节零宽都会在这里露馅。
    #[test]
    fn every_u64_round_trips_at_its_minimal_length(value in any::<u64>()) {
        let mut bytes = Vec::new();
        leb::write(&mut bytes, value);
        prop_assert_eq!(bytes.len(), minimal_len(value), "{} 的编码不是最小长度", value);
        let mut cursor = leb::Cursor::new(&bytes);
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
        leb::write(&mut bytes, head);
        bytes.extend_from_slice(&tail);
        let mut cursor = leb::Cursor::new(&bytes);
        let _: u64 = cursor.read().expect("首段可读");
        prop_assert_eq!(cursor.remaining(), tail.as_slice());
        prop_assert_eq!(cursor.pos(), bytes.len() - tail.len());
    }

    /// 任意单比特翻转都必须改变 CRC（标准多项式的突发检错覆盖单比特）。
    /// 这条失败意味着校验和对某类损坏是瞎的。
    #[test]
    fn a_single_flipped_bit_always_changes_the_checksum(
        data in proptest::collection::vec(any::<u8>(), 1..=32),
        at in any::<proptest::sample::Idx>(),
        bit in 0u8..8,
    ) {
        let idx = at.idx(data.len());
        let mut flipped = data.clone();
        flipped[idx] ^= 1 << bit;
        prop_assert_ne!(
            crc::crc32(&flipped),
            crc::crc32(&data),
            "第 {} 字节第 {} 位翻转后 CRC 未变（{:?}）",
            idx,
            bit,
            data,
        );
    }

    /// 分块顺序无关：`update` 的状态是串行链，先左后右与一次喂完必须同值。
    /// 这条对任意原始状态成立——不依赖"从 `!0` 起步"这个约定。
    #[test]
    fn chunking_a_buf_does_not_change_the_state(
        state in any::<u32>(),
        data in proptest::collection::vec(any::<u8>(), 0..=32),
        at in any::<proptest::sample::Idx>(),
    ) {
        let (head, tail) = data.split_at(at.idx(data.len() + 1));
        prop_assert_eq!(crc::update(crc::update(state, head), tail), crc::update(state, &data));
    }

    /// 封口径的唯一约定：`crc32` 就是"原始状态 `!0` 起步、末尾取反"。
    /// 续封一段时取反两次还原，于是 `crc32(head ++ tail)` 可从 `crc32(head)` 接着算。
    #[test]
    fn sealing_resumes_from_a_sealed_prefix(
        data in proptest::collection::vec(any::<u8>(), 0..=32),
        at in any::<proptest::sample::Idx>(),
    ) {
        let (head, tail) = data.split_at(at.idx(data.len() + 1));
        prop_assert_eq!(crc::crc32(head), !crc::update(!0, head));
        prop_assert_eq!(crc::crc32(&data), !crc::update(!crc::crc32(head), tail));
    }

    /// 固定缓冲只报告溢出、不越界：溢出是**整块拒绝**，已经写下的那段字节不动。
    #[test]
    fn a_slice_sink_refuses_the_whole_chunk_that_does_not_fit(
        capacity in 0usize..=8,
        payload in proptest::collection::vec(any::<u8>(), 0..=16),
    ) {
        let mut storage = vec![0u8; capacity];
        let (written, overflow) = {
            let mut sink = SliceSink::new(&mut storage);
            sink.put(&payload);
            (sink.written(), sink.overflow())
        };
        prop_assert_eq!(overflow, payload.len() > capacity);
        if overflow {
            prop_assert_eq!(written, 0, "整块拒绝时不该写下一半");
        } else {
            prop_assert_eq!(written, payload.len());
            prop_assert_eq!(&storage[..written], payload.as_slice());
        }
        // 两种情况下写出位置之外都必须还是初始零：溢出不得覆写到缓冲区边界外或"半截"。
        prop_assert!(storage[written..].iter().all(|byte| *byte == 0));
    }
}
