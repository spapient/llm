//! 帧编解码的性质：往返恒等、尾部纪律、CRC 单比特检错、截断必报错。
//!
//! 这几条都是**等号或不等号**：编码少写一段校验尾、解码多吞一个字节、
//! CRC 对某类损坏是瞎的，都会在这里露馅——帧是流式协议的封口，封口松一寸，
//! 后面整条连接就全错位。

use llm_serve::proto::{Envelope, Kind, Version};
use proptest::prelude::*;
use testkit::prop;

/// 本端支持的两个版本：第 1 版无校验尾，第 2 版有。
fn version() -> impl Strategy<Value = Version> {
    prop_oneof![Just(Version::FIRST), Just(Version::current())]
}

/// 闭集里的四种帧类型。
fn kind() -> impl Strategy<Value = Kind> {
    prop_oneof![
        Just(Kind::Request),
        Just(Kind::Chunk),
        Just(Kind::Error),
        Just(Kind::Done),
    ]
}

proptest! {
    #![proptest_config(prop::config())]

    /// 编码→解码是恒等，且声明长度与实际字节数一致：`encoded_len` 是调用方
    /// 预分配缓冲的依据，算错一个字节就会在复用处露馅。
    #[test]
    fn every_frame_round_trips(
        version in version(),
        kind in kind(),
        payload in proptest::collection::vec(any::<u8>(), 0..=2048),
    ) {
        let envelope = Envelope::new(version, kind, payload);
        let bytes = envelope.frame().expect("合法帧必须可编码");
        prop_assert_eq!(bytes.len(), envelope.encoded_len().expect("长度可算"));
        let (decoded, rest) = Envelope::parse_rest(&bytes).expect("自己写出的帧必须可读");
        prop_assert_eq!(decoded, envelope);
        prop_assert!(rest.is_empty(), "解码后应当正好读空");
    }

    /// 尾部纪律：读完一帧，`parse_rest` 交还的恰是后续字节——既不吞一个，也不回退。
    /// 两帧相连时按序各读各的，流式解码的位置契约全在这条上。
    #[test]
    fn a_frame_hands_the_tail_back_untouched(
        version in version(),
        kind in kind(),
        payload in proptest::collection::vec(any::<u8>(), 0..=512),
        tail in proptest::collection::vec(any::<u8>(), 0..=64),
    ) {
        let envelope = Envelope::new(version, kind, payload);
        let mut bytes = envelope.frame().expect("可编码");
        bytes.extend_from_slice(&tail);
        let (decoded, rest) = Envelope::parse_rest(&bytes).expect("可读");
        prop_assert_eq!(decoded, envelope);
        prop_assert_eq!(rest, tail.as_slice());
    }

    /// 第 2 版的校验范围是帧头 + 载荷 + 校验尾自身：任意单比特翻转都必须被拒绝。
    /// 这条失败意味着 CRC 对某类损坏是瞎的，封口形同虚设。
    #[test]
    fn a_single_flipped_bit_always_fails_the_checksum(
        kind in kind(),
        payload in proptest::collection::vec(any::<u8>(), 0..=512),
        at in any::<proptest::sample::Index>(),
        bit in 0u8..8,
    ) {
        let bytes = Envelope::new(Version::current(), kind, payload)
            .frame()
            .expect("可编码");
        let idx = at.index(bytes.len());
        let mut flipped = bytes.clone();
        flipped[idx] ^= 1 << bit;
        prop_assert!(
            Envelope::parse(&flipped).is_err(),
            "第 {} 字节第 {} 位翻转后仍被接受（{:?}）",
            idx, bit, flipped,
        );
    }

    /// 真前缀必报错：帧头没到齐、载荷没到齐都不能当成一帧——把"还没到齐"
    /// 混进成功里，流式解码就会把半帧当整帧发出去。
    #[test]
    fn every_proper_prefix_is_rejected(
        version in version(),
        kind in kind(),
        payload in proptest::collection::vec(any::<u8>(), 0..=512),
        cut in any::<proptest::sample::Index>(),
    ) {
        let bytes = Envelope::new(version, kind, payload)
            .frame()
            .expect("可编码");
        let len = cut.index(bytes.len());
        prop_assert!(
            Envelope::parse(&bytes[..len]).is_err(),
            "前 {} 字节（整帧 {}）被当成了整帧",
            len, bytes.len(),
        );
    }
}
