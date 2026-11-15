//! 属性骨架：随机字符串要么命中闭集并按规范名往返，要么被拒绝；解析路径不得 panic。
//!
//! 闭集内每个变体的下标/规范名/`Display`/`FromStr` 双向契约由 `identity.rs` 逐条锁定，
//! 这里只覆盖不受信输入分布：随机拼出的名字即使命中，也必须能从规范名解析回同一个变体。

use llm_dtype::DType;

#[test]
fn random_names_either_round_trip_or_are_rejected() {
    let mut rng = testkit::Rng::new(0xD1_7E);
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789_";
    // 用例串行生成：下一个字符依赖上一次 PRNG 状态；不变量是命中即往返、未命中即 `None`。
    for _ in 0..128 {
        let len = rng.below(12) as usize;
        let text: String = (0..len)
            .map(|_| CHARSET[rng.below(CHARSET.len() as u32) as usize] as char)
            .collect();
        if let Some(value) = DType::parse_name(&text) {
            assert_eq!(value.name(), text, "命中的名字必须按规范名往返");
            assert_eq!(DType::parse_name(value.name()), Some(value));
        }
    }
}
