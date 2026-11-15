//! 黑盒契约：分词器是一层 trait，"换实现不影响上层"的承诺只能由公共面验证。
//!
//! 内联测试钉住 `Bytes` 的逐字节映射；这里验证跨实现的那部分契约：往返恒等、词表边界、
//! 非法 UTF-8 与越界编号必须报错而不是替换，以及 `vocab_size` 与可接受编号范围一致。

use llm_tokenizer::{Bytes, Token, Tokenizer};

#[test]
fn text_round_trips_through_the_reference_tokenizer() {
    // 覆盖 ASCII、多字节 UTF-8、空串与需要组合的边界文本。
    let texts = ["", "a", "hé", "中文", "🙂", "line\nbreak\ttab"];
    for text in texts {
        let tokens = Bytes.encode(text).expect("任何 &str 都能编码");
        assert_eq!(
            tokens.len(),
            text.len(),
            "字节级实现的 token 数等于字节数，这是它的性能基线口径"
        );
        assert_eq!(Bytes.decode(&tokens).expect("自产 token 必能解码"), text);
    }
}

#[test]
fn every_encodable_token_is_inside_the_declared_vocabulary() {
    let text = "混合 ascii 与中文 🙂";
    for token in Bytes.encode(text).expect("编码") {
        assert!(
            (token.id() as usize) < Bytes.vocab_size(),
            "编码产出的编号必须落在 vocab_size 内，否则解码方必然越界"
        );
    }
}

#[test]
fn ids_outside_the_vocabulary_are_rejected_not_wrapped() {
    let error = Bytes
        .decode(&[Token::new(256)])
        .expect_err("越界编号必须拒绝");
    assert_eq!(error.code().as_str(), "invalid", "错误类别是对外契约");
    assert!(
        error.message().contains("256"),
        "拒绝时要把越界的编号说出来：{}",
        error.message()
    );
}

#[test]
fn a_byte_sequence_that_is_not_utf8_is_rejected_with_its_cause() {
    // 0xE4 是三个字节序列的开头，单独出现不构成合法 UTF-8。
    let error = Bytes
        .decode(&[Token::new(0xE4)])
        .expect_err("半个字符必须拒绝");
    assert_eq!(error.code().as_str(), "invalid");
    assert!(!error.message().trim().is_empty());
}

#[test]
fn an_empty_sequence_decodes_to_the_empty_string() {
    assert_eq!(
        Bytes.decode(&[]).expect("空序列合法"),
        "",
        "空输入不是错误：它对应空提示"
    );
}
