use super::*;

#[test]
fn encoding_and_decoding_round_trip() {
    for text in ["", "hello", "héllo", "中文", "🙂 emoji"] {
        let tokens = Bytes.encode(text).expect("编码");
        assert_eq!(tokens.len(), text.len(), "字节级分词：token 数等于字节数");
        assert_eq!(Bytes.decode(&tokens).expect("解码"), text);
    }
}

#[test]
fn the_vocabulary_is_the_byte_range() {
    assert_eq!(Bytes.vocab_size(), 256);
    let error = Bytes.decode(&[Token::new(256)]).expect_err("超出词表");
    assert_eq!(error.code(), Code::Invalid);
}

/// 截断的多字节序列要让调用方看得到原因（契约第三条：拒绝必须有话说）。
#[test]
fn an_incomplete_utf8_sequence_is_refused_with_a_message() {
    let tokens = Bytes.encode("中").expect("编码");
    let truncated = &tokens[..1];
    let error = Bytes
        .decode(truncated)
        .expect_err("截断的多字节序列必须报错");
    assert_eq!(error.code(), Code::Invalid);
    assert!(
        !error.message().trim().is_empty(),
        "拒绝不能说空话：{error}"
    );
}
