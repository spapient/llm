//! 黑盒契约：分词器是一层 trait，"换实现不影响上层"的承诺只能由公共面验证。
//!
//! 内联测试钉住 `Bytes` 的逐字节映射；这里验证跨实现的那部分契约：往返恒等、词表边界、
//! 非法 UTF-8 与越界编号必须报错而不是替换，以及 `vocab_size` 与可接受编号范围一致。

use llm_tokenizer::{
    Bpe, Bytes, DecoderStream, Merge, Token, Tokenizer, Unigram, Vocab, VocabFamily, VocabSource,
    fixtures::unigram_source,
};

/// 合成 BPE 词表：`a b c` 单字节 + `ab`/`bc` 合并片；秩按 merges 表内顺序。
fn bpe_fixture() -> Bpe {
    let source = VocabSource {
        family: VocabFamily::Bpe,
        pieces: vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"c".to_vec(),
            b"ab".to_vec(),
            b"bc".to_vec(),
        ],
        merges: vec![
            Merge {
                left: b"a".to_vec(),
                right: b"b".to_vec(),
            },
            Merge {
                left: b"b".to_vec(),
                right: b"c".to_vec(),
            },
        ],
        ..VocabSource::default()
    };
    let vocab = Vocab::try_new(source).expect("合成 BPE 词表构造");
    Bpe::new(vocab).expect("合成 BPE 实例")
}

/// 合成 Unigram 词表：单字母 -1.0，`ab` -0.1，`abc` -2.0，`<unk>` 编号 5。
fn unigram_fixture() -> Unigram {
    let vocab = Vocab::try_new(unigram_source(true)).expect("合成 Unigram 词表构造");
    Unigram::new(vocab).expect("合成 Unigram 实例")
}

/// 编码产出的编号必须全部落在声明词表内——这是解码方不越界的共同前提。
fn contract_id_range(tokenizer: &dyn Tokenizer, text: &str) {
    for token in tokenizer.encode(text).expect("编码") {
        assert!(
            (token.id() as usize) < tokenizer.vocab_size(),
            "编号 {} 超出词表（0..{}）",
            token.id(),
            tokenizer.vocab_size()
        );
    }
}

/// 越界编号必须拒绝，且错误类别对外固定为 `invalid`。
fn contract_out_of_range_rejected(tokenizer: &dyn Tokenizer) {
    let bad = Token::new(tokenizer.vocab_size() as u32);
    let error = tokenizer
        .decode(&[bad])
        .expect_err("词表外编号必须拒绝，不能回绕");
    assert_eq!(error.code().as_str(), "invalid");
}

/// 空序列解码为空串，两个实现都不能把空输入当错误。
fn contract_empty_round_trip(tokenizer: &dyn Tokenizer) {
    assert_eq!(tokenizer.decode(&[]).expect("空序列合法"), "");
}

/// 计数必须与完整编码等长（默认计数路径只是不另猜）。
fn contract_count_matches(tokenizer: &dyn Tokenizer, text: &str) {
    let encoded = tokenizer.encode(text).expect("编码");
    assert_eq!(tokenizer.count(text).expect("计数"), encoded.len());
}

/// 可编码文本往返恒等；语料由各实现给出（只含词表内片段）。
fn contract_round_trip(tokenizer: &dyn Tokenizer, text: &str) {
    let tokens = tokenizer.encode(text).expect("编码");
    assert_eq!(
        tokenizer.decode(&tokens).expect("自产 token 必能解码"),
        text
    );
}

#[test]
fn contract_id_range_holds_for_both_impls() {
    let bpe = bpe_fixture();
    let unigram = unigram_fixture();
    for text in ["", "a", "hé", "中文", "🙂"] {
        contract_id_range(&Bytes, text);
    }
    for text in ["abc", "bc", "cab", ""] {
        contract_id_range(&bpe, text);
    }
    for text in ["abc", "ab", "a", "b", "c", ""] {
        contract_id_range(&unigram, text);
    }
}

#[test]
fn contract_out_of_range_rejected_for_both_impls() {
    let bpe = bpe_fixture();
    let unigram = unigram_fixture();
    contract_out_of_range_rejected(&Bytes);
    contract_out_of_range_rejected(&bpe);
    contract_out_of_range_rejected(&unigram);
}

#[test]
fn contract_empty_sequence_for_both_impls() {
    let bpe = bpe_fixture();
    let unigram = unigram_fixture();
    contract_empty_round_trip(&Bytes);
    contract_empty_round_trip(&bpe);
    contract_empty_round_trip(&unigram);
}

#[test]
fn contract_count_matches_for_both_impls() {
    let bpe = bpe_fixture();
    let unigram = unigram_fixture();
    for text in ["", "a", "hé", "中文", "🙂"] {
        contract_count_matches(&Bytes, text);
    }
    for text in ["abc", "bc", "cab", ""] {
        contract_count_matches(&bpe, text);
    }
    for text in ["abc", "ab", "a", "b", "c", ""] {
        contract_count_matches(&unigram, text);
    }
}

#[test]
fn contract_round_trip_for_encodable_texts() {
    let bpe = bpe_fixture();
    let unigram = unigram_fixture();
    for text in [
        "",
        "a",
        "hé",
        "中文",
        "🙂",
        "line\nbreak\ttab",
        "  空格  重复",
        "混合abc中文🙂",
    ] {
        contract_round_trip(&Bytes, text);
    }
    for text in ["abc", "bc", "cab", ""] {
        contract_round_trip(&bpe, text);
    }
    for text in ["abc", "ab", "a", "b", "c", ""] {
        contract_round_trip(&unigram, text);
    }
}

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

#[test]
fn fingerprint_identifies_the_byte_vocabulary() {
    let fingerprint = Bytes.fingerprint();
    assert_eq!(fingerprint.vocab_size(), Bytes.vocab_size());
    assert_eq!(fingerprint.family(), "bytes");
    assert_eq!(fingerprint.hex().len(), 16);
}

#[test]
fn count_matches_the_encoded_length_without_a_second_guess() {
    for text in ["", "a", "hé", "中文", "🙂"] {
        let encoded = Bytes.encode(text).expect("编码");
        assert_eq!(Bytes.count(text).expect("计数"), encoded.len());
    }
}

#[test]
fn detailed_encoding_offsets_line_up_with_bytes() {
    let encoding = Bytes.encode_detailed("ab").expect("详细编码");
    assert_eq!(encoding.ids().len(), encoding.offsets().len());
    assert_eq!(encoding.offsets(), &[(0, 1), (1, 2)]);
    assert!(
        encoding
            .ids()
            .iter()
            .all(|token| token.id() < Bytes.vocab_size() as u32)
    );
}

#[test]
fn reverse_lookup_round_trips_single_bytes() {
    let token = Bytes.lookup(b"Z").expect("查表").expect("命中");
    assert_eq!(token.id(), u32::from(b'Z'));
    assert_eq!(Bytes.piece(token).expect("反查"), Some(vec![b'Z']));
}

#[test]
fn stream_decodes_generated_tokens_one_chunk_at_a_time() {
    let mut stream = DecoderStream::new();
    // 生成侧逐块吐字节；多字节字符跨块时缓冲负责拼合。
    assert_eq!(stream.push(&[0xE4]).expect("首块"), "");
    assert_eq!(stream.push(&[0xB8, 0xAD]).expect("次块"), "中");
    assert_eq!(stream.finish().expect("收尾"), "");
}

#[test]
fn baseline_has_no_special_tokens() {
    let special = Bytes.special_tokens();
    assert!(special.is_empty());
    assert!(!special.is_special(0));
}
