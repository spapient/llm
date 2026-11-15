use super::*;
use diagnostic::Code;

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

#[test]
fn special_tokens_register_and_match_ids() {
    let special = SpecialTokens::empty().with_bos(1).with_eos(2).with_unk(0);
    assert_eq!(special.len(), 3);
    assert!(!special.is_empty());
    assert!(special.is_special(1));
    assert!(special.is_special(2));
    assert!(special.is_special(0));
    assert!(!special.is_special(7));
    assert_eq!(special.bos(), Some(Token::new(1)));
    assert_eq!(special.eos(), Some(Token::new(2)));
    assert_eq!(special.pad(), None);
}

#[test]
fn encoding_rejects_mismatched_metadata_lengths() {
    let ids = vec![Token::new(1), Token::new(2)];
    let offsets = vec![(0, 1)];
    let special = vec![false, true];
    let error = Encoding::new(ids, offsets, special).expect_err("长度不齐必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn detailed_encoding_carries_byte_offsets() {
    let encoding = Bytes.encode_detailed("hé").expect("详细编码");
    assert_eq!(encoding.len(), "hé".len());
    assert_eq!(encoding.offsets(), &[(0, 1), (1, 2), (2, 3)]);
    assert_eq!(encoding.is_special(0), Some(false));
    assert!(encoding.offsets().len() == encoding.len());
}

#[test]
fn count_is_zero_allocation_length_for_bytes() {
    assert_eq!(Bytes.count("中文🙂").expect("计数"), "中文🙂".len());
}

#[test]
fn lookup_and_piece_are_inverse_for_single_bytes() {
    for byte in [0u8, 65, 255] {
        let token = Bytes.lookup(&[byte]).expect("查表").expect("单字节必命中");
        assert_eq!(token.id(), u32::from(byte));
        assert_eq!(Bytes.piece(token).expect("反查"), Some(vec![byte]));
    }
    assert!(Bytes.lookup(&[1, 2]).expect("多字节返回空").is_none());
}

#[test]
fn fingerprint_is_stable_for_a_descriptor() {
    let first = Fingerprint::new("bytes", 256, "one-token-per-byte");
    let second = Fingerprint::new("bytes", 256, "one-token-per-byte");
    assert_eq!(first, second);
    assert_eq!(first.hex().len(), 16);
    assert_eq!(first.family(), "bytes");
    assert_eq!(first.vocab_size(), 256);

    let other = Fingerprint::new("bytes", 256, "different");
    assert_ne!(first.digest(), other.digest());
}

#[test]
fn padding_rejects_zero_length() {
    let error = Padding::new(0, 0, PadDirection::Right).expect_err("零长度填充必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn strategy_defaults_to_no_op() {
    let strategy = Strategy::default();
    assert!(!strategy.add_bos());
    assert!(!strategy.add_eos());
    assert!(!strategy.add_sep());
    assert_eq!(strategy.max_len(), None);
    assert_eq!(strategy.truncation(), None);
    assert!(strategy.padding().is_none());

    let padded = Strategy::no_op()
        .with_max_len(128)
        .with_truncation(Truncation::Right)
        .with_padding(Padding::new(128, 0, PadDirection::Left).expect("填充配置"));
    assert_eq!(padded.max_len(), Some(128));
    assert_eq!(padded.truncation(), Some(Truncation::Right));
    assert!(padded.padding().is_some());
}

#[test]
fn strategy_sep_flag_is_off_by_default() {
    assert!(!Strategy::no_op().add_sep());
    assert!(Strategy::no_op().with_sep().add_sep());
}

fn word_level(pieces: Vec<&[u8]>) -> VocabSource {
    VocabSource {
        family: VocabFamily::WordLevel,
        pieces: pieces
            .into_iter()
            .map(|piece: &[u8]| piece.to_vec())
            .collect(),
        ..VocabSource::default()
    }
}

#[test]
fn vocab_builds_bidirectional_lookup() {
    let special = SpecialTokens::EMPTY.with_bos(0);
    let source = VocabSource {
        special,
        ..word_level(vec!["a".as_bytes(), "bc".as_bytes(), "中".as_bytes()])
    };
    let vocab = Vocab::try_new(source).expect("词表构造");

    assert_eq!(vocab.vocab_size(), 3);
    assert_eq!(vocab.family(), VocabFamily::WordLevel);
    assert_eq!(vocab.lookup(b"bc"), Some(Token::new(1)));
    assert_eq!(vocab.piece(2), Some("中".as_bytes()));
    assert!(vocab.lookup(b"missing").is_none());
    assert_eq!(vocab.fingerprint().vocab_size(), 3);
    assert!(vocab.is_special(0));
}

#[test]
fn vocab_rejects_duplicate_piece() {
    let source = word_level(vec!["a".as_bytes(), "a".as_bytes()]);
    let error = Vocab::try_new(source).expect_err("重复词片必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn vocab_rejects_special_id_out_of_range() {
    let special = SpecialTokens::EMPTY.with_bos(9);
    let source = VocabSource {
        special,
        ..word_level(vec!["a".as_bytes()])
    };
    let error = Vocab::try_new(source).expect_err("越界特殊编号必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn bpe_requires_a_merge_table_and_others_reject_one() {
    let empty = VocabSource {
        family: VocabFamily::Bpe,
        pieces: vec![b"a".to_vec()],
        ..VocabSource::default()
    };
    let error = Vocab::try_new(empty).expect_err("BPE 缺合并表");
    assert_eq!(error.code(), Code::Missing);

    let merge = Merge {
        left: vec![b'a'],
        right: vec![b'b'],
    };
    let with_merges = VocabSource {
        family: VocabFamily::Bpe,
        pieces: vec![b"a".to_vec(), b"b".to_vec(), b"ab".to_vec()],
        merges: vec![merge],
        ..VocabSource::default()
    };
    Vocab::try_new(with_merges).expect("带合并表的 BPE");

    let stray = VocabSource {
        merges: vec![Merge {
            left: vec![b'a'],
            right: vec![b'b'],
        }],
        ..word_level(vec!["a".as_bytes()])
    };
    let error = Vocab::try_new(stray).expect_err("非 BPE 携带合并表必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn bpe_merge_sides_must_be_in_the_vocabulary() {
    let source = VocabSource {
        family: VocabFamily::Bpe,
        pieces: vec![b"a".to_vec(), b"ab".to_vec()],
        merges: vec![Merge {
            left: b"a".to_vec(),
            right: b"b".to_vec(),
        }],
        ..VocabSource::default()
    };
    let error = Vocab::try_new(source).expect_err("合并右片不在词表内必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn bpe_rejects_a_duplicate_merge_pair() {
    let source = VocabSource {
        family: VocabFamily::Bpe,
        pieces: vec![b"a".to_vec(), b"b".to_vec(), b"ab".to_vec()],
        merges: vec![
            Merge {
                left: b"a".to_vec(),
                right: b"b".to_vec(),
            },
            Merge {
                left: b"a".to_vec(),
                right: b"b".to_vec(),
            },
        ],
        ..VocabSource::default()
    };
    let error = Vocab::try_new(source).expect_err("重复合并对必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn unigram_scores_must_match_piece_count() {
    let pieces = vec![b"a".to_vec(), b"b".to_vec()];
    let mismatch = VocabSource {
        family: VocabFamily::Unigram,
        pieces: pieces.clone(),
        scores: vec![0.1],
        ..VocabSource::default()
    };
    let error = Vocab::try_new(mismatch).expect_err("分数个数不齐必须拒绝");
    assert_eq!(error.code(), Code::Invalid);

    let balanced = VocabSource {
        family: VocabFamily::Unigram,
        scores: vec![0.1, -0.2],
        pieces,
        ..VocabSource::default()
    };
    Vocab::try_new(balanced).expect("分数与词片齐平的 Unigram");
}

fn bpe_vocab() -> Vocab {
    let source = VocabSource {
        family: VocabFamily::Bpe,
        pieces: vec![
            b"a".to_vec(),  // 0
            b"b".to_vec(),  // 1
            b"c".to_vec(),  // 2
            b"ab".to_vec(), // 3
            b"bc".to_vec(), // 4
        ],
        merges: vec![
            Merge {
                left: b"a".to_vec(),
                right: b"b".to_vec(),
            }, // rank 0
            Merge {
                left: b"b".to_vec(),
                right: b"c".to_vec(),
            }, // rank 1
        ],
        ..VocabSource::default()
    };
    Vocab::try_new(source).expect("BPE 词表构造")
}

#[test]
fn bpe_merges_by_lowest_rank_and_round_trips() {
    let encoder = Bpe::new(bpe_vocab()).expect("BPE 实例");

    // a b c：(a,b) 秩 0 先于 (b,c) 秩 1 归并，得到 ab + c。
    assert_eq!(
        encoder.encode("abc").expect("编码"),
        [Token::new(3), Token::new(2)]
    );
    assert_eq!(
        encoder
            .decode(&[Token::new(3), Token::new(2)])
            .expect("解码"),
        "abc"
    );

    // b c：唯一对归并为 bc。
    assert_eq!(encoder.encode("bc").expect("编码"), [Token::new(4)]);
    assert_eq!(encoder.decode(&[Token::new(4)]).expect("解码"), "bc");
}

#[test]
fn bpe_keeps_unmerged_split_bytes() {
    let encoder = Bpe::new(bpe_vocab()).expect("BPE 实例");
    // c a b：先归并尾部 a b，c 保留单字节。
    assert_eq!(
        encoder.encode("cab").expect("编码"),
        [Token::new(2), Token::new(3)]
    );
}

#[test]
fn bpe_rejects_non_bpe_family() {
    let source = VocabSource {
        family: VocabFamily::Unigram,
        pieces: vec![b"a".to_vec()],
        scores: vec![0.0],
        ..VocabSource::default()
    };
    let vocab = Vocab::try_new(source).expect("Unigram 词表");
    let error = Bpe::new(vocab).expect_err("非 BPE 族必须拒绝");
    assert_eq!(error.code(), Code::Unsupported);
}

#[test]
fn bpe_rejects_piece_absent_from_vocab() {
    let encoder = Bpe::new(bpe_vocab()).expect("BPE 实例");
    let error = encoder.encode("d").expect_err("词表外字节必须报错");
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn bpe_lookup_and_piece_are_inverse() {
    let encoder = Bpe::new(bpe_vocab()).expect("BPE 实例");
    let token = encoder.lookup(b"ab").expect("查表").expect("ab 命中");
    assert_eq!(token.id(), 3);
    assert_eq!(encoder.piece(token).expect("反查"), Some(b"ab".to_vec()));
}

#[test]
fn fingerprint_separates_bpe_merge_order() {
    let pieces = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
    let order_a = VocabSource {
        family: VocabFamily::Bpe,
        pieces: pieces.clone(),
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
    let order_b = VocabSource {
        family: VocabFamily::Bpe,
        pieces: pieces.clone(),
        merges: vec![
            Merge {
                left: b"b".to_vec(),
                right: b"c".to_vec(),
            },
            Merge {
                left: b"a".to_vec(),
                right: b"b".to_vec(),
            },
        ],
        ..VocabSource::default()
    };
    let vocab_a = Vocab::try_new(order_a).expect("词表 A");
    let vocab_b = Vocab::try_new(order_b).expect("词表 B");
    assert_ne!(
        vocab_a.fingerprint().digest(),
        vocab_b.fingerprint().digest(),
        "合并秩序不同必须得到不同指纹"
    );
}

#[test]
fn fingerprint_separates_unigram_scores() {
    let pieces = vec![b"a".to_vec(), b"b".to_vec()];
    let scored = |scores: Vec<f32>| VocabSource {
        family: VocabFamily::Unigram,
        pieces: pieces.clone(),
        scores,
        ..VocabSource::default()
    };
    let first = Vocab::try_new(scored(vec![0.1, 0.2])).expect("词表 A");
    let second = Vocab::try_new(scored(vec![0.1, 0.3])).expect("词表 B");
    assert_ne!(
        first.fingerprint().digest(),
        second.fingerprint().digest(),
        "分数内容不同必须得到不同指纹"
    );
}

#[test]
fn stream_decodes_across_chunk_boundaries() {
    let mut stream = DecoderStream::new();
    // "中" 是三字节 EF BC 93；切成两块喂入。
    let prefix = stream.push(&[0xE4, 0xB8]).expect("首块不完整，前缀为空");
    assert_eq!(prefix, "");
    let complete = stream.push(&[0xAD]).expect("次块拼出完整字符");
    assert_eq!(complete, "中");
    assert_eq!(stream.finish().expect("收尾"), "");
}

#[test]
fn stream_rejects_interior_invalid_bytes() {
    let mut stream = DecoderStream::new();
    let error = stream
        .push(&[0xE4, 0x00])
        .expect_err("中间非法字节必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn stream_finish_refuses_incomplete_tail() {
    let mut stream = DecoderStream::new();
    stream.push(&[0xE4]).expect("不完整尾字节留在缓冲");
    let error = stream.finish().expect_err("收尾仍不完整必须报错");
    assert_eq!(error.code(), Code::Invalid);
}
