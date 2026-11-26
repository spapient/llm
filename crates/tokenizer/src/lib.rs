//! 文本 ↔ token 的唯一边界：[`Token`]、[`Tokenizer`] 与字节级参考实现 [`Bytes`]。
//!
//! **拥有**：`Tokenizer`、`Token`、`Bytes` 以及中性契约值对象（[`SpecialTokens`]、
//! [`Encoding`]、[`Strategy`]、[`Fingerprint`]、[`DecoderStream`]、[`Vocab`]），
//! 与 BPE 具体实现 [`Bpe`]、Unigram 具体实现 [`Unigram`]。
//! **不拥有**：WordPiece / SentencePiece 算法与容器加载（后续变体落地）、
//! 采样（由 `engine` 的调用方决定）。**依赖**：`diagnostic`。**被依赖**：`serve`。
//!
//! 先有 trait 再有实现：上层（调度、服务）只认 trait，所以换词表不影响它们。`Bytes` 不是玩具，
//! 它是"最坏情况下的 1 token/byte"基线，对拍与回归都用得上。
//!
//! # 例子
//!
//! ```
//! use llm_tokenizer::{Bytes, Token, Tokenizer};
//!
//! let tokens = Bytes.encode("hé").expect("编码");
//! assert_eq!(tokens, [Token::new(104), Token::new(195), Token::new(169)]);
//! assert_eq!(Bytes.decode(&tokens).expect("解码"), "hé");
//! ```

#![deny(unsafe_code)]

#[macro_use]
mod delegates {
    //! 词表委托方法的共享宏：`Bpe` 与 `Unigram` 在 [`Vocab`] 上的只读面完全相同，
    //! 差异只在错误文案里的词表名与调用点标记。这里用一张宏表收敛，新分词算法
    //! （WordPiece / SentencePiece）落地时只传两个字符串，不再重写这五个方法。

    /// 生成 `Tokenizer` 里五个只读委托：`decode`、`fingerprint`、`special_tokens`、
    /// `lookup`、`piece`。调用点在 `impl Tokenizer for X` 内，且该结构体必须持有
    /// `vocab: crate::Vocab` 字段。
    macro_rules! vocab_tokenizer_delegates {
        ($label:literal, $marker:literal) => {
            fn decode(&self, tokens: &[crate::Token]) -> diagnostic::Result<String> {
                let mut bytes = Vec::with_capacity(tokens.len());
                for token in tokens {
                    let piece = self.vocab.piece(token.id()).ok_or_else(|| {
                        diagnostic::Error::new(
                            diagnostic::Code::Invalid,
                            format!(
                                "token {} 超出 {} 词表（0..{}）",
                                token.id(),
                                $label,
                                self.vocab_size()
                            ),
                        )
                        .with($marker)
                    })?;
                    bytes.extend_from_slice(piece);
                }
                String::from_utf8(bytes).map_err(|error| {
                    diagnostic::Error::new(
                        diagnostic::Code::Invalid,
                        format!("token 序列不是合法 UTF-8：{error}"),
                    )
                    .with($marker)
                })
            }

            fn fingerprint(&self) -> crate::Fingerprint {
                *self.vocab.fingerprint()
            }

            fn special_tokens(&self) -> &crate::SpecialTokens {
                self.vocab.special()
            }

            fn lookup(&self, piece: &[u8]) -> diagnostic::Result<Option<crate::Token>> {
                Ok(self.vocab.lookup(piece))
            }

            fn piece(&self, token: crate::Token) -> diagnostic::Result<Option<Vec<u8>>> {
                Ok(self.vocab.piece(token.id()).map(<[u8]>::to_vec))
            }
        };
    }
}

mod bpe;
mod bytes;
/// 词表中性线格式；实现与词表值对象同住在 `vocab`。
pub mod codec {
    pub use crate::vocab::{decode_vocab, encode_vocab};
}
mod encoding;
#[doc(hidden)]
pub mod fixtures;
mod special;
mod strategy;
mod unigram;
mod vocab;

pub use bpe::Bpe;
pub use bytes::Bytes;
pub use codec::{decode_vocab, encode_vocab};
pub use encoding::{DecoderStream, Encoding};
pub use special::SpecialTokens;
pub use strategy::{PadDirection, Padding, Strategy, Truncation};
pub use unigram::Unigram;
pub use vocab::{Fingerprint, Merge, Vocab, VocabFamily, VocabSource};

use diagnostic::Result;

/// 一个 token 编号（词表内的下标）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Token(u32);

impl Token {
    /// 由编号构造。
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// 编号。
    #[must_use]
    pub fn id(&self) -> u32 {
        self.0
    }
}

impl From<u32> for Token {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<Token> for u32 {
    fn from(token: Token) -> Self {
        token.0
    }
}

impl AsRef<u32> for Token {
    fn as_ref(&self) -> &u32 {
        &self.0
    }
}

/// 分词器契约：编号与文本互相转换，且两者必须能往返。
///
/// 必需方法只覆盖最薄的编码/解码面；对齐元数据、反查、计数与特殊句柄都有默认实现，
/// 实现者按需覆盖。
///
/// # 不变量
///
/// - [`encode`](Self::encode) 产出的每个编号都满足 `token.id() < self.vocab_size()`；
///   解码方据此把越界视为输入错误，而不是回绕取模。
/// - 对词表内可表达的文本，`decode(&encode(text)?)` 必须还原原文本；词表外片段按各自
///   实现的 OOV 策略报错或回退到未知编号。
/// - 特殊 token 由 [`special_tokens`](Self::special_tokens) 句柄登记，不占普通词片查表路径。
pub trait Tokenizer {
    /// 词表大小（含特殊 token）。
    fn vocab_size(&self) -> usize;

    /// 文本 → token。
    fn encode(&self, text: &str) -> Result<Vec<Token>>;

    /// token → 文本；遇到词表外的编号或非法 UTF-8 必须报错，不能悄悄替换。
    fn decode(&self, tokens: &[Token]) -> Result<String>;

    /// 词表身份指纹：实验记录与对拍时标识一份词表。
    fn fingerprint(&self) -> Fingerprint;

    /// 特殊 token 句柄；基线为空。
    fn special_tokens(&self) -> &SpecialTokens {
        &SpecialTokens::EMPTY
    }

    /// 只数 token 个数，不分配编号向量；默认走 [`Self::encode`]，实现可给零分配路径。
    fn count(&self, text: &str) -> Result<usize> {
        Ok(self.encode(text)?.len())
    }

    /// 文本 → 带对齐元数据的编码结果；默认只填编号，区间与特殊标记留空。
    fn encode_detailed(&self, text: &str) -> Result<Encoding> {
        Ok(Encoding::from_ids(self.encode(text)?))
    }

    /// 词片字节 → 编号；词表不支持反查时返回 `Ok(None)`。
    fn lookup(&self, _piece: &[u8]) -> Result<Option<Token>> {
        Ok(None)
    }

    /// 编号 → 词片字节；词表不支持反查时返回 `Ok(None)`。
    fn piece(&self, token: Token) -> Result<Option<Vec<u8>>> {
        let _ = token;
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::unigram_source;
    use diagnostic::Code;

    testkit::test_cases! {
    fn encoding_and_decoding_round_trip() {
        for text in ["", "hello", "héllo", "中文", "🙂 emoji"] {
            let tokens = Bytes.encode(text).expect("编码");
            assert_eq!(tokens.len(), text.len(), "字节级分词：token 数等于字节数");
            assert_eq!(Bytes.decode(&tokens).expect("解码"), text);
        }
    }

    fn the_vocabulary_is_the_byte_range() {
        assert_eq!(Bytes.vocab_size(), 256);
        let error = Bytes.decode(&[Token::new(256)]).expect_err("超出词表");
        assert_eq!(error.code(), Code::Invalid);
    }

    /// 截断的多字节序列要让调用方看得到原因（契约第三条：拒绝必须有话说）。
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

    fn encoding_rejects_mismatched_metadata_lengths() {
        let ids = vec![Token::new(1), Token::new(2)];
        let offsets = vec![(0, 1)];
        let special = vec![false, true];
        let error = Encoding::new(ids, offsets, special).expect_err("长度不齐必须拒绝");
        assert_eq!(error.code(), Code::Invalid);
    }

    fn detailed_encoding_carries_byte_offsets() {
        let encoding = Bytes.encode_detailed("hé").expect("详细编码");
        assert_eq!(encoding.len(), "hé".len());
        assert_eq!(encoding.offsets(), &[(0, 1), (1, 2), (2, 3)]);
        assert_eq!(encoding.is_special(0), Some(false));
        assert!(encoding.offsets().len() == encoding.len());
    }

    fn count_is_zero_allocation_length_for_bytes() {
        assert_eq!(Bytes.count("中文🙂").expect("计数"), "中文🙂".len());
    }

    fn lookup_and_piece_are_inverse_for_single_bytes() {
        for byte in [0u8, 65, 255] {
            let token = Bytes.lookup(&[byte]).expect("查表").expect("单字节必命中");
            assert_eq!(token.id(), u32::from(byte));
            assert_eq!(Bytes.piece(token).expect("反查"), Some(vec![byte]));
        }
        assert!(Bytes.lookup(&[1, 2]).expect("多字节返回空").is_none());
    }

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

    fn padding_rejects_zero_length() {
        let error = Padding::new(0, 0, PadDirection::Right).expect_err("零长度填充必须拒绝");
        assert_eq!(error.code(), Code::Invalid);
    }

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

    fn strategy_sep_flag_is_off_by_default() {
        assert!(!Strategy::no_op().add_sep());
        assert!(Strategy::no_op().with_sep().add_sep());
    }
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

    testkit::test_cases! {
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

    fn vocab_rejects_duplicate_piece() {
        let source = word_level(vec!["a".as_bytes(), "a".as_bytes()]);
        let error = Vocab::try_new(source).expect_err("重复词片必须拒绝");
        assert_eq!(error.code(), Code::Invalid);
    }

    fn vocab_rejects_special_id_out_of_range() {
        let special = SpecialTokens::EMPTY.with_bos(9);
        let source = VocabSource {
            special,
            ..word_level(vec!["a".as_bytes()])
        };
        let error = Vocab::try_new(source).expect_err("越界特殊编号必须拒绝");
        assert_eq!(error.code(), Code::Invalid);
    }

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

    testkit::test_cases! {
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

    fn bpe_keeps_unmerged_split_bytes() {
        let encoder = Bpe::new(bpe_vocab()).expect("BPE 实例");
        // c a b：先归并尾部 a b，c 保留单字节。
        assert_eq!(
            encoder.encode("cab").expect("编码"),
            [Token::new(2), Token::new(3)]
        );
    }

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

    fn bpe_rejects_piece_absent_from_vocab() {
        let encoder = Bpe::new(bpe_vocab()).expect("BPE 实例");
        let error = encoder.encode("d").expect_err("词表外字节必须报错");
        assert_eq!(error.code(), Code::Invalid);
    }

    fn bpe_lookup_and_piece_are_inverse() {
        let encoder = Bpe::new(bpe_vocab()).expect("BPE 实例");
        let token = encoder.lookup(b"ab").expect("查表").expect("ab 命中");
        assert_eq!(token.id(), 3);
        assert_eq!(encoder.piece(token).expect("反查"), Some(b"ab".to_vec()));
    }

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
            pieces,
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
    }

    fn unigram_vocab() -> Vocab {
        Vocab::try_new(unigram_source(true)).expect("合成 Unigram 词表构造")
    }

    testkit::test_cases! {
    fn unigram_prefers_the_higher_scored_segment() {
        let encoder = Unigram::new(unigram_vocab()).expect("Unigram 实例");
        // ab（-0.1）优于 a+b（-2.0）。
        assert_eq!(encoder.encode("ab").expect("编码"), [Token::new(3)]);
    }

    fn unigram_viterbi_picks_the_best_multi_piece_path() {
        let encoder = Unigram::new(unigram_vocab()).expect("Unigram 实例");
        // ab+c（-1.1）优于 abc（-2.0）与 a+b+c（-3.0）。
        assert_eq!(
            encoder.encode("abc").expect("编码"),
            [Token::new(3), Token::new(2)]
        );
    }

    fn unigram_round_trips_through_decode() {
        let encoder = Unigram::new(unigram_vocab()).expect("Unigram 实例");
        let tokens = encoder.encode("abc").expect("编码");
        assert_eq!(encoder.decode(&tokens).expect("解码"), "abc");
    }

    fn unigram_falls_back_to_registered_unknown_token() {
        let encoder = Unigram::new(unigram_vocab()).expect("Unigram 实例");
        // d 不在词表：回退到编号 5 的 unknown 片。
        assert_eq!(encoder.encode("d").expect("编码"), [Token::new(5)]);
    }

    fn unigram_rejects_uncovered_input_without_an_unknown_slot() {
        let vocab = Vocab::try_new(unigram_source(false)).expect("无 unk 词表");
        let encoder = Unigram::new(vocab).expect("Unigram 实例");
        let error = encoder
            .encode("d")
            .expect_err("无覆盖路径且未配 unk 必须拒绝");
        assert_eq!(error.code(), Code::Invalid);
    }

    fn unigram_rejects_non_unigram_family() {
        let source = VocabSource {
            family: VocabFamily::Byte,
            pieces: vec![b"a".to_vec()],
            ..VocabSource::default()
        };
        let vocab = Vocab::try_new(source).expect("字节词表");
        let error = Unigram::new(vocab).expect_err("族不匹配必须拒绝");
        assert_eq!(error.code(), Code::Unsupported);
    }

    fn unigram_lookup_and_piece_are_inverse() {
        let encoder = Unigram::new(unigram_vocab()).expect("Unigram 实例");
        let token = encoder.lookup(b"ab").expect("查表").expect("ab 命中");
        assert_eq!(token.id(), 3);
        assert_eq!(encoder.piece(token).expect("反查"), Some(b"ab".to_vec()));
    }

    fn stream_decodes_across_chunk_boundaries() {
        let mut stream = DecoderStream::new();
        // "中" 是三字节 EF BC 93；切成两块喂入。
        let prefix = stream.push(&[0xE4, 0xB8]).expect("首块不完整，前缀为空");
        assert_eq!(prefix, "");
        let complete = stream.push(&[0xAD]).expect("次块拼出完整字符");
        assert_eq!(complete, "中");
        assert_eq!(stream.finish().expect("收尾"), "");
    }

    fn stream_rejects_interior_invalid_bytes() {
        let mut stream = DecoderStream::new();
        let error = stream
            .push(&[0xE4, 0x00])
            .expect_err("中间非法字节必须拒绝");
        assert_eq!(error.code(), Code::Invalid);
    }

    fn stream_finish_refuses_incomplete_tail() {
        let mut stream = DecoderStream::new();
        stream.push(&[0xE4]).expect("不完整尾字节留在缓冲");
        let error = stream.finish().expect_err("收尾仍不完整必须报错");
        assert_eq!(error.code(), Code::Invalid);
    }
    }

    // —— P0-6 属性边界用例：空输入、非对齐长度、越界索引、饱和值与重复边界 ——

    testkit::test_cases! {
    /// 空词表装配：没有任何词片，查表与反查都必须返回空，族附属表按各自规则校验。
    fn vocab_empty_vocabulary_has_no_lookup_hits() {
        let source = VocabSource::default();
        let vocab = Vocab::try_new(source).expect("空词表构造合法");
        assert_eq!(vocab.vocab_size(), 0);
        assert!(vocab.lookup(b"a").is_none());
        assert!(vocab.piece(0).is_none());
        assert!(vocab.piece(u32::MAX).is_none());
    }

    /// 编号边界：越界编号（词表大小、`u32::MAX`）反查为空；词表内编号往返一致。
    fn vocab_piece_lookup_saturates_at_the_vocabulary_boundary() {
        let source = word_level(vec!["a".as_bytes(), "bc".as_bytes()]);
        let vocab = Vocab::try_new(source).expect("词表构造");
        let out_of_range = [vocab.vocab_size() as u32, u32::MAX];
        for id in out_of_range {
            assert!(vocab.piece(id).is_none(), "编号 {id} 越界必须返回空");
        }
        for id in 0..vocab.vocab_size() as u32 {
            let piece = vocab.piece(id).expect("词表内编号必命中");
            assert_eq!(vocab.lookup(piece), Some(Token::new(id)));
        }
    }

    /// 词片键边界：空片、重复字节、多字节片段在查表路径上互不串扰。
    fn vocab_lookup_handles_empty_repeated_and_multibyte_pieces() {
        let source = word_level(vec![b"".as_slice(), b"aa".as_slice(), "中".as_bytes()]);
        let vocab = Vocab::try_new(source).expect("词表构造");
        assert_eq!(vocab.lookup(b""), Some(Token::new(0)));
        assert_eq!(vocab.lookup(b"aa"), Some(Token::new(1)));
        assert_eq!(vocab.lookup(b"a"), None, "单字节不是 `aa` 的前缀命中");
        assert_eq!(vocab.lookup(&[0xE4, 0xB8, 0xAD]), Some(Token::new(2)));
        assert_eq!(vocab.lookup(&[0xE4, 0xB8, 0xAD, 0x00]), None);
    }

    /// 特殊句柄：空句柄与边界编号查询都不命中；登记的编号逐个可反查。
    fn special_handles_boundary_ids_do_not_collide() {
        let empty = SpecialTokens::EMPTY;
        assert!(empty.is_empty());
        for id in [0u32, 1, 255, u32::MAX] {
            assert!(!empty.is_special(id));
        }
        let handles = SpecialTokens::empty()
            .with_bos(0)
            .with_eos(1)
            .with_unk(2)
            .with_pad(3)
            .with_sep(4)
            .with_cls(5)
            .with_mask(6);
        assert_eq!(handles.len(), 7);
        for id in 0..=6 {
            assert!(handles.is_special(id));
        }
        assert!(!handles.is_special(7));
    }

    /// 编码元数据三段长度不齐：编号多、区间多、标记多三种非对齐都必须拒绝。
    fn encoding_rejects_each_misaligned_metadata_side() {
        let ids = vec![Token::new(1), Token::new(2)];
        let offsets = vec![(0, 1), (1, 2)];
        let special = vec![false, true];
        let cases = [
            (ids.clone(), offsets.clone(), vec![false], "special 少一段"),
            (ids.clone(), vec![(0, 1)], special.clone(), "offsets 少一段"),
            (
                vec![Token::new(1)],
                offsets.clone(),
                special.clone(),
                "ids 少一段",
            ),
        ];
        for (i, (left, middle, right, label)) in cases.into_iter().enumerate() {
            let error = Encoding::new(left, middle, right).expect_err(label);
            assert_eq!(error.code(), Code::Invalid, "用例 {i}：{label}");
        }
        let aligned = Encoding::new(ids, offsets, special).expect("三段齐平合法");
        assert_eq!(aligned.len(), 2);
        assert!(!aligned.is_empty());
        assert_eq!(aligned.is_special(0), Some(false));
        assert_eq!(aligned.is_special(2), None, "标记越界返回空");
    }

    /// 字节级解码：上界编号、截断多字节、非法续节与重复字节边界都带说明拒绝。
    fn bytes_decode_rejects_saturated_ids_and_invalid_utf8() {
        let cases: [&[Token]; 5] = [
            &[Token::new(256)],
            &[Token::new(u32::MAX)],
            &[Token::new(0xE4)],
            &[Token::new(0xE4), Token::new(0x00)],
            &[Token::new(u32::from(b'a')), Token::new(u32::from(b'a'))],
        ];
        // 最后一例是合法 ASCII 重复字节；前四例必须拒绝。
        for (idx, tokens) in cases.iter().enumerate() {
            match Bytes.decode(tokens) {
                Ok(text) => assert_eq!(text, "aa", "重复 ASCII 往返为原文本"),
                Err(error) => {
                    assert_eq!(error.code(), Code::Invalid, "用例 {idx}");
                    assert!(!error.message().trim().is_empty(), "拒绝必须有话说");
                }
            }
        }
    }

    /// 解码空序列为空串；空文本编码为空编号，计数为零。
    fn bytes_empty_round_trips_through_encode_decode_count() {
        assert_eq!(Bytes.encode("").expect("空编码"), []);
        assert_eq!(Bytes.decode(&[]).expect("空解码"), "");
        assert_eq!(Bytes.count("").expect("空计数"), 0);
    }

    /// 流式解码：空块、跨块完整字符、重复续节与收尾残留边界。
    fn stream_handles_empty_chunks_and_repeated_tail_bytes() {
        let mut stream = DecoderStream::new();
        assert_eq!(stream.push(&[]).expect("空块放行"), "");
        assert_eq!(stream.push("ab".as_bytes()).expect("完整 ASCII"), "ab");
        assert_eq!(stream.finish().expect("收尾"), "");

        let mut repeated = DecoderStream::new();
        repeated.push(&[0xE4, 0xB8]).expect("首块留尾");
        repeated
            .push(&[0xAD, 0xE4, 0xB8])
            .expect("次块拼出首字并再留尾");
        let error = repeated.finish().expect_err("残留半字符必须拒绝");
        assert_eq!(error.code(), Code::Invalid);
    }
    }
}
