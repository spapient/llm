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
    ///
    /// 参数：`$label` 是错误文案里的词表名（如 `"BPE"`），`$marker` 是错误调用点标记
    /// （如 `"tokenizer::Bpe::decode"`）。
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
mod encoding;
#[doc(hidden)]
pub mod fixtures;
mod special;
mod strategy;
mod unigram;
mod vocab;
mod vocab_codec;

pub use bpe::Bpe;
pub use bytes::Bytes;
pub use encoding::{DecoderStream, Encoding};
pub use special::SpecialTokens;
pub use strategy::{PadDirection, Padding, Strategy, Truncation};
pub use unigram::Unigram;
pub use vocab::{Fingerprint, Merge, Vocab, VocabFamily, VocabSource};
pub use vocab_codec::{decode_vocab, encode_vocab};

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
mod tests;
