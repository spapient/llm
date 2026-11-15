//! 文本 ↔ token 的唯一边界：[`Token`]、[`Tokenizer`] 与字节级参考实现 [`Bytes`]。
//!
//! **拥有**：`Tokenizer`、`Token`、`Bytes`。**不拥有**：BPE / SentencePiece 词表（P1 落地，
//! 属于实现而不是接口）、采样（由 `engine` 的调用方决定）。
//! **依赖**：`diagnostic`。**被依赖**：`serve`。
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

use diagnostic::{Code, Error, Result};

/// 一个 token 编号（词表内的下标）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Token(u32);

impl Token {
    /// 由编号构造。
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// 编号。
    pub fn id(&self) -> u32 {
        self.0
    }
}

/// 分词器契约：编号与文本互相转换，且两者必须能往返。
pub trait Tokenizer {
    /// 词表大小（含特殊 token）。
    fn vocab_size(&self) -> usize;

    /// 文本 → token。
    fn encode(&self, text: &str) -> Result<Vec<Token>>;

    /// token → 文本；遇到词表外的编号或非法 UTF-8 必须报错，不能悄悄替换。
    fn decode(&self, tokens: &[Token]) -> Result<String>;
}

/// 字节级参考实现：每个 UTF-8 字节是一个 token，词表 256。
///
/// 它在"压缩率"上没有价值，但给出了两层基线：正确性基线（往返必须是恒等）与性能基线
/// （最坏情况 1 token/byte）。真词表落地后，测试可以用它交叉验证边界行为。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Bytes;

impl Tokenizer for Bytes {
    fn vocab_size(&self) -> usize {
        256
    }

    fn encode(&self, text: &str) -> Result<Vec<Token>> {
        Ok(text
            .bytes()
            .map(|byte| Token::new(u32::from(byte)))
            .collect())
    }

    fn decode(&self, tokens: &[Token]) -> Result<String> {
        let mut bytes = Vec::with_capacity(tokens.len());
        for token in tokens {
            let id = token.id();
            let byte = u8::try_from(id).map_err(|_| {
                Error::new(
                    Code::Invalid,
                    format!("token {id} 超出字节级词表（0..256）"),
                )
                .with("tokenizer::Bytes::decode")
            })?;
            bytes.push(byte);
        }
        String::from_utf8(bytes).map_err(|error| {
            Error::new(Code::Invalid, format!("token 序列不是合法 UTF-8：{error}"))
                .with("tokenizer::Bytes::decode")
        })
    }
}

#[cfg(test)]
mod tests;
