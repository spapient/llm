//! 一次编码的结果：编号、字符区间与特殊标记三者等长。

use crate::Token;
use diagnostic::{Code, Error, Result};

pub use stream::DecoderStream;

/// 编码产出的对齐记录：token 编号、在原文本中的字节区间、是否特殊 token。
///
/// 三段长度在构造期闭合；暂不产出元数据的实现可用 [`Encoding::from_ids`]，
/// 区间与标记留空。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoding {
    ids: Vec<Token>,
    offsets: Vec<(usize, usize)>,
    special: Vec<bool>,
}

impl Encoding {
    /// 只带编号；对齐区间与特殊标记留空。
    #[must_use]
    pub fn from_ids(ids: Vec<Token>) -> Self {
        Self {
            ids,
            offsets: Vec::new(),
            special: Vec::new(),
        }
    }

    /// 带完整元数据；三段长度必须一致，否则报 [`Code::Invalid`]。
    ///
    /// 区间是原文本的 UTF-8 字节 `[start, end)`；`special` 标记该编号是否特殊 token。
    pub fn new(ids: Vec<Token>, offsets: Vec<(usize, usize)>, special: Vec<bool>) -> Result<Self> {
        let len = ids.len();
        if offsets.len() != len || special.len() != len {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "编码元数据长度不齐：编号 {len}、区间 {}、特殊标记 {}",
                    offsets.len(),
                    special.len()
                ),
            )
            .with("tokenizer::Encoding::new"));
        }
        Ok(Self {
            ids,
            offsets,
            special,
        })
    }

    /// token 编号序列。
    #[must_use]
    pub fn ids(&self) -> &[Token] {
        &self.ids
    }

    /// 每个编号对应的字节区间；未产出时为空。
    #[must_use]
    pub fn offsets(&self) -> &[(usize, usize)] {
        &self.offsets
    }

    /// 第 `index` 条是否特殊 token；越界返回 [`None`]。
    #[must_use]
    pub fn is_special(&self, index: usize) -> Option<bool> {
        self.special.get(index).copied()
    }

    /// token 个数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// 是否为空序列。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// 拆开编号序列，丢弃对齐元数据。
    #[must_use]
    pub fn into_ids(self) -> Vec<Token> {
        self.ids
    }
}

mod stream {
    //! 逐 token 解码的增量缓冲：跨块的多字节字符留在缓冲里，只放出完整 UTF-8 前缀。

    use diagnostic::{Code, Error, Result};

    /// UTF-8 边界安全的流式解码器。
    ///
    /// 推理逐 token 产出文本时，一块解码字节可能在多字节字符中间断开；本结构把不完整的
    /// 尾字节留在缓冲，下次追加后再拼合。中间出现非法字节不是分片边界，直接拒绝。
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct DecoderStream {
        buffer: Vec<u8>,
    }

    impl DecoderStream {
        /// 空解码器。
        pub fn new() -> Self {
            Self::default()
        }

        /// 追加一块解码字节，返回当前已成完整字符的前缀；不完整尾字节留在缓冲。
        pub fn push(&mut self, chunk: &[u8]) -> Result<String> {
            self.buffer.extend_from_slice(chunk);
            match std::str::from_utf8(&self.buffer) {
                Ok(_) => {
                    let complete = String::from_utf8(std::mem::take(&mut self.buffer))
                        .map_err(|error| self.invalid(error))?;
                    Ok(complete)
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    match error.error_len() {
                        // 尾部不完整多字节序列：留到下一块。
                        None => {
                            let prefix: Vec<u8> = self.buffer[..valid].to_vec();
                            let tail = self.buffer.split_off(valid);
                            self.buffer = tail;
                            String::from_utf8(prefix).map_err(|error| self.invalid(error))
                        }
                        // 中间的非法字节：不是分片边界。
                        Some(_) => Err(self.invalid(error)),
                    }
                }
            }
        }

        /// 收尾：缓冲必须是合法 UTF-8，否则报 [`Code::Invalid`]。
        pub fn finish(self) -> Result<String> {
            String::from_utf8(self.buffer).map_err(|error| {
                Error::new(Code::Invalid, format!("收尾序列不是合法 UTF-8：{error}"))
                    .with("tokenizer::DecoderStream::finish")
            })
        }

        fn invalid(&self, error: impl std::fmt::Display) -> Error {
            Error::new(Code::Invalid, format!("解码序列含非法 UTF-8：{error}"))
                .with("tokenizer::DecoderStream::push")
        }
    }
}
