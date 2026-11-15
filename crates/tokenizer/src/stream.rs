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
