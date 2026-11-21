//! 字节级参考实现：每个 UTF-8 字节是一个 token，词表 256。

use crate::{Encoding, Fingerprint, SpecialTokens, Token, Tokenizer};
use diagnostic::{Code, Error, Result};

/// 字节级分词视图；无内部状态。
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

    fn fingerprint(&self) -> Fingerprint {
        Fingerprint::new("bytes", 256, "one-token-per-byte;utf-8-byte-vocab")
    }

    fn count(&self, text: &str) -> Result<usize> {
        Ok(text.len())
    }

    fn encode_detailed(&self, text: &str) -> Result<Encoding> {
        let ids = self.encode(text)?;
        let offsets = (0..text.len()).map(|i| (i, i + 1)).collect();
        let special = vec![false; ids.len()];
        Encoding::new(ids, offsets, special)
    }

    fn lookup(&self, piece: &[u8]) -> Result<Option<Token>> {
        let [byte] = piece else { return Ok(None) };
        Ok(Some(Token::new(u32::from(*byte))))
    }

    fn piece(&self, token: Token) -> Result<Option<Vec<u8>>> {
        let id = token.id();
        Ok(u8::try_from(id).ok().map(|byte| vec![byte]))
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &SpecialTokens::EMPTY
    }
}
