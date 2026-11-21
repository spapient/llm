//! 词表装配适配器：`llm.vocab` 段字节 → 注入用 [`tokenizer::Tokenizer`]。

use diagnostic::{Code, Error, Result};

const SITE: &str = "serve::vocab";

/// 把词表段字节装配成可注入的分词器。
pub fn tokenizer_from_segment(bytes: &[u8]) -> Result<Box<dyn tokenizer::Tokenizer>> {
    let (source, digest) = tokenizer::decode_vocab(bytes)?;
    let vocab = tokenizer::Vocab::try_new(source)?;
    if vocab.fingerprint().digest() != digest {
        return Err(Error::new(Code::Invalid, "词表指纹与段尾摘要不符").with(SITE));
    }
    vocab.into_tokenizer()
}

#[cfg(test)]
#[path = "../../tests/vocab_tests.rs"]
mod tests;
