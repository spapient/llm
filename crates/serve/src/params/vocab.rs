//! 词表装配适配器：`llm.vocab` 段字节 → 注入用 [`tokenizer::Tokenizer`]。
//!
//! 只吃已经由容器按段级 CRC 校验过、按名取出的词表段字节；本模块不接触容器字节、
//! 不复制指纹算法。装配顺序是：中性解码 → 闭合词表不变量 → 指纹对拍（段尾摘要 vs
//! 重算指纹）→ 按族折成具体分词器。指纹不符在构造期报 [`Code::Invalid`]，不静默降级。

use diagnostic::{Code, Error, Result};

const SITE: &str = "serve::vocab";

/// 把词表段字节装配成可注入的分词器。
///
/// 返回 [`Box<dyn tokenizer::Tokenizer>`]，上层只认 trait；族无实现时照常上抛
/// [`Code::Unsupported`]，不在此处补平行分支。
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
