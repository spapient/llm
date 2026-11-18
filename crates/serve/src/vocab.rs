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
mod tests {
    use super::*;
    use tokenizer::{Merge, SpecialTokens, VocabFamily, VocabSource};

    fn bpe_source() -> VocabSource {
        VocabSource {
            family: VocabFamily::Bpe,
            pieces: vec![b"a".to_vec(), b"b".to_vec(), b"ab".to_vec()],
            special: SpecialTokens::EMPTY.with_bos(0).with_eos(2),
            scores: Vec::new(),
            merges: vec![Merge {
                left: b"a".to_vec(),
                right: b"b".to_vec(),
            }],
        }
    }

    /// 按真实装配路径造段：词表重算指纹作为段尾摘要，再中性编码。
    fn segment(source: &VocabSource) -> Vec<u8> {
        let vocab = tokenizer::Vocab::try_new(source.clone()).expect("词表");
        tokenizer::encode_vocab(source, vocab.fingerprint().digest())
    }

    #[test]
    fn bpe_segment_assembles_and_round_trips() {
        let tokenizer = tokenizer_from_segment(&segment(&bpe_source())).expect("装配");
        let tokens = tokenizer.encode("ab").expect("编码");
        assert_eq!(tokenizer.decode(&tokens).expect("解码"), "ab");
    }

    #[test]
    fn tampered_digest_is_rejected() {
        let source = bpe_source();
        let vocab = tokenizer::Vocab::try_new(source.clone()).expect("词表");
        let bytes = tokenizer::encode_vocab(&source, vocab.fingerprint().digest() ^ 1);
        let error = tokenizer_from_segment(&bytes).err().expect("应报错");
        assert_eq!(error.code(), Code::Invalid);
    }
}
