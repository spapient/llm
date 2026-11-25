use super::*;
use tokenizer::VocabSource;
use tokenizer::fixtures::bpe_source;

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
