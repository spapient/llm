use super::*;

fn roundtrip(source: VocabSource) {
    let digest = 0xdead_beef_1234_5678u64;
    let bytes = encode_vocab(&source, digest);
    let (decoded, got) = decode_vocab(&bytes).expect("解码");
    assert_eq!(decoded, source, "往返一致");
    assert_eq!(got, digest, "摘要透传");
}

#[test]
fn byte_family_round_trips() {
    roundtrip(VocabSource {
        family: VocabFamily::Byte,
        pieces: vec![b"a".to_vec(), b"b".to_vec()],
        special: SpecialTokens::EMPTY,
        scores: Vec::new(),
        merges: Vec::new(),
    });
}

#[test]
fn bpe_with_merges_and_special_round_trips() {
    roundtrip(crate::fixtures::bpe_source());
}

#[test]
fn unigram_with_scores_round_trips() {
    roundtrip(VocabSource {
        family: VocabFamily::Unigram,
        pieces: vec![b"a".to_vec(), b"b".to_vec()],
        special: SpecialTokens::EMPTY.with_unk(0),
        scores: vec![0.5, -0.25],
        merges: Vec::new(),
    });
}

#[test]
fn bad_magic_is_rejected() {
    let mut bytes = encode_vocab(&VocabSource::default(), 0);
    bytes[0] ^= 0xff;
    assert_eq!(decode_vocab(&bytes).unwrap_err().code(), Code::Invalid);
}

#[test]
fn truncation_is_rejected() {
    let bytes = encode_vocab(&VocabSource::default(), 0);
    let cut = &bytes[..bytes.len() - 1];
    assert_eq!(decode_vocab(cut).unwrap_err().code(), Code::Invalid);
}
