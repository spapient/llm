//! Golden alignment against the real GPT-2 `tokenizer.json` and the HF
//! `tokenizers` reference output in `tests/fixtures/gpt2/gpt2_golden.json`.
//!
//! The golden was produced with `add_special_tokens = false`, which still
//! recognizes special tokens inside the text, so encoding is compared against
//! [`tokenizer::encodespecial`]; decoding is compared against the no-skip
//! variant because this tokenizer never skips special tokens.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use ::tokenizer::tokenizer as tokenizerengine;
use lib::tokenid;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gpt2");

fn loadtokenizer() -> tokenizerengine {
    let path = format!("{FIXTURES}/tokenizer.json");
    let document = std::fs::read_to_string(path).unwrap();
    tokenizerengine::fromjson(&document).unwrap()
}

fn loadsamples() -> serde_json::Value {
    let path = format!("{FIXTURES}/gpt2_golden.json");
    let text = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&text).unwrap()
}

#[test]
fn gpt2_encodes_match_the_hf_reference() {
    let tok = loadtokenizer();
    let golden = loadsamples();
    for sample in golden["samples"].as_array().unwrap() {
        let input = sample["input"].as_str().unwrap();
        let expected: Vec<u32> = sample["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect();
        let ids = tok.encodespecial(input).unwrap();
        let got: Vec<u32> = ids.iter().map(|id| id.get()).collect();
        assert_eq!(got, expected, "encode mismatch for {input:?}");
    }
}

#[test]
fn gpt2_decodes_match_the_hf_reference() {
    let tok = loadtokenizer();
    let golden = loadsamples();
    for sample in golden["samples"].as_array().unwrap() {
        let input = sample["input"].as_str().unwrap();
        let ids: Vec<tokenid> = sample["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| tokenid::new(value.as_u64().unwrap() as u32))
            .collect();
        let expected = sample["decoded_no_skip"].as_str().unwrap();
        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(decoded, expected, "decode mismatch for {input:?}");
    }
}
