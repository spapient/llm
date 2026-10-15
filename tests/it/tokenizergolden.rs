//! Golden alignment against real `tokenizer.json` files and HF `tokenizers`
//! reference output under `tests/fixtures/<model>/`.
//!
//! The golden was produced with `add_special_tokens = false`, which still
//! recognizes special tokens inside the text, so encoding is compared against
//! [`tokenizer::encodespecial`]. Decoding is compared against the no-skip
//! variant (`decoded_no_skip` for gpt2, `decoded` for the others) because this
//! tokenizer never skips special tokens.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use ::tokenizer::tokenizer as tokenizerengine;
use lib::tokenid;

const MODELS: [&str; 6] = ["gpt2", "llama2", "llama3", "qwen3", "deepseek", "gemma"];

fn loadtokenizer(model: &str) -> tokenizerengine {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let path = format!("{dir}/{model}/tokenizer.json");
    let document = std::fs::read_to_string(path).unwrap();
    tokenizerengine::fromjson(&document).unwrap()
}

fn loadsamples(model: &str) -> serde_json::Value {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let path = format!("{dir}/{model}/{model}_golden.json");
    let text = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn sampletext(sample: &serde_json::Value) -> &str {
    sample
        .get("input")
        .or_else(|| sample.get("text"))
        .and_then(serde_json::Value::as_str)
        .unwrap()
}

fn samplerawids(sample: &serde_json::Value) -> Vec<u32> {
    sample
        .get("ids")
        .or_else(|| sample.get("raw_ids"))
        .and_then(serde_json::Value::as_array)
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as u32)
        .collect()
}

#[test]
fn encodes_match_the_hf_reference() {
    for model in MODELS {
        let tok = loadtokenizer(model);
        let golden = loadsamples(model);
        for sample in golden["samples"].as_array().unwrap() {
            let input = sampletext(sample);
            let expected = samplerawids(sample);
            let ids = tok.encodespecial(input).unwrap();
            let got: Vec<u32> = ids.iter().map(|id| id.get()).collect();
            assert_eq!(got, expected, "{model} encode mismatch for {input:?}");
        }
    }
}

#[test]
fn decodes_match_the_hf_reference() {
    for model in MODELS {
        let tok = loadtokenizer(model);
        let golden = loadsamples(model);
        for sample in golden["samples"].as_array().unwrap() {
            let input = sampletext(sample);
            let ids: Vec<tokenid> = samplerawids(sample).into_iter().map(tokenid::new).collect();
            let expected = sample
                .get("decoded_no_skip")
                .or_else(|| sample.get("decoded"))
                .and_then(serde_json::Value::as_str)
                .unwrap();
            // A special-token-only sample decodes to "" under
            // `skip_special_tokens=true` (the golden's `decoded` field), a quirk
            // this tokenizer does not reproduce (it never skips specials).
            if expected.is_empty() && !ids.is_empty() {
                continue;
            }
            let decoded = tok.decode(&ids).unwrap();
            assert_eq!(decoded, expected, "{model} decode mismatch for {input:?}");
        }
    }
}
