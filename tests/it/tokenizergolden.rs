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

#[test]
fn o200k_matches_the_tiktoken_reference() {
    use ::tokenizer::tiktokenizer;
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/o200k");
    let content = std::fs::read_to_string(format!("{dir}/o200k_base.tiktoken")).unwrap();
    let specials = [
        ("<|endoftext|>".to_string(), 199_999u32),
        ("<|endofprompt|>".to_string(), 200_018u32),
    ];
    let tok = tiktokenizer::fromfile(&content, &specials).unwrap();
    let golden: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/o200k_golden.json")).unwrap())
            .unwrap();
    for sample in golden["samples"].as_array().unwrap() {
        let text = sample["text"].as_str().unwrap();
        let expected: Vec<u32> = sample["raw_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect();
        let ids = tok.encode(text).unwrap();
        let got: Vec<u32> = ids.iter().map(|id| id.get()).collect();
        assert_eq!(got, expected, "o200k encode mismatch for {text:?}");
        let ids: Vec<tokenid> = expected.iter().map(|rank| tokenid::new(*rank)).collect();
        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(
            decoded,
            sample["decoded"].as_str().unwrap(),
            "o200k decode mismatch for {text:?}"
        );
    }
}

#[test]
fn postprocess_matches_the_wrapped_reference() {
    // `wrapped_ids` is the `add_special_tokens = true` encoding: the raw ids
    // with the post-processor applied (BOS/EOS insertion). Only the models that
    // carry a TemplateProcessing post-processor emit a BOS prefix; ByteLevel
    // alone (qwen3/deepseek) is a no-op, so wrapped == raw.
    for model in MODELS {
        let tok = loadtokenizer(model);
        let golden = loadsamples(model);
        for sample in golden["samples"].as_array().unwrap() {
            let Some(wrapped) = sample
                .get("wrapped_ids")
                .and_then(serde_json::Value::as_array)
            else {
                continue;
            };
            let text = sampletext(sample);
            let expected: Vec<u32> = wrapped
                .iter()
                .map(|value| value.as_u64().unwrap() as u32)
                .collect();
            let raw = tok.encodespecial(text).unwrap();
            let got: Vec<u32> = tok.postprocess(&raw).iter().map(|id| id.get()).collect();
            assert_eq!(got, expected, "{model} postprocess mismatch for {text:?}");
        }
    }
}
