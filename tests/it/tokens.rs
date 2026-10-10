//! Cross-crate integration tests for the public `tokenizer` API.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use ::tokenizer::tokenizer as tokenizerengine;
use ::tokenizer::{chatmessage, chatrole, chattemplate, chattemplateentry, tokenizererror};
use lib::tokenid;

const SAMPLE: &str = r#"{
    "model": {
        "type": "BPE",
        "vocab": {
            "a": 0, "b": 1, "ab": 2, "c": 3, " ": 10,
            "<0xE4>": 4, "<0xB8>": 5, "<0xAD>": 6, "<unk>": 7
        },
        "merges": ["a b"],
        "byte_fallback": true,
        "unk_token": "<unk>"
    },
    "added_tokens": [
        {"id": 8, "content": "<s>", "special": true},
        {"id": 9, "content": "</s>", "special": true}
    ]
}"#;

const NO_FALLBACK: &str = r#"{
    "model": {
        "type": "BPE",
        "vocab": {"a": 0, "b": 1, "<unk>": 2},
        "merges": [],
        "unk_token": "<unk>"
    }
}"#;

fn raw(ids: &[tokenid]) -> Vec<u32> {
    ids.iter().map(|id| id.get()).collect()
}

#[test]
fn encodes_and_decodes_ascii_roundtrip() {
    let tok = tokenizerengine::fromjson(SAMPLE).unwrap();
    let ids = tok.encode("ab c").unwrap();
    assert_eq!(raw(&ids), vec![2, 10, 3]);
    assert_eq!(tok.decode(&ids).unwrap(), "ab c");
}

#[test]
fn decodes_byte_fallback_utf8_roundtrip() {
    let tok = tokenizerengine::fromjson(SAMPLE).unwrap();
    let ids = tok.encode("\u{4e2d}").unwrap();
    assert_eq!(raw(&ids), vec![4, 5, 6]);
    assert_eq!(tok.decode(&ids).unwrap(), "\u{4e2d}");
}

#[test]
fn unknown_token_falls_back_to_unk_id() {
    let tok = tokenizerengine::fromjson(NO_FALLBACK).unwrap();
    let ids = tok.encode("\u{4e2d}").unwrap();
    assert_eq!(raw(&ids), vec![2]);
}

#[test]
fn special_tokens_are_structural_when_requested() {
    let tok = tokenizerengine::fromjson(SAMPLE).unwrap();
    let ids = tok.encodespecial("<s>ab</s>").unwrap();
    assert_eq!(raw(&ids), vec![8, 2, 9]);
}

#[test]
fn chat_template_prefixes_bos_and_renders_roles() {
    let mut tok = tokenizerengine::fromjson(SAMPLE).unwrap();
    let template = chattemplate::new(
        Some(tokenid::new(8)),
        None,
        vec![chattemplateentry {
            role: chatrole::user,
            prefix: "<s>".to_string(),
            suffix: "</s>".to_string(),
        }],
    );
    tok.setchattemplate(template);
    let messages = vec![chatmessage {
        role: chatrole::user,
        content: "ab".to_string(),
    }];
    let ids = tok.applychattemplate(&messages).unwrap();
    // BOS, then the rendered "<s>ab</s>".
    assert_eq!(raw(&ids), vec![8, 8, 2, 9]);
}

#[test]
fn chat_template_absent_is_reported() {
    let tok = tokenizerengine::fromjson(SAMPLE).unwrap();
    assert_eq!(
        tok.applychattemplate(&[]),
        Err(tokenizererror::nochattemplate)
    );
}

#[test]
fn decode_rejects_ids_absent_from_the_vocabulary() {
    let tok = tokenizerengine::fromjson(SAMPLE).unwrap();
    assert_eq!(
        tok.decode(&[tokenid::new(99)]),
        Err(tokenizererror::unknowntokenid { id: 99 })
    );
}

#[test]
fn malformed_documents_are_rejected_without_panicking() {
    for raw in [
        "not json",
        "[]",
        r#"{"model": {"type": "WordPiece", "vocab": {}}}"#,
        r#"{"model": {"type": "BPE", "vocab": {"a": 0}, "merges": ["ab"]}}"#,
    ] {
        assert!(tokenizerengine::fromjson(raw).is_err(), "accepted {raw:?}");
    }
}
