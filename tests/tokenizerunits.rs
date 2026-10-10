//! Relocated unit tests for the `tokenizer` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/tokenizer/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/decision-records/blackboxtesting.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/tokenizer/src/chat.rs`.
mod chat {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::tokenizer::chatmessage;
    use ::tokenizer::chatrole;
    use ::tokenizer::chattemplate;
    use ::tokenizer::chattemplateentry;
    use ::tokenizer::tokenizererror;
    use lib::tokenid;

    fn template() -> chattemplate {
        chattemplate::new(
            Some(tokenid::new(8)),
            Some(tokenid::new(9)),
            vec![
                chattemplateentry {
                    role: chatrole::user,
                    prefix: "[user]".to_string(),
                    suffix: "[/user]".to_string(),
                },
                chattemplateentry {
                    role: chatrole::assistant,
                    prefix: "[assistant]".to_string(),
                    suffix: "[/assistant]".to_string(),
                },
            ],
        )
    }

    #[test]
    fn role_names_are_stable() {
        assert_eq!(chatrole::system.getname(), "system");
        assert_eq!(chatrole::user.getname(), "user");
        assert_eq!(chatrole::assistant.getname(), "assistant");
    }

    #[test]
    fn renders_wrapped_messages() {
        let t = template();
        let messages = vec![
            chatmessage {
                role: chatrole::user,
                content: "hi".to_string(),
            },
            chatmessage {
                role: chatrole::assistant,
                content: "hello".to_string(),
            },
        ];
        assert_eq!(
            t.render(&messages).unwrap(),
            "[user]hi[/user][assistant]hello[/assistant]"
        );
        assert_eq!(t.getbos(), Some(tokenid::new(8)));
        assert_eq!(t.geteos(), Some(tokenid::new(9)));
    }

    #[test]
    fn rejects_uncovered_role() {
        let t = template();
        let messages = vec![chatmessage {
            role: chatrole::system,
            content: "x".to_string(),
        }];
        assert_eq!(
            t.render(&messages),
            Err(tokenizererror::unsupportedrole { role: "system" })
        );
    }

    #[test]
    fn renders_empty_conversation_to_empty_string() {
        let t = template();
        assert_eq!(t.render(&[]).unwrap(), "");
    }
}

/// Tests relocated from `crates/tokenizer/src/decode.rs`.
///
/// The former unit tests called the crate-private `decodetokens` and
/// `parsebytetoken`. Both are reachable through the public `tokenizer::decode`,
/// so the decode contract is verified there.
mod decode {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::tokenizer::{tokenizer, tokenizererror};
    use lib::tokenid;

    /// Two plain tokens and the three UTF-8 byte tokens for U+4E2D.
    const BYTES: &str = r#"{
        "model": {
            "type": "BPE",
            "vocab": {
                "a": 0, "bc": 1,
                "<0xE4>": 4, "<0xB8>": 5, "<0xAD>": 6
            },
            "merges": []
        }
    }"#;

    fn tok() -> tokenizer {
        tokenizer::fromjson(BYTES).unwrap()
    }

    #[test]
    fn decodes_plain_tokens() {
        assert_eq!(
            tok().decode(&[tokenid::new(0), tokenid::new(1)]).unwrap(),
            "abc"
        );
    }

    #[test]
    fn reconstructs_utf8_from_byte_tokens() {
        assert_eq!(
            tok()
                .decode(&[tokenid::new(4), tokenid::new(5), tokenid::new(6)])
                .unwrap(),
            "\u{4e2d}"
        );
    }

    #[test]
    fn empty_ids_decode_to_empty_string() {
        assert_eq!(tok().decode(&[]).unwrap(), "");
    }

    #[test]
    fn rejects_ids_absent_from_the_vocabulary() {
        assert_eq!(
            tok().decode(&[tokenid::new(9)]),
            Err(tokenizererror::unknowntokenid { id: 9 })
        );
    }

    #[test]
    fn rejects_incomplete_utf8_byte_sequences() {
        assert_eq!(
            tok().decode(&[tokenid::new(4), tokenid::new(5)]),
            Err(tokenizererror::invalidutf8)
        );
    }

    #[test]
    fn malformed_angle_tokens_decode_as_literal_text() {
        // `<0xZZ>` is not hexadecimal and `<0x4>` is too short, so neither is a
        // byte token; they decode verbatim instead of contributing bytes.
        let json = r#"{"model":{"type":"BPE","vocab":{"<0xZZ>":0,"<0x4>":1,"A":2},"merges":[]}}"#;
        let t = tokenizer::fromjson(json).unwrap();
        assert_eq!(
            t.decode(&[tokenid::new(0), tokenid::new(1), tokenid::new(2)])
                .unwrap(),
            "<0xZZ><0x4>A"
        );
    }
}

/// Tests relocated from `crates/tokenizer/src/encode.rs`.
///
/// The former unit tests called the crate-private `encodeplain` and
/// `encodespecial`. Both are reachable through `tokenizer::encode` and
/// `tokenizer::encodespecial`, so the encode contract is verified there.
mod encode {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::tokenizer::{tokenizer, tokenizererror};
    use lib::tokenid;

    /// Byte-fallback BPE with one merge, a shared unknown token, and `<s>`.
    const WITHFALLBACK: &str = r#"{
        "model": {
            "type": "BPE",
            "vocab": {"a":0,"b":1,"ab":2,"c":3,"<0xE4>":4,"<0xB8>":5,"<0xAD>":6,"<unk>":7},
            "merges": ["a b"],
            "byte_fallback": true,
            "unk_token": "<unk>"
        },
        "added_tokens": [{"id": 8, "content": "<s>", "special": true}]
    }"#;

    /// The same model without byte fallback, still with an unknown token.
    const NOFALLBACKUNK: &str = r#"{
        "model": {
            "type": "BPE",
            "vocab": {"a":0,"b":1,"ab":2,"c":3,"<0xE4>":4,"<0xB8>":5,"<0xAD>":6,"<unk>":7},
            "merges": ["a b"],
            "byte_fallback": false,
            "unk_token": "<unk>"
        }
    }"#;

    /// The same model with neither byte fallback nor an unknown token.
    const NOFALLBACKNOUNK: &str = r#"{
        "model": {
            "type": "BPE",
            "vocab": {"a":0,"b":1,"ab":2,"c":3,"<0xE4>":4,"<0xB8>":5,"<0xAD>":6,"<unk>":7},
            "merges": ["a b"],
            "byte_fallback": false
        }
    }"#;

    fn tok(json: &str) -> tokenizer {
        tokenizer::fromjson(json).unwrap()
    }

    #[test]
    fn empty_text_encodes_to_nothing() {
        assert_eq!(tok(WITHFALLBACK).encode("").unwrap(), Vec::<tokenid>::new());
    }

    #[test]
    fn applies_merges_before_lookup() {
        let t = tok(WITHFALLBACK);
        assert_eq!(t.encode("ab").unwrap(), vec![tokenid::new(2)]);
        assert_eq!(
            t.encode("ac").unwrap(),
            vec![tokenid::new(0), tokenid::new(3)]
        );
    }

    #[test]
    fn byte_fallback_covers_unknown_symbols() {
        assert_eq!(
            tok(WITHFALLBACK).encode("\u{4e2d}").unwrap(),
            vec![tokenid::new(4), tokenid::new(5), tokenid::new(6)]
        );
    }

    #[test]
    fn unknown_symbol_without_fallback_is_an_error() {
        assert_eq!(
            tok(NOFALLBACKNOUNK).encode("\u{4e2d}"),
            Err(tokenizererror::unknowntoken {
                token: "\u{4e2d}".to_string(),
            })
        );
    }

    #[test]
    fn unknown_symbol_falls_back_to_unknown_token() {
        assert_eq!(
            tok(NOFALLBACKUNK).encode("\u{4e2d}").unwrap(),
            vec![tokenid::new(7)]
        );
    }

    #[test]
    fn special_literals_split_the_input() {
        assert_eq!(
            tok(WITHFALLBACK).encodespecial("<s>ab").unwrap(),
            vec![tokenid::new(8), tokenid::new(2)]
        );
    }

    #[test]
    fn plain_encoding_does_not_recognize_specials() {
        // `<s>` is three ordinary characters; with no byte entries for '<' or
        // '>', each resolves to the unknown token.
        assert_eq!(
            tok(NOFALLBACKUNK).encode("<s>").unwrap(),
            vec![tokenid::new(7), tokenid::new(7), tokenid::new(7)]
        );
    }
}

/// Tests relocated from `crates/tokenizer/src/engine.rs`.
mod engine {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::tokenizer::tokenizer;
    use ::tokenizer::tokenizererror;
    use lib::tokenid;

    const SAMPLE: &str = r#"{
    "model": {
        "type": "BPE",
        "vocab": {
            "a": 0, "b": 1, "ab": 2, "c": 3,
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

    #[test]
    fn loads_and_exposes_components() {
        let tok = tokenizer::fromjson(SAMPLE).unwrap();
        assert_eq!(tok.getvocabulary().getcount(), 10);
        assert_eq!(tok.getbytepair().getcount(), 1);
        assert_eq!(tok.getspecialtokens().getcount(), 2);
        assert!(tok.getchattemplate().is_none());
    }

    #[test]
    fn rejects_invalid_json() {
        assert!(matches!(
            tokenizer::fromjson("{"),
            Err(tokenizererror::invalidjson { .. })
        ));
    }

    #[test]
    fn rejects_non_object_root() {
        assert_eq!(
            tokenizer::fromjson("[]").unwrap_err(),
            tokenizererror::rootnotobject
        );
    }

    #[test]
    fn rejects_missing_model_type() {
        let json = r#"{"model": {"vocab": {}}}"#;
        assert_eq!(
            tokenizer::fromjson(json).unwrap_err(),
            tokenizererror::missingfield {
                field: "model.type"
            }
        );
    }

    #[test]
    fn rejects_non_bpe_model() {
        let json = r#"{"model": {"type": "WordPiece", "vocab": {}}}"#;
        assert_eq!(
            tokenizer::fromjson(json).unwrap_err(),
            tokenizererror::unsupportedmodel {
                modeltype: "WordPiece".to_string()
            }
        );
    }

    #[test]
    fn rejects_non_integer_vocab_id() {
        let json = r#"{"model": {"type": "BPE", "vocab": {"a": "x"}}}"#;
        assert_eq!(
            tokenizer::fromjson(json).unwrap_err(),
            tokenizererror::unexpectedtype {
                field: "model.vocab",
                expected: "integer"
            }
        );
    }

    #[test]
    fn rejects_out_of_range_id() {
        let json = r#"{"model": {"type": "BPE", "vocab": {"a": 4294967296}}}"#;
        assert_eq!(
            tokenizer::fromjson(json).unwrap_err(),
            tokenizererror::idoutofrange { id: 4294967296 }
        );
    }

    #[test]
    fn rejects_unknown_configured_unknown_token() {
        let json = r#"{"model": {"type": "BPE", "vocab": {"a": 0}, "unk_token": "<unk>"}}"#;
        assert_eq!(
            tokenizer::fromjson(json).unwrap_err(),
            tokenizererror::unknowntoken {
                token: "<unk>".to_string()
            }
        );
    }

    #[test]
    fn encode_decode_roundtrip() {
        let tok = tokenizer::fromjson(SAMPLE).unwrap();
        let ids = tok.encode("ab").unwrap();
        assert_eq!(ids, vec![tokenid::new(2)]);
        assert_eq!(tok.decode(&ids).unwrap(), "ab");
    }

    #[test]
    fn special_aware_encoding_emits_markers() {
        let tok = tokenizer::fromjson(SAMPLE).unwrap();
        assert_eq!(
            tok.encodespecial("<s>ab</s>").unwrap(),
            vec![tokenid::new(8), tokenid::new(2), tokenid::new(9)]
        );
    }

    #[test]
    fn apply_chat_template_requires_a_template() {
        let tok = tokenizer::fromjson(SAMPLE).unwrap();
        assert_eq!(
            tok.applychattemplate(&[]),
            Err(tokenizererror::nochattemplate)
        );
    }
}

/// Tests relocated from `crates/tokenizer/src/error.rs`.
mod error {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::tokenizer::tokenizererror;

    #[test]
    fn carries_structured_context() {
        assert_eq!(
            tokenizererror::invalidjson { line: 3, column: 7 }.to_string(),
            "invalid JSON at line 3 column 7"
        );
        assert_eq!(
            tokenizererror::missingfield {
                field: "model.type"
            }
            .to_string(),
            "missing required field: model.type"
        );
        assert_eq!(
            tokenizererror::unknowntoken { token: "x".into() }.to_string(),
            "token \"x\" is not in the vocabulary and has no fallback"
        );
    }

    #[test]
    fn variants_are_distinguishable() {
        assert_ne!(tokenizererror::rootnotobject, tokenizererror::invalidutf8);
        assert_ne!(
            tokenizererror::unsupportedrole { role: "tool" },
            tokenizererror::unsupportedrole { role: "user" }
        );
        assert_ne!(
            tokenizererror::unsupportedmodel {
                modeltype: "WordPiece".into(),
            },
            tokenizererror::unsupportedmodel {
                modeltype: "BPE".into(),
            }
        );
    }
}

/// Tests relocated from `crates/tokenizer/src/merges.rs`.
mod merges {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::tokenizer::bytepair;
    use ::tokenizer::tokenizererror;

    fn symbols(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn parses_valid_merges_and_reports_rank() {
        let bpe = bytepair::frommerges(&["a b".to_string(), "ab c".to_string()]).unwrap();
        assert_eq!(bpe.getcount(), 2);
        assert_eq!(bpe.getrank("a", "b"), Some(0));
        assert_eq!(bpe.getrank("ab", "c"), Some(1));
        assert_eq!(bpe.getrank("b", "a"), None);
    }

    #[test]
    fn rejects_entries_without_separator() {
        assert_eq!(
            bytepair::frommerges(&["ab".to_string()]),
            Err(tokenizererror::malformedmerge { index: 0 })
        );
    }

    #[test]
    fn rejects_entries_with_empty_side() {
        assert_eq!(
            bytepair::frommerges(&["a b".to_string(), "a ".to_string()]),
            Err(tokenizererror::malformedmerge { index: 1 })
        );
        assert_eq!(
            bytepair::frommerges(&[" a".to_string()]),
            Err(tokenizererror::malformedmerge { index: 0 })
        );
    }

    #[test]
    fn merges_lowest_rank_first() {
        // Merging `a b` (rank 1) before `b c` (rank 0) would yield `["ab", "c"]`,
        // so the result distinguishes the precedence order.
        let bpe = bytepair::frommerges(&["b c".to_string(), "a b".to_string()]).unwrap();
        assert_eq!(bpe.apply(&symbols(&["a", "b", "c"])), symbols(&["a", "bc"]));
    }

    #[test]
    fn merges_repeatedly_to_a_single_symbol() {
        let bpe = bytepair::frommerges(&["a b".to_string(), "ab c".to_string()]).unwrap();
        assert_eq!(bpe.apply(&symbols(&["a", "b", "c"])), symbols(&["abc"]));
    }

    #[test]
    fn leaves_unmergeable_sequences_unchanged() {
        let bpe = bytepair::frommerges(&["x y".to_string()]).unwrap();
        assert_eq!(bpe.apply(&symbols(&["a", "b"])), symbols(&["a", "b"]));
        assert_eq!(bpe.apply(&symbols(&["a"])), symbols(&["a"]));
        assert_eq!(bpe.apply(&[]), Vec::<String>::new());
    }

    #[test]
    fn prefers_leftmost_among_equal_ranks() {
        let bpe = bytepair::frommerges(&["a b".to_string()]).unwrap();
        assert_eq!(
            bpe.apply(&symbols(&["a", "b", "a", "b"])),
            symbols(&["ab", "ab"])
        );
    }
}

/// Tests relocated from `crates/tokenizer/src/special.rs`.
mod special {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::tokenizer::specialtokens;
    use lib::tokenid;

    fn tokens() -> specialtokens {
        specialtokens::new(vec![
            ("<s>".to_string(), 8),
            ("</s>".to_string(), 9),
            ("<unk>".to_string(), 7),
        ])
    }

    #[test]
    fn resolves_both_directions() {
        let s = tokens();
        assert_eq!(s.getid("<s>"), Some(tokenid::new(8)));
        assert_eq!(s.getcontent(tokenid::new(9)), Some("</s>"));
        assert_eq!(s.getid("<missing>"), None);
        assert_eq!(s.getcontent(tokenid::new(1)), None);
    }

    #[test]
    fn contains_and_count() {
        let s = tokens();
        assert!(s.contains("<unk>"));
        assert!(!s.contains("unk"));
        assert_eq!(s.getcount(), 3);
    }

    #[test]
    fn iterates_in_content_order() {
        let s = tokens();
        let contents: Vec<&str> = s.iter().map(|(content, _)| content).collect();
        assert_eq!(contents, vec!["</s>", "<s>", "<unk>"]);
    }

    #[test]
    fn empty_set_is_empty() {
        let s = specialtokens::default();
        assert_eq!(s.getcount(), 0);
        assert!(s.iter().next().is_none());
    }
}

/// Tests relocated from `crates/tokenizer/src/vocab.rs`.
mod vocab {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::tokenizer::tokenizererror;
    use ::tokenizer::vocabulary;
    use lib::tokenid;

    fn entries() -> Vec<(String, u32)> {
        vec![
            ("a".to_string(), 0),
            ("b".to_string(), 1),
            ("ab".to_string(), 2),
        ]
    }

    #[test]
    fn resolves_both_directions() {
        let v = vocabulary::fromentries(entries()).unwrap();
        assert_eq!(v.getid("ab"), Some(tokenid::new(2)));
        assert_eq!(v.getcontent(tokenid::new(2)), Some("ab"));
        assert_eq!(v.getid("missing"), None);
        assert_eq!(v.getcontent(tokenid::new(9)), None);
    }

    #[test]
    fn reports_count_and_bounds() {
        let v = vocabulary::fromentries(entries()).unwrap();
        assert_eq!(v.getcount(), 3);
        assert_eq!(v.getminid(), Some(tokenid::new(0)));
        assert_eq!(v.getmaxid(), Some(tokenid::new(2)));
    }

    #[test]
    fn rejects_one_id_for_two_tokens() {
        let bad = vec![("a".to_string(), 0), ("b".to_string(), 0)];
        assert_eq!(
            vocabulary::fromentries(bad),
            Err(tokenizererror::duplicateid { id: 0 })
        );
    }

    #[test]
    fn rejects_one_token_for_two_ids() {
        let bad = vec![("a".to_string(), 0), ("a".to_string(), 1)];
        assert_eq!(
            vocabulary::fromentries(bad),
            Err(tokenizererror::duplicatecontent {
                token: "a".to_string(),
            })
        );
    }

    #[test]
    fn accepts_repeated_identical_pairs() {
        let repeated = vec![("a".to_string(), 0), ("a".to_string(), 0)];
        let v = vocabulary::fromentries(repeated).unwrap();
        assert_eq!(v.getcount(), 1);
    }

    #[test]
    fn empty_vocabulary_has_no_bounds() {
        let v = vocabulary::fromentries(Vec::new()).unwrap();
        assert_eq!(v.getcount(), 0);
        assert_eq!(v.getminid(), None);
        assert_eq!(v.getmaxid(), None);
    }
}
