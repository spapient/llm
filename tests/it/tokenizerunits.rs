//! Relocated unit tests for the `tokenizer` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/tokenizer/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/tokenizer/src/chat.rs`.
mod chat {
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
    use ::tokenizer::tokenizer;
    use ::tokenizer::tokenizererror;
    use lib::tokenid;
    use rstest::rstest;

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
    fn encode_into_reuses_scratch_across_calls() {
        let tok = tokenizer::fromjson(SAMPLE).unwrap();
        let mut scratch = ::tokenizer::mergescratch::default();
        let mut out = Vec::new();
        // Largest first, then shorter and empty, so the reused buffers must be
        // correctly reset rather than leaking stale linked-list or run state.
        tok.encode_into("abc", &mut scratch, &mut out).unwrap();
        assert_eq!(out, vec![tokenid::new(2), tokenid::new(3)]);
        out.clear();
        tok.encode_into("ab", &mut scratch, &mut out).unwrap();
        assert_eq!(out, vec![tokenid::new(2)]);
        out.clear();
        tok.encode_into("", &mut scratch, &mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn encode_batch_matches_individual_encodes() {
        let tok = tokenizer::fromjson(SAMPLE).unwrap();
        let texts = ["ab", "abc", "ab", "ac", ""];
        let mut scratch = ::tokenizer::mergescratch::default();
        let mut batched = Vec::new();
        tok.encode_batch(&texts, &mut scratch, &mut batched)
            .unwrap();
        let expected: Vec<Vec<tokenid>> = texts.iter().map(|t| tok.encode(t).unwrap()).collect();
        assert_eq!(batched, expected);
    }

    #[test]
    fn rejects_invalid_json() {
        assert!(matches!(
            tokenizer::fromjson("{"),
            Err(tokenizererror::invalidjson { .. })
        ));
    }

    /// Each case is a malformed document and the exact error the parser must
    /// report for it.
    #[rstest]
    #[case::non_object_root("[]", tokenizererror::rootnotobject)]
    #[case::non_bpe_model(
        r#"{"model": {"type": "WordPiece", "vocab": {}}}"#,
        tokenizererror::unsupportedmodel { modeltype: "WordPiece".to_string() }
    )]
    #[case::non_integer_vocab_id(
        r#"{"model": {"type": "BPE", "vocab": {"a": "x"}}}"#,
        tokenizererror::unexpectedtype { field: "model.vocab", expected: "integer" }
    )]
    #[case::out_of_range_id(
        r#"{"model": {"type": "BPE", "vocab": {"a": 4294967296}}}"#,
        tokenizererror::idoutofrange { id: 4294967296 }
    )]
    #[case::unknown_configured_unknown_token(
        r#"{"model": {"type": "BPE", "vocab": {"a": 0}, "unk_token": "<unk>"}}"#,
        tokenizererror::unknowntoken { token: "<unk>".to_string() }
    )]
    fn rejects_malformed_documents(#[case] json: &str, #[case] expected: tokenizererror) {
        assert_eq!(tokenizer::fromjson(json).unwrap_err(), expected);
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
    use ::tokenizer::bytepair;
    use ::tokenizer::tokenizererror;
    use ::tokenizer::vocabulary;
    use lib::tokenid;
    use proptest::prelude::*;

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

    /// One of `a`..`h`, chosen uniformly as a `String`.
    fn alpha() -> impl Strategy<Value = String> {
        (0u8..8).prop_map(|index| ((b'a' + index) as char).to_string())
    }

    /// Independent naive reference: rescans the whole sequence each step and
    /// resolves each rank by linear search, so it shares no code with
    /// `bytepair::apply`.
    fn naiveapply(merges: &[String], input: &[String]) -> Vec<String> {
        let mut work: Vec<String> = input.to_vec();
        loop {
            let mut best: Option<(usize, usize)> = None;
            for (index, pair) in work.windows(2).enumerate() {
                let needle = format!("{} {}", pair[0], pair[1]);
                if let Some(rank) = merges
                    .iter()
                    .position(|entry| entry.as_str() == needle.as_str())
                    && best.is_none_or(|(bestrank, _)| rank < bestrank)
                {
                    best = Some((rank, index));
                }
            }
            match best {
                Some((_, index)) => {
                    let merged = format!("{}{}", work[index], work[index + 1]);
                    work[index] = merged;
                    work.remove(index + 1);
                }
                None => break,
            }
        }
        work
    }

    proptest! {
        #[test]
        fn apply_matches_an_independent_naive_reference(
            input in proptest::collection::vec(alpha(), 0..24),
            pairs in proptest::collection::vec((alpha(), alpha()), 0..40),
        ) {
            let merges: Vec<String> = pairs
                .iter()
                .map(|(left, right)| format!("{left} {right}"))
                .collect();
            let bpe = bytepair::frommerges(&merges).unwrap();
            prop_assert_eq!(bpe.apply(&input), naiveapply(&merges, &input));
        }
    }

    /// Builds a vocabulary covering every `a`..`h` single character and every
    /// two-character combination, so any single-character merge over this
    /// alphabet has in-vocabulary sides and results.
    fn idvocab() -> vocabulary {
        let chars: Vec<char> = (b'a'..=b'h').map(char::from).collect();
        let mut entries: Vec<(String, u32)> = Vec::new();
        let mut id = 0u32;
        for &c in &chars {
            entries.push((c.to_string(), id));
            id += 1;
        }
        for &l in &chars {
            for &r in &chars {
                let mut s = String::new();
                s.push(l);
                s.push(r);
                entries.push((s, id));
                id += 1;
            }
        }
        vocabulary::fromentries(entries).unwrap()
    }

    proptest! {
        #[test]
        fn applyids_matches_string_apply(
            input in proptest::collection::vec(alpha(), 0..24),
            pairs in proptest::collection::vec((alpha(), alpha()), 0..40),
        ) {
            let merges: Vec<String> = pairs
                .iter()
                .map(|(left, right)| format!("{left} {right}"))
                .collect();
            let vocab = idvocab();
            let bpe = bytepair::frommergesandvocab(&merges, &vocab).unwrap();
            let string_ids: Vec<u32> = bpe
                .apply(&input)
                .iter()
                .map(|symbol| vocab.getid(symbol).unwrap().get())
                .collect();
            let symbols: Vec<tokenid> = input.iter().map(|s| vocab.getid(s).unwrap()).collect();
            let id_ids: Vec<u32> = bpe.applyids(&symbols).iter().map(|id| id.get()).collect();
            prop_assert_eq!(id_ids, string_ids);
        }
    }
}

/// Tests relocated from `crates/tokenizer/src/special.rs`.
mod special {
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

/// Tests relocated from `crates/tokenizer/src/pretoken.rs`.
mod pretoken {
    use ::tokenizer::{
        bytetounicode, gpt2split, o200ksplit, tokenizer, tokenizererror, unicodetobyte,
    };
    use fancy_regex::Regex;
    use proptest::prelude::*;
    use serde_json::json;

    #[test]
    fn byte_to_unicode_is_a_bijection() {
        let forward = bytetounicode();
        let inverse = unicodetobyte();
        for (byte, &character) in forward.iter().enumerate() {
            assert_eq!(inverse[character as usize], Some(byte as u8), "byte {byte}");
        }
    }

    #[test]
    fn byte_to_unicode_matches_known_vectors() {
        let table = bytetounicode();
        assert_eq!(table[0], '\u{100}');
        assert_eq!(table[32], '\u{120}');
        assert_eq!(table[33], '!');
        assert_eq!(table[65], 'A');
        assert_eq!(table[255], '\u{ff}');
    }

    #[test]
    fn gpt2_splitter_matches_known_vectors() {
        assert_eq!(gpt2split("it's"), ["it", "'s"]);
        assert_eq!(gpt2split("don't"), ["don", "'t"]);
        assert_eq!(gpt2split("I've"), ["I", "'ve"]);
        assert_eq!(gpt2split("hello world"), ["hello", " world"]);
        assert_eq!(gpt2split("hello  world"), ["hello", " ", " world"]);
        assert_eq!(gpt2split("2 cats"), ["2", " cats"]);
        assert_eq!(gpt2split("你好"), ["你好"]);
        assert_eq!(gpt2split("中文 test"), ["中文", " test"]);
        assert_eq!(gpt2split(""), Vec::<&str>::new());
    }

    /// Builds a GPT-2-style tokenizer whose vocabulary is exactly the 256
    /// byte-to-unicode characters, with ByteLevel pre-tokenization and
    /// decoding and no merge rules.
    fn byteleveljson() -> String {
        let mut vocab = serde_json::Map::new();
        for (byte, character) in bytetounicode().iter().enumerate() {
            vocab.insert(character.to_string(), json!(byte));
        }
        json!({
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": [],
                "byte_fallback": false,
            },
            "pre_tokenizer": {"type": "ByteLevel"},
            "decoder": {"type": "ByteLevel"},
        })
        .to_string()
    }

    #[test]
    fn bytelevel_encode_decode_roundtrip() {
        let tok = tokenizer::fromjson(&byteleveljson()).unwrap();
        for text in ["hello world", "it's a test", "你好，世界！", "123 456", ""] {
            let ids = tok.encode(text).unwrap();
            assert_eq!(tok.decode(&ids).unwrap(), text, "roundtrip for {text:?}");
        }
    }

    #[test]
    fn bytelevel_encodes_bytes_not_characters() {
        let tok = tokenizer::fromjson(&byteleveljson()).unwrap();
        // "你好" is two scalars but six UTF-8 bytes; byte-level encoding yields
        // one token per byte.
        assert_eq!(tok.encode("你好").unwrap().len(), 6);
    }

    #[test]
    fn rejects_unsupported_pre_tokenizer() {
        let doc = r#"{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"pre_tokenizer":{"type":"WordPiece"}}"#;
        assert!(matches!(
            tokenizer::fromjson(doc),
            Err(tokenizererror::unsupportedpretokenizer { .. })
        ));
    }

    #[test]
    fn rejects_unsupported_decoder() {
        let doc = r#"{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"decoder":{"type":"CTC"}}"#;
        assert!(matches!(
            tokenizer::fromjson(doc),
            Err(tokenizererror::unsupporteddecoder { .. })
        ));
    }

    #[test]
    fn special_longest_match_wins() {
        let doc = r#"{
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
            "added_tokens": [
                {"id": 2, "content": "<", "special": true},
                {"id": 3, "content": "<s>", "special": true}
            ]
        }"#;
        let tok = tokenizer::fromjson(doc).unwrap();
        assert_eq!(
            tok.encodespecial("<s>").unwrap(),
            vec![lib::tokenid::new(3)]
        );
    }

    #[test]
    fn bytelevel_roundtrip_with_repeated_words() {
        let tok = tokenizer::fromjson(&byteleveljson()).unwrap();
        let text = "the cat and the dog and the cat";
        let ids = tok.encode(text).unwrap();
        assert_eq!(tok.decode(&ids).unwrap(), text);
    }

    /// Reference splitter over the o200k pattern, filling gaps between matches
    /// so it yields the same contiguous chunks as [`o200ksplit`].
    fn o200kregex(text: &str) -> Vec<String> {
        const O200K: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        let regex = Regex::new(O200K).unwrap();
        let mut out = Vec::new();
        let mut last = 0;
        for found in regex.find_iter(text) {
            let found = found.unwrap();
            if found.start() > last {
                out.push(text[last..found.start()].to_string());
            }
            out.push(text[found.start()..found.end()].to_string());
            last = found.end();
        }
        if last < text.len() {
            out.push(text[last..].to_string());
        }
        out
    }

    fn o200kchar() -> impl Strategy<Value = char> {
        prop_oneof![
            4 => prop::char::range('a', 'z'),
            2 => prop::char::range('A', 'Z'),
            2 => prop::char::range('0', '9'),
            4 => Just(' '),
            2 => Just('\t'),
            2 => Just('\n'),
            1 => Just('\r'),
            1 => Just('\''),
            1 => Just('!'),
            1 => Just('?'),
            1 => Just('.'),
            1 => Just(','),
            1 => Just('é'),
            1 => Just('中'),
        ]
    }

    proptest! {
        #[test]
        fn o200ksplit_matches_the_regex_reference(
            chars in proptest::collection::vec(o200kchar(), 0..40),
        ) {
            let text: String = chars.into_iter().collect();
            let got: Vec<String> = o200ksplit(&text).iter().map(|s| s.to_string()).collect();
            prop_assert_eq!(got, o200kregex(&text));
        }
    }
}

/// Tests for the TemplateProcessing post-processor.
mod postprocess {
    use ::tokenizer::tokenizer;
    use lib::tokenid;

    const DOC: &str = r#"{
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}, "merges": []},
        "added_tokens": [
            {"id": 3, "content": "<s>", "special": true},
            {"id": 4, "content": "</s>", "special": true}
        ],
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}}
            ],
            "pair": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}}
            ]
        }
    }"#;

    fn ids(raw: &[u32]) -> Vec<tokenid> {
        raw.iter().map(|r| tokenid::new(*r)).collect()
    }

    #[test]
    fn applies_single_and_pair_templates() {
        let tok = tokenizer::fromjson(DOC).unwrap();
        let encoded = tok.encode("ab").unwrap();
        assert_eq!(encoded, ids(&[0, 1]));
        assert_eq!(tok.postprocess(&encoded), ids(&[3, 0, 1, 4]));
        assert_eq!(
            tok.postprocesspair(&encoded, &ids(&[2])),
            ids(&[3, 0, 1, 4, 2, 4])
        );
    }

    #[test]
    fn is_the_identity_without_a_post_processor() {
        let tok =
            tokenizer::fromjson(r#"{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#).unwrap();
        let encoded = tok.encode("a").unwrap();
        assert_eq!(tok.postprocess(&encoded), encoded);
    }
}

/// Tests for the Jinja chat-template subset.
mod jinja {
    use ::tokenizer::{chatmessage, chatrole, tokenizer};

    const LOOP: &str = r#"{
        "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
        "bos_token": "<s>",
        "eos_token": "</s>",
        "chat_template": "{{ bos_token }}{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}{% if not loop.last %}|{% endif %}{% endfor %}"
    }"#;

    const BRANCH: &str = r#"{
        "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
        "chat_template": "{% for message in messages %}{% if message['role'] == 'system' %}SYS:{% else %}{{ message['role'] }}:{% endif %}{{ message['content'] }}{% endfor %}"
    }"#;

    fn conversation() -> Vec<chatmessage> {
        vec![
            chatmessage {
                role: chatrole::user,
                content: "hi".to_string(),
            },
            chatmessage {
                role: chatrole::assistant,
                content: "hello".to_string(),
            },
        ]
    }

    #[test]
    fn renders_loop_and_interpolation() {
        let tok = tokenizer::fromjson(LOOP).unwrap();
        assert_eq!(
            tok.renderchat(&conversation()).unwrap(),
            "<s>user:hi|assistant:hello"
        );
    }

    #[test]
    fn renders_if_else_branching() {
        let tok = tokenizer::fromjson(BRANCH).unwrap();
        let messages = vec![
            chatmessage {
                role: chatrole::system,
                content: "be brief".to_string(),
            },
            chatmessage {
                role: chatrole::user,
                content: "hi".to_string(),
            },
        ];
        assert_eq!(tok.renderchat(&messages).unwrap(), "SYS:be briefuser:hi");
    }
}

/// Tests for the `normalizer` section.
mod normalize {
    use ::tokenizer::{normalizer, tokenizer};
    use lib::tokenid;

    #[test]
    fn applies_lowercase() {
        let doc = r#"{"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]},"normalizer":{"type":"Lowercase"}}"#;
        let tok = tokenizer::fromjson(doc).unwrap();
        assert_eq!(tok.encode("A").unwrap(), vec![tokenid::new(0)]);
    }

    #[test]
    fn applies_a_sequence_in_order() {
        let doc = r#"{
            "model": {"type": "BPE", "vocab": {"a":0,"b":1}, "merges": []},
            "normalizer": {"type": "Sequence", "normalizers": [{"type": "Strip"}, {"type": "Lowercase"}]}
        }"#;
        let tok = tokenizer::fromjson(doc).unwrap();
        assert_eq!(tok.encode("  A  ").unwrap(), vec![tokenid::new(0)]);
    }

    #[test]
    fn nfc_normalizes_combining_sequences() {
        // "é" as e + U+0301 combining acute must NFC-normalize to U+00E9.
        assert_eq!(normalizer::nfc.apply("e\u{0301}"), "\u{e9}");
    }

    #[test]
    fn null_normalizer_is_ignored() {
        let doc = r#"{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"normalizer":null}"#;
        let tok = tokenizer::fromjson(doc).unwrap();
        assert_eq!(tok.encode("a").unwrap(), vec![tokenid::new(0)]);
    }

    #[test]
    fn normalizer_applies_directly() {
        assert_eq!(normalizer::lowercase.apply("AbC"), "abc");
        assert_eq!(normalizer::strip.apply("  x  "), "x");
        assert_eq!(
            normalizer::sequence(vec![normalizer::strip, normalizer::lowercase]).apply("  AbC  "),
            "abc"
        );
    }
}
