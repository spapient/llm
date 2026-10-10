//! Relocated unit tests for the `format` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/format/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/format/src/configs.rs`.
mod configs {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::format::formaterror;
    use ::format::modelconfig;
    use tensor::dtype;

    fn valid() -> &'static str {
        r#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "num_hidden_layers": 4,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "vocab_size": 256,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
        "tie_word_embeddings": false,
        "vendor_extra": {"ignored": true}
    }"#
    }

    #[test]
    fn parses_a_valid_configuration() {
        let c = modelconfig::parse(valid().as_bytes()).unwrap();
        assert_eq!(c.gethiddensize(), 64);
        assert_eq!(c.getintermediatesize(), 128);
        assert_eq!(c.getattentionheads(), 8);
        assert_eq!(c.getkeyvalueheads(), 2);
        assert_eq!(c.gethiddenlayers(), 4);
        assert_eq!(c.getmaxpositions(), 128);
        assert_eq!(c.getvocabsize(), 256);
        assert_eq!(c.getheadsize(), 8);
        assert_eq!(c.getkeyvaluegroups(), 4);
        assert_eq!(c.gettorchdtype(), Some(dtype::f32));
        assert_eq!(c.gettiewordembeddings(), Some(false));
        assert_eq!(c.getarchitectures(), &["LlamaForCausalLM".to_string()]);
    }

    #[test]
    fn key_value_heads_default_to_attention_heads() {
        let json = r#"{
        "model_type": "llama",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "max_position_embeddings": 8,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32,
        "rope_theta": 10000.0
    }"#;
        let c = modelconfig::parse(json.as_bytes()).unwrap();
        assert_eq!(c.getkeyvalueheads(), 4);
        assert_eq!(c.getkeyvaluegroups(), 1);
        assert_eq!(c.gettorchdtype(), None);
        assert_eq!(c.gettiewordembeddings(), None);
    }

    #[test]
    fn unsupported_model_type_is_rejected() {
        let json = valid().replace("\"llama\"", "\"gpt2\"");
        assert_eq!(
            modelconfig::parse(json.as_bytes()),
            Err(formaterror::unsupportedmodeltype {
                found: "gpt2".to_string(),
            })
        );
    }

    #[test]
    fn missing_required_field_is_reported_as_invalid_json() {
        let json = r#"{"model_type":"llama"}"#;
        assert!(matches!(
            modelconfig::parse(json.as_bytes()),
            Err(formaterror::invalidjson { .. })
        ));
    }

    #[test]
    fn zero_size_is_rejected() {
        let json = valid().replace("\"hidden_size\": 64", "\"hidden_size\": 0");
        assert_eq!(
            modelconfig::parse(json.as_bytes()),
            Err(formaterror::invalidvalue {
                field: "hidden_size",
                reason: "must be non-zero".to_string(),
            })
        );
    }

    #[test]
    fn indivisible_head_count_is_rejected() {
        let json = valid().replace("\"num_attention_heads\": 8", "\"num_attention_heads\": 7");
        assert_eq!(
            modelconfig::parse(json.as_bytes()),
            Err(formaterror::invalidvalue {
                field: "hidden_size",
                reason: "not divisible by num_attention_heads".to_string(),
            })
        );
    }

    #[test]
    fn indivisible_key_value_heads_is_rejected() {
        let json = valid().replace("\"num_key_value_heads\": 2", "\"num_key_value_heads\": 3");
        assert_eq!(
            modelconfig::parse(json.as_bytes()),
            Err(formaterror::invalidvalue {
                field: "num_attention_heads",
                reason: "not divisible by num_key_value_heads".to_string(),
            })
        );
    }

    #[test]
    fn non_positive_epsilon_is_rejected() {
        let json = valid().replace("\"rms_norm_eps\": 1e-5", "\"rms_norm_eps\": 0.0");
        assert_eq!(
            modelconfig::parse(json.as_bytes()),
            Err(formaterror::invalidvalue {
                field: "rms_norm_eps",
                reason: "must be finite and positive".to_string(),
            })
        );
    }

    #[test]
    fn unsupported_torch_dtype_is_rejected() {
        let json = valid().replace("float32", "int8");
        assert_eq!(
            modelconfig::parse(json.as_bytes()),
            Err(formaterror::unsupportedtorchdtype {
                found: "int8".to_string(),
            })
        );
    }

    #[test]
    fn malformed_json_is_rejected() {
        assert!(matches!(
            modelconfig::parse(b"{not json"),
            Err(formaterror::invalidjson { .. })
        ));
    }
}

/// Tests relocated from `crates/format/src/error.rs`.
mod error {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::format::formaterror;

    #[test]
    fn variants_are_distinguishable() {
        assert_ne!(
            formaterror::truncated {
                needed: 8,
                available: 4,
            },
            formaterror::emptyshape {
                name: "w".to_string(),
            }
        );
    }

    #[test]
    fn display_reports_carried_values() {
        let e = formaterror::unsupporteddtype {
            name: "w".to_string(),
            dtype: "BOOL".to_string(),
        };
        assert_eq!(e.to_string(), "unsupported dtype \"BOOL\" for tensor \"w\"");
    }
}

/// Tests relocated from `crates/format/src/metadata.rs`.
///
/// `tensormetadata` values are only produced by the validating parser, so the
/// metadata rules are exercised through the public `safetensorfile::parse`
/// seam; the dtype-token maps are public functions and are tested directly.
mod metadata {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::format::{dtypetoken, formaterror, safetensorfile, tokendtype};
    use ::lib::limits;
    use ::tensor::dtype;

    fn permissive() -> limits {
        limits::new(usize::MAX, usize::MAX, usize::MAX, usize::MAX)
    }

    fn artifact(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn token_roundtrips_for_every_supported_dtype() {
        for dt in [
            dtype::f32,
            dtype::f64,
            dtype::f16,
            dtype::bf16,
            dtype::i8,
            dtype::u8,
            dtype::i32,
            dtype::i64,
        ] {
            assert_eq!(tokendtype(dtypetoken(dt)), Some(dt));
        }
    }

    #[test]
    fn unsupported_tokens_return_none() {
        for token in ["BOOL", "I16", "U16", "U32", "U64", "F8_E4M3", "f32", ""] {
            assert_eq!(tokendtype(token), None);
        }
    }

    #[test]
    fn valid_entry_reports_all_fields() {
        let header = r#"{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[4,28]}}"#;
        let bytes = artifact(header, &[0u8; 28]);
        let file = safetensorfile::parse(&bytes, &permissive()).unwrap();
        let m = file.gettensor("w").unwrap();
        assert_eq!(m.getname(), "w");
        assert_eq!(m.getdtype(), dtype::f32);
        assert_eq!(m.getshape().getdims(), &[2, 3]);
        assert_eq!(m.getelementcount(), 6);
        assert_eq!(m.getbegin(), 4);
        assert_eq!(m.getend(), 28);
        assert_eq!(m.getnbytes(), 24);
    }

    #[test]
    fn unsupported_dtype_is_rejected() {
        let header = r#"{"w":{"dtype":"BOOL","shape":[1],"data_offsets":[0,1]}}"#;
        let bytes = artifact(header, &[1u8]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::unsupporteddtype { .. })
        ));
    }

    #[test]
    fn empty_shape_is_rejected() {
        let header = r#"{"w":{"dtype":"F32","shape":[],"data_offsets":[0,0]}}"#;
        let bytes = artifact(header, &[]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::emptyshape { .. })
        ));
    }

    #[test]
    fn element_count_overflow_is_rejected() {
        let header =
            r#"{"w":{"dtype":"I64","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#;
        let bytes = artifact(header, &[]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::shapeoverflow { .. })
        ));
    }

    #[test]
    fn element_budget_is_enforced() {
        let header = r#"{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let bytes = artifact(header, &[0u8; 24]);
        let lim = limits::new(4, usize::MAX, usize::MAX, usize::MAX);
        assert!(matches!(
            safetensorfile::parse(&bytes, &lim),
            Err(formaterror::elementlimitexceeded { .. })
        ));
    }

    #[test]
    fn offset_past_data_section_is_rejected() {
        let header = r#"{"w":{"dtype":"F32","shape":[17],"data_offsets":[0,68]}}"#;
        let bytes = artifact(header, &[0u8; 64]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::offsetsoutofrange { .. })
        ));
    }

    #[test]
    fn reversed_offsets_are_rejected() {
        let header = r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}"#;
        let bytes = artifact(header, &[0u8; 8]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::offsetsoutofrange { .. })
        ));
    }

    #[test]
    fn byte_length_mismatch_is_rejected() {
        let header = r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
        let bytes = artifact(header, &[0u8; 8]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::bytelengthmismatch { .. })
        ));
    }

    #[test]
    fn zero_element_tensor_is_accepted() {
        let header = r#"{"w":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
        let bytes = artifact(header, &[]);
        let file = safetensorfile::parse(&bytes, &permissive()).unwrap();
        let m = file.gettensor("w").unwrap();
        assert_eq!(m.getelementcount(), 0);
        assert_eq!(m.getnbytes(), 0);
    }
}

/// Tests relocated from `crates/format/src/safetensor.rs`.
mod safetensor {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::format::HEADERLENGTHSIZE;
    use ::format::formaterror;
    use ::format::safetensorfile;
    use lib::limits;
    use tensor::dtype;

    fn permissive() -> limits {
        limits::new(usize::MAX, usize::MAX, usize::MAX, usize::MAX)
    }

    /// Builds a safetensors buffer from a header JSON string and data bytes.
    fn build(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn parses_a_well_formed_artifact() {
        let header = r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let bytes = build(header, &[0u8; 16]);
        let file = safetensorfile::parse(&bytes, &permissive()).unwrap();
        assert_eq!(file.getheaderlength(), header.len());
        assert_eq!(file.getdatalength(), 16);
        assert_eq!(
            file.getmetadata().get("format").map(String::as_str),
            Some("pt")
        );
        let w = file.gettensor("w").unwrap();
        assert_eq!(w.getdtype(), dtype::f32);
        assert_eq!(w.getshape().getdims(), &[2, 2]);
        assert_eq!(file.readtensor(&bytes, "w").unwrap().len(), 16);
    }

    #[test]
    fn empty_data_section_is_accepted() {
        let header = r#"{"w":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
        let bytes = build(header, &[]);
        let file = safetensorfile::parse(&bytes, &permissive()).unwrap();
        assert_eq!(file.getdatalength(), 0);
    }

    #[test]
    fn truncated_length_prefix_is_rejected() {
        assert_eq!(
            safetensorfile::parse(&[0, 1, 2], &permissive()),
            Err(formaterror::truncated {
                needed: HEADERLENGTHSIZE,
                available: 3,
            })
        );
    }

    #[test]
    fn oversized_header_length_is_rejected() {
        let bytes = build("{}", &[0u8; 4]);
        let mut corrupted = bytes.clone();
        corrupted[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            safetensorfile::parse(&corrupted, &permissive()),
            Err(formaterror::headertoolarge { .. })
        ));
    }

    #[test]
    fn invalid_json_header_is_rejected() {
        let bytes = build("{not-json", &[]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::invalidjson { .. })
        ));
    }

    #[test]
    fn unsupported_dtype_is_rejected() {
        let header = r#"{"w":{"dtype":"BOOL","shape":[1],"data_offsets":[0,1]}}"#;
        let bytes = build(header, &[1u8]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::unsupporteddtype { .. })
        ));
    }

    #[test]
    fn offset_past_data_section_is_rejected() {
        let header = r#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let bytes = build(header, &[0u8; 8]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::offsetsoutofrange { .. })
        ));
    }

    #[test]
    fn byte_length_mismatch_is_rejected() {
        let header = r#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}"#;
        let bytes = build(header, &[0u8; 8]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::bytelengthmismatch { .. })
        ));
    }

    #[test]
    fn overlapping_tensors_are_rejected() {
        let header = r#"{"a":{"dtype":"F32","shape":[8],"data_offsets":[0,32]},"b":{"dtype":"F32","shape":[4],"data_offsets":[16,32]}}"#;
        let bytes = build(header, &[0u8; 32]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::overlappingtensors { .. })
        ));
    }

    #[test]
    fn malformed_metadata_is_rejected() {
        let header =
            r#"{"__metadata__":{"k":1},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let bytes = build(header, &[0u8; 4]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::invalidmetadata { .. })
        ));
    }

    #[test]
    fn readtensor_with_unknown_name_is_rejected() {
        let header = r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let bytes = build(header, &[0u8; 4]);
        let file = safetensorfile::parse(&bytes, &permissive()).unwrap();
        assert!(matches!(
            file.readtensor(&bytes, "missing"),
            Err(formaterror::invalidentry { .. })
        ));
    }

    #[test]
    fn readtensor_revalidates_against_supplied_buffer() {
        let header = r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let bytes = build(header, &[0u8; 8]);
        let file = safetensorfile::parse(&bytes, &permissive()).unwrap();
        // A shorter buffer no longer contains the validated range.
        let short = &bytes[..bytes.len() - 1];
        assert!(matches!(
            file.readtensor(short, "w"),
            Err(formaterror::offsetsoutofrange { .. })
        ));
    }

    #[test]
    fn element_budget_rejects_an_oversized_tensor() {
        let header = r#"{"w":{"dtype":"F32","shape":[16],"data_offsets":[0,64]}}"#;
        let bytes = build(header, &[0u8; 64]);
        let lim = limits::new(8, usize::MAX, usize::MAX, usize::MAX);
        assert!(matches!(
            safetensorfile::parse(&bytes, &lim),
            Err(formaterror::elementlimitexceeded { .. })
        ));
    }
}

/// Tests relocated from `crates/format/src/validation.rs`.
///
/// Entry-schema rejection and range-overlap detection are only reachable
/// through the public parser, so they are exercised via `safetensorfile::parse`
/// plus the public `checkoverlaps` over parsed metadata.
mod validation {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::format::{checkoverlaps, formaterror, safetensorfile};
    use ::lib::limits;

    fn permissive() -> limits {
        limits::new(usize::MAX, usize::MAX, usize::MAX, usize::MAX)
    }

    fn artifact(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn well_formed_entry_is_accepted() {
        let header = r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let bytes = artifact(header, &[0u8; 16]);
        let file = safetensorfile::parse(&bytes, &permissive()).unwrap();
        assert_eq!(file.gettensor("w").unwrap().getnbytes(), 16);
    }

    #[test]
    fn entry_missing_a_field_is_rejected() {
        let header = r#"{"w":{"dtype":"F32","shape":[2,2]}}"#;
        let bytes = artifact(header, &[0u8; 16]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::invalidentry { .. })
        ));
    }

    #[test]
    fn non_object_entry_is_rejected() {
        let header = r#"{"w":[1,2,3]}"#;
        let bytes = artifact(header, &[]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::invalidentry { .. })
        ));
    }

    #[test]
    fn disjoint_ranges_are_accepted() {
        let header = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"b":{"dtype":"F32","shape":[4],"data_offsets":[16,32]}}"#;
        let bytes = artifact(header, &[0u8; 32]);
        let file = safetensorfile::parse(&bytes, &permissive()).unwrap();
        assert_eq!(checkoverlaps(file.gettensors()), Ok(()));
    }

    #[test]
    fn overlapping_ranges_are_rejected() {
        let header = r#"{"a":{"dtype":"F32","shape":[8],"data_offsets":[0,32]},"b":{"dtype":"F32","shape":[4],"data_offsets":[16,32]}}"#;
        let bytes = artifact(header, &[0u8; 32]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::overlappingtensors { .. })
        ));
    }

    #[test]
    fn nested_range_is_detected_regardless_of_order() {
        // Outer range listed first; the sort inside the check must still catch
        // the enclosed inner range.
        let header = r#"{"outer":{"dtype":"F32","shape":[8],"data_offsets":[0,32]},"inner":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}}"#;
        let bytes = artifact(header, &[0u8; 32]);
        assert!(matches!(
            safetensorfile::parse(&bytes, &permissive()),
            Err(formaterror::overlappingtensors { .. })
        ));
    }

    #[test]
    fn zero_length_tensors_are_ignored() {
        let header = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"z":{"dtype":"F32","shape":[0],"data_offsets":[16,16]},"b":{"dtype":"F32","shape":[4],"data_offsets":[16,32]}}"#;
        let bytes = artifact(header, &[0u8; 32]);
        let file = safetensorfile::parse(&bytes, &permissive()).unwrap();
        assert_eq!(checkoverlaps(file.gettensors()), Ok(()));
    }
}
