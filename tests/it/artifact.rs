#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests exercising the public `format` API end to end.

use format::{formaterror, modelconfig, safetensorfile};
use lib::limits;
use tensor::dtype;

fn permissive() -> limits {
    limits::new(usize::MAX, usize::MAX, usize::MAX, usize::MAX)
}

fn build(header: &str, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

#[test]
fn parses_a_two_tensor_artifact_and_reads_values() {
    let header = r#"{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b":{"dtype":"I64","shape":[2],"data_offsets":[16,32]}}"#;
    let mut data = vec![0u8; 32];
    data[0..4].copy_from_slice(&1.0f32.to_le_bytes());
    data[16..24].copy_from_slice(&7i64.to_le_bytes());
    let bytes = build(header, &data);

    let file = safetensorfile::parse(&bytes, &permissive()).unwrap();
    assert_eq!(file.gettensors().len(), 2);

    let a = file.gettensor("a").unwrap();
    assert_eq!(a.getdtype(), dtype::f32);
    assert_eq!(a.getshape().getdims(), &[2, 2]);
    assert_eq!(a.getelementcount(), 4);
    assert_eq!(file.readtensor(&bytes, "a").unwrap().len(), 16);

    let b = file.gettensor("b").unwrap();
    assert_eq!(b.getdtype(), dtype::i64);
    let raw = file.readtensor(&bytes, "b").unwrap();
    assert_eq!(i64::from_le_bytes(raw[0..8].try_into().unwrap()), 7);
}

#[test]
fn truncated_artifact_reports_a_typed_error() {
    let header = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
    let bytes = build(header, &[0u8; 4]);
    assert!(matches!(
        safetensorfile::parse(&bytes, &permissive()),
        Err(formaterror::offsetsoutofrange { .. })
    ));
}

#[test]
fn unsupported_dtype_is_reported_explicitly() {
    let header = r#"{"a":{"dtype":"U16","shape":[4],"data_offsets":[0,8]}}"#;
    let bytes = build(header, &[0u8; 8]);
    assert!(matches!(
        safetensorfile::parse(&bytes, &permissive()),
        Err(formaterror::unsupporteddtype { .. })
    ));
}

#[test]
fn half_precision_is_accepted_as_storage_only() {
    let header = r#"{"a":{"dtype":"F16","shape":[4],"data_offsets":[0,8]},"b":{"dtype":"BF16","shape":[4],"data_offsets":[8,16]}}"#;
    let bytes = build(header, &[0u8; 16]);
    let file = safetensorfile::parse(&bytes, &permissive()).unwrap();
    assert_eq!(file.gettensor("a").unwrap().getdtype(), dtype::f16);
    assert_eq!(file.gettensor("b").unwrap().getdtype(), dtype::bf16);
}

#[test]
fn parses_a_model_config_and_rejects_a_bad_one() {
    let good = br#"{
        "model_type": "llama",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "vocab_size": 128,
        "rope_theta": 10000.0,
        "torch_dtype": "bfloat16"
    }"#;
    let config = modelconfig::parse(good).unwrap();
    assert_eq!(config.gethiddensize(), 32);
    assert_eq!(config.getheadsize(), 8);
    assert_eq!(config.getkeyvaluegroups(), 2);
    assert_eq!(config.gettorchdtype(), Some(dtype::bf16));

    let bad = br#"{
        "model_type": "mistral",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "vocab_size": 128,
        "rope_theta": 10000.0
    }"#;
    assert!(matches!(
        modelconfig::parse(bad),
        Err(formaterror::unsupportedmodeltype { .. })
    ));
}
