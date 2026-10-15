//! End-to-end integration tests for the `cli` adapter.
//!
//! Each test drives the public `runcli` entry point with an `argv` vector and
//! in-memory writers, so the whole parse → dispatch → exit-code path is
//! exercised without spawning a process. Only the library surface is used.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use cli::{
    OPERATIONALEXITCODE, SUCCESSEXITCODE, UNSUPPORTEDEXITCODE, USAGEEXITCODE, VALIDATIONEXITCODE,
    runcli,
};
use tensor::{dtype, shape};

/// A valid `config.json` for a tiny Llama-family model.
const VALIDCONFIG: &str = r#"{
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
    "tie_word_embeddings": false
}"#;

/// A valid training-configuration document for `train`.
const VALIDTRAINCONFIG: &str = r#"{
    "accumulationsteps": 2,
    "maxsteps": 10,
    "seed": 7,
    "learningrate": 0.01,
    "momentum": 0.9,
    "weightdecay": 0.0,
    "optimizer": "adamw"
}"#;

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Returns a unique temporary path that no test shares.
fn temppath(tag: &str) -> PathBuf {
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!("cli-it-{}-{tag}-{unique}", std::process::id()));
    path
}

/// Writes `bytes` to a fresh temporary file and returns its path.
fn writeartifact(tag: &str, bytes: &[u8]) -> PathBuf {
    let path = temppath(tag);
    std::fs::write(&path, bytes).unwrap();
    path
}

/// Encodes a small valid checkpoint with one tensor to a temporary file.
fn writecheckpoint(tag: &str) -> PathBuf {
    let mut container = checkpoint::checkpoint::new(checkpoint::CHECKPOINTVERSION);
    let dims = shape::new(vec![2]).unwrap();
    container
        .addtensor("weight".to_string(), dtype::f32, dims, &[0_u8; 8])
        .unwrap();
    container.addmetadata("step".to_string(), "0".to_string());
    let bytes = container.encode().unwrap();
    writeartifact(tag, &bytes)
}

/// Runs `argv` through `runcli` and returns the exit code and both streams.
fn run(argv: &[&str]) -> (u8, String, String) {
    let mut out = Vec::new();
    let mut err = Vec::new();
    let code = runcli(argv.iter().copied(), &mut out, &mut err);
    (
        code,
        String::from_utf8_lossy(&out).into_owned(),
        String::from_utf8_lossy(&err).into_owned(),
    )
}

#[test]
fn exit_codes_are_distinct_and_stable() {
    assert_eq!(SUCCESSEXITCODE, 0);
    assert_eq!(OPERATIONALEXITCODE, 1);
    assert_eq!(USAGEEXITCODE, 2);
    assert_eq!(VALIDATIONEXITCODE, 3);
    assert_eq!(UNSUPPORTEDEXITCODE, 4);
}

#[test]
fn help_is_written_to_stdout_and_succeeds() {
    let (code, out, err) = run(&["cli", "--help"]);
    assert_eq!(code, SUCCESSEXITCODE);
    assert!(out.contains("Usage"), "stdout was {out:?}");
    assert!(err.is_empty(), "stderr was {err:?}");
}

#[test]
fn unknown_subcommands_are_usage_errors() {
    let (code, out, err) = run(&["cli", "frobnicate"]);
    assert_eq!(code, USAGEEXITCODE);
    assert!(out.is_empty(), "stdout was {out:?}");
    assert!(!err.is_empty(), "a usage diagnostic is expected");
}

#[test]
fn a_missing_required_argument_is_a_usage_error() {
    let (code, _out, err) = run(&["cli", "inspect"]);
    assert_eq!(code, USAGEEXITCODE);
    assert!(err.contains("--path"), "stderr was {err:?}");
}

#[test]
fn inspect_reports_config_metadata() {
    let path = writeartifact("inspect-config", VALIDCONFIG.as_bytes());
    let (code, out, err) = run(&["cli", "inspect", "--path", &path.to_string_lossy()]);
    assert_eq!(code, SUCCESSEXITCODE, "stderr: {err}");
    assert!(out.contains("artifact: modelconfig"), "stdout was {out:?}");
    assert!(out.contains("hidden_size: 64"), "stdout was {out:?}");
    assert!(out.contains("vocab_size: 256"), "stdout was {out:?}");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn inspect_of_a_missing_file_is_operational() {
    let path = temppath("inspect-missing");
    let (code, _out, err) = run(&["cli", "inspect", "--path", &path.to_string_lossy()]);
    assert_eq!(code, OPERATIONALEXITCODE);
    assert!(err.contains("i/o error"), "stderr was {err:?}");
}

#[test]
fn inspect_of_a_directory_is_operational() {
    let path = temppath("inspect-dir");
    std::fs::create_dir(&path).unwrap();
    let (code, _out, err) = run(&["cli", "inspect", "--path", &path.to_string_lossy()]);
    assert_eq!(code, OPERATIONALEXITCODE);
    assert!(err.contains("invalid path"), "stderr was {err:?}");
    std::fs::remove_dir(&path).unwrap();
}

#[test]
fn validate_accepts_a_valid_config() {
    let path = writeartifact("validate-config", VALIDCONFIG.as_bytes());
    let (code, out, err) = run(&["cli", "validate", "--path", &path.to_string_lossy()]);
    assert_eq!(code, SUCCESSEXITCODE, "stderr: {err}");
    assert!(out.contains("status: valid"), "stdout was {out:?}");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn validate_rejects_malformed_json_with_the_validation_code() {
    let path = writeartifact("validate-malformed", b"{not json");
    let (code, _out, err) = run(&["cli", "validate", "--path", &path.to_string_lossy()]);
    assert_eq!(code, VALIDATIONEXITCODE);
    assert!(!err.is_empty(), "a structured error is expected");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn validate_rejects_an_unsupported_model_type() {
    let json = VALIDCONFIG.replace("\"llama\"", "\"gpt2\"");
    let path = writeartifact("validate-gpt2", json.as_bytes());
    let (code, _out, err) = run(&["cli", "validate", "--path", &path.to_string_lossy()]);
    assert_eq!(code, VALIDATIONEXITCODE);
    assert!(!err.is_empty(), "a structured error is expected");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn inspect_reports_checkpoint_metadata() {
    let path = writecheckpoint("inspect-ckpt");
    let (code, out, err) = run(&["cli", "inspect", "--path", &path.to_string_lossy()]);
    assert_eq!(code, SUCCESSEXITCODE, "stderr: {err}");
    assert!(out.contains("artifact: checkpoint"), "stdout was {out:?}");
    assert!(out.contains("version: 1"), "stdout was {out:?}");
    assert!(out.contains("weight"), "stdout was {out:?}");
    assert!(out.contains("step = 0"), "stdout was {out:?}");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn validate_accepts_a_checkpoint() {
    let path = writecheckpoint("validate-ckpt");
    let (code, out, err) = run(&["cli", "validate", "--path", &path.to_string_lossy()]);
    assert_eq!(code, SUCCESSEXITCODE, "stderr: {err}");
    assert!(out.contains("artifact: checkpoint"), "stdout was {out:?}");
    assert!(out.contains("status: valid"), "stdout was {out:?}");
    assert!(out.contains("tensors: 1"), "stdout was {out:?}");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn generate_runs_the_real_engine_deterministically() {
    let first = run(&["cli", "generate", "--tokens", "1,2", "--maxnewtokens", "2"]);
    assert_eq!(first.0, SUCCESSEXITCODE, "stderr: {}", first.2);
    assert!(first.1.contains("model: llama"), "stdout was {:?}", first.1);
    assert!(first.1.contains("generated:"), "stdout was {:?}", first.1);
    assert!(first.1.contains("stopreason:"), "stdout was {:?}", first.1);

    let second = run(&["cli", "generate", "--tokens", "1,2", "--maxnewtokens", "2"]);
    assert_eq!(first.1, second.1, "the same seed must reproduce the output");
}

#[test]
fn generate_rejects_a_zero_token_cap() {
    let (code, _out, err) = run(&["cli", "generate", "--tokens", "1", "--maxnewtokens", "0"]);
    assert_eq!(code, USAGEEXITCODE);
    assert!(err.contains("maxnewtokens"), "stderr was {err:?}");
}

#[test]
fn generate_rejects_a_token_outside_the_vocabulary() {
    let (code, _out, err) = run(&["cli", "generate", "--tokens", "7", "--vocab", "4"]);
    assert_eq!(code, USAGEEXITCODE);
    assert!(err.contains("vocabulary"), "stderr was {err:?}");
}

#[test]
fn generate_text_requires_a_tokenizer() {
    let (code, _out, _err) = run(&["cli", "generate", "--text", "hello"]);
    assert_eq!(code, USAGEEXITCODE);
}

#[test]
fn train_check_only_validates_the_configuration() {
    let path = writeartifact("train-ok", VALIDTRAINCONFIG.as_bytes());
    let (code, out, err) = run(&[
        "cli",
        "train",
        "--config",
        &path.to_string_lossy(),
        "--check-only",
    ]);
    assert_eq!(code, SUCCESSEXITCODE, "stderr: {err}");
    assert!(out.contains("mode: check-only"), "stdout was {out:?}");
    assert!(out.contains("optimizer: adamw"), "stdout was {out:?}");
    assert!(out.contains("accumulationsteps: 2"), "stdout was {out:?}");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn train_without_check_only_reports_the_declared_limitation() {
    let path = writeartifact("train-limited", VALIDTRAINCONFIG.as_bytes());
    let (code, _out, err) = run(&["cli", "train", "--config", &path.to_string_lossy()]);
    assert_eq!(code, UNSUPPORTEDEXITCODE);
    assert!(err.contains("not supported"), "stderr was {err:?}");
    assert!(
        err.contains("end-to-end llama training"),
        "the diagnostic must name the operation; stderr was {err:?}"
    );
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn train_rejects_a_malformed_configuration_document() {
    let path = writeartifact("train-malformed", b"{}");
    let (code, _out, err) = run(&[
        "cli",
        "train",
        "--config",
        &path.to_string_lossy(),
        "--check-only",
    ]);
    assert_eq!(code, USAGEEXITCODE);
    assert!(
        err.contains("invalid training configuration"),
        "stderr was {err:?}"
    );
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn train_rejects_zero_accumulation_steps() {
    let json = VALIDTRAINCONFIG.replace("\"accumulationsteps\": 2", "\"accumulationsteps\": 0");
    let path = writeartifact("train-accum", json.as_bytes());
    let (code, _out, _err) = run(&[
        "cli",
        "train",
        "--config",
        &path.to_string_lossy(),
        "--check-only",
    ]);
    assert_eq!(code, OPERATIONALEXITCODE);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn decode_prints_text_for_a_token_list() {
    let tokenizer = r#"{"model":{"type":"BPE","vocab":{"hello":0," ":1,"world":2},"merges":[]}}"#;
    let path = writeartifact("tokenizer.json", tokenizer.as_bytes());
    let (code, out, err) = run(&[
        "cli",
        "decode",
        "--tokenizer",
        path.to_str().unwrap(),
        "--tokens",
        "0,1,2",
    ]);
    assert_eq!(code, SUCCESSEXITCODE);
    assert_eq!(out, "hello world\n");
    assert!(err.is_empty(), "stderr was {err:?}");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn decode_rejects_a_non_numeric_id() {
    let tokenizer = r#"{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#;
    let path = writeartifact("tokenizer.json", tokenizer.as_bytes());
    let (code, _out, err) = run(&[
        "cli",
        "decode",
        "--tokenizer",
        path.to_str().unwrap(),
        "--tokens",
        "a",
    ]);
    assert_eq!(code, USAGEEXITCODE);
    assert!(err.contains("non-numeric"), "stderr was {err:?}");
    std::fs::remove_file(&path).unwrap();
}
