//! Relocated unit tests for the `cli` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/cli/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/decision-records/blackboxtesting.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/cli/src/args.rs`.
mod args {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::cli::command;

    #[test]
    fn parses_inspect_and_validate_paths() {
        let parsed = ::cli::parse(["cli", "inspect", "--path", "model.bin"]).unwrap();
        assert!(
            matches!(parsed.command, command::inspect(args) if args.path.ends_with("model.bin"))
        );
        let parsed = ::cli::parse(["cli", "validate", "--path", "model.bin"]).unwrap();
        assert!(
            matches!(parsed.command, command::validate(args) if args.path.ends_with("model.bin"))
        );
    }

    #[test]
    fn generate_requires_exactly_one_prompt_source() {
        assert!(::cli::parse(["cli", "generate"]).is_err());
        assert!(::cli::parse(["cli", "generate", "--tokens", "1,2"]).is_ok());
        assert!(
            ::cli::parse(["cli", "generate", "--tokens", "1", "--text", "a"]).is_err(),
            "both prompt sources must be rejected"
        );
        assert!(
            ::cli::parse(["cli", "generate", "--text", "a"]).is_err(),
            "text requires a tokenizer"
        );
    }

    #[test]
    fn generate_defaults_are_applied() {
        let parsed = ::cli::parse(["cli", "generate", "--tokens", "1"]).unwrap();
        let command::generate(args) = parsed.command else {
            unreachable!("the subcommand is generate");
        };
        assert_eq!(args.maxnewtokens, 4);
        assert_eq!(args.vocab, 8);
        assert_eq!(args.seed, 1);
        assert_eq!(args.temperature, 0.0);
        assert!(args.topk.is_none());
        assert!(args.topp.is_none());
    }

    #[test]
    fn train_parses_the_check_only_flag() {
        let parsed =
            ::cli::parse(["cli", "train", "--config", "run.json", "--check-only"]).unwrap();
        let command::train(args) = parsed.command else {
            unreachable!("the subcommand is train");
        };
        assert!(args.check_only);
        assert!(args.config.ends_with("run.json"));
    }

    #[test]
    fn unknown_subcommands_are_rejected() {
        assert!(::cli::parse(["cli", "frobnicate"]).is_err());
    }
}

/// Tests relocated from `crates/cli/src/commands/mod.rs`.
///
/// Classification and reading are exercised through the public `run` seam. The
/// missing-path and directory rejections are covered end to end by
/// `tests/cli.rs`; the size cap is one gibibyte and has no practical trigger.
mod commands {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use ::cli::{parse, run};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temppath(tag: &str) -> PathBuf {
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!("cli-unit-{}-{tag}-{unique}", std::process::id()));
        path
    }

    fn artifact(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn inspect_classifies_and_reads_a_safetensors_artifact() {
        let path = temppath("safe");
        let header = r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        std::fs::write(&path, artifact(header, &[0u8; 4])).unwrap();
        let p = path.to_string_lossy().into_owned();
        let parsed = parse(["cli", "inspect", "--path", p.as_str()]).unwrap();
        let out = run(parsed).unwrap();
        assert!(
            out.gettext().contains("artifact: safetensors"),
            "stdout was {}",
            out.gettext()
        );
        std::fs::remove_file(&path).unwrap();
    }
}

/// Tests relocated from `crates/cli/src/error.rs`.
mod error {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::path::PathBuf;

    use ::cli::OPERATIONALEXITCODE;
    use ::cli::UNSUPPORTEDEXITCODE;
    use ::cli::USAGEEXITCODE;
    use ::cli::VALIDATIONEXITCODE;
    use ::cli::clierror;
    use ::cli::exitcode;

    #[test]
    fn usage_and_config_errors_map_to_usage() {
        let argument = clierror::invalidarguments {
            detail: "prompt must not be empty".to_string(),
        };
        let config = clierror::invalidconfigjson {
            line: 1,
            column: 2,
            message: "missing field".to_string(),
        };
        assert_eq!(exitcode(&argument), USAGEEXITCODE);
        assert_eq!(exitcode(&config), USAGEEXITCODE);
    }

    #[test]
    fn artifact_failures_map_to_validation() {
        let bad = clierror::format(format::formaterror::truncated {
            needed: 8,
            available: 3,
        });
        let checkpoint = clierror::checkpoint(checkpoint::checkpointerror::badmagic);
        assert_eq!(exitcode(&bad), VALIDATIONEXITCODE);
        assert_eq!(exitcode(&checkpoint), VALIDATIONEXITCODE);
    }

    #[test]
    fn declared_limitations_map_to_unsupported() {
        let error = clierror::notsupported {
            operation: "end-to-end llama training",
            detail: "no backward path",
        };
        assert_eq!(exitcode(&error), UNSUPPORTEDEXITCODE);
    }

    #[test]
    fn path_and_io_failures_map_to_operational() {
        let io = clierror::io {
            path: PathBuf::from("missing.bin"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
        };
        let path = clierror::invalidpath {
            path: PathBuf::from("."),
            detail: "not a regular file",
        };
        let size = clierror::sizeexceeded {
            path: PathBuf::from("big.bin"),
            observed: 10,
            limit: 8,
        };
        let unknown = clierror::unrecognizedartifact {
            path: PathBuf::from("x.bin"),
        };
        assert_eq!(exitcode(&io), OPERATIONALEXITCODE);
        assert_eq!(exitcode(&path), OPERATIONALEXITCODE);
        assert_eq!(exitcode(&size), OPERATIONALEXITCODE);
        assert_eq!(exitcode(&unknown), OPERATIONALEXITCODE);
    }

    #[test]
    fn carried_workflow_failures_map_to_operational() {
        let error = clierror::infer(infer::infererror::emptylogits);
        assert_eq!(exitcode(&error), OPERATIONALEXITCODE);
    }

    #[test]
    fn display_carries_structured_context() {
        let error = clierror::invalidpath {
            path: PathBuf::from("model.bin"),
            detail: "not a regular file",
        };
        assert_eq!(
            error.to_string(),
            "invalid path model.bin: not a regular file"
        );
    }
}

/// Tests relocated from `crates/cli/src/lib.rs`.
mod root {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::cli::SUCCESSEXITCODE;
    use ::cli::USAGEEXITCODE;
    use ::cli::runcli;

    #[test]
    fn help_is_written_to_stdout_and_succeeds() {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let code = runcli(["cli", "--help"], &mut out, &mut err);
        assert_eq!(code, SUCCESSEXITCODE);
        assert!(String::from_utf8_lossy(&out).contains("Usage"));
        assert!(err.is_empty());
    }

    #[test]
    fn a_parse_error_is_written_to_stderr_with_the_usage_code() {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let code = runcli(["cli", "generate"], &mut out, &mut err);
        assert_eq!(code, USAGEEXITCODE);
        assert!(out.is_empty());
        assert!(!err.is_empty());
    }

    #[test]
    fn the_output_display_matches_its_text() {
        let output = ::cli::commandoutput::new("hello\n");
        assert_eq!(output.gettext(), "hello\n");
        assert_eq!(output.to_string(), "hello\n");
    }
}
