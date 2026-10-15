#![forbid(unsafe_code)]
//! The command-line application for the LLM runtime.
//!
//! `cli` is a Layer 4 application crate. It is a thin adapter around the
//! reusable workflows owned by the library crates: it parses arguments, selects
//! configuration, calls one real library workflow, and maps the outcome to
//! stdout/stderr and a process exit code. It owns no tensor math, no model
//! internals, no training loop, and no inference engine (crate-boundaries 5.16;
//! AGENTS.md §3).
//!
//! # Commands
//!
//! * `inspect` — print the metadata of a `config.json`, a checkpoint, or a
//!   safetensors artifact.
//! * `validate` — run the structural validation of one of those artifacts and
//!   report the structured error on failure.
//! * `generate` — drive the real [`infer`] engine over a deterministic tiny
//!   Llama built from a fixed configuration and an explicit seed, using the
//!   real [`cpu`] provider and optionally the real [`tokenizer`].
//! * `decode` — decode a comma-separated token-id list to text with a real
//!   [`tokenizer`] loaded from a `tokenizer.json`.
//! * `train` — load and validate a training configuration through
//!   [`train::trainconfig`] and [`optim`].
//!
//! # Declared limitation
//!
//! `nn`/`models` are forward-only and `autograd` has no Llama or
//! cross-entropy backward path, so an end-to-end `train` cannot fine-tune a
//! Llama. `train` validates the configuration and then returns
//! [`clierror::notsupported`]; `train --check-only` reports the validated plan
//! and succeeds. `cli` never fabricates a training success.
//!
//! # Exit behavior
//!
//! | Code | Meaning |
//! | ---- | ------- |
//! | 0 | success |
//! | 1 | operational failure (I/O, path, size, model/inference/training) |
//! | 2 | usage error (parse or invalid argument value) |
//! | 3 | structurally invalid artifact |
//! | 4 | declared limitation (`notsupported`) |
//!
//! The single mapping from [`clierror`] to a code is [`exitcode`].
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated. Module file names differ from the types they export
//! because a module and a re-exported type share one name namespace.

mod args;
mod commands;
mod error;

use std::ffi::OsString;
use std::io::Write;

use lib::limits;
use tracing::debug;

pub use args::{cli, command, decodeargs, generateargs, inspectargs, trainargs, validateargs};
pub use error::{
    OPERATIONALEXITCODE, SUCCESSEXITCODE, UNSUPPORTEDEXITCODE, USAGEEXITCODE, VALIDATIONEXITCODE,
    clierror, exitcode,
};

/// The default tiny model vocabulary size.
pub(crate) const TINYVOCAB: usize = 8;
/// The default tiny model dimension.
pub(crate) const TINYDIM: usize = 4;
/// The default tiny model layer count.
pub(crate) const TINYLAYERS: usize = 1;
/// The default tiny model query-head count.
pub(crate) const TINYHEADS: usize = 1;
/// The default tiny model key/value-head count.
pub(crate) const TINYKVHEADS: usize = 1;
/// The default tiny model feed-forward hidden size.
pub(crate) const TINYHIDDEN: usize = 8;
/// The default tiny model maximum sequence length.
pub(crate) const TINYMAXSEQ: usize = 16;
/// The default tiny model RMS-normalization epsilon.
pub(crate) const TINYRMSEPS: f32 = 1.0e-5;
/// The default tiny model RoPE base period.
pub(crate) const TINYROPEBASE: f32 = 10_000.0;

/// The default new-token cap for `generate`.
pub(crate) const DEFAULTMAXNEWTOKENS: usize = 4;
/// The inclusive upper bound on the requested new-token cap.
///
/// Basis: a bounded decode budget keeps the execution step budget and the
/// prompt-plus-cap context check finite; a larger cap is a batch workload, not
/// an interactive CLI request.
pub(crate) const MAXNEWTOKENSLIMIT: usize = 1024;
/// The inclusive upper bound on parsed prompt tokens, as a parse guard.
///
/// Basis: the model's maximum sequence length is the real bound; this only
/// stops an absurd input from being expanded before the engine rejects it.
pub(crate) const MAXPROMPTTOKENS: usize = 4096;

/// The execution memory budget for one generation request, in bytes.
///
/// Basis: the tiny model and its activations are a few kilobytes; one megabyte
/// leaves room for the logits buffer and fails fast rather than growing.
pub(crate) const DEFAULTMEMORYBYTES: usize = 1 << 20;
/// The execution worker count for one generation request.
pub(crate) const DEFAULTMAXWORKERS: usize = 1;
/// The inclusive upper bound on a file read for inspection or validation.
///
/// Basis: an artifact larger than one gibibyte is not an interactive
/// inspect/validate target and is rejected before it is read into memory.
pub(crate) const DEFAULTMAXARTIFACTBYTES: usize = 1 << 30;
/// The element, byte, sequence, and batch budget for artifact parsing and CPU
/// execution.
///
/// Basis: per-tensor element and byte bounds sized for the tiny model, with the
/// same explicit limits applied to both the parser and the CPU provider so no
/// undocumented default is invented.
pub(crate) const DEFAULTLIMITS: limits = limits::new(1 << 20, 1 << 24, 1 << 16, 1 << 12);

/// The AdamW first-moment decay used for `train --optimizer adamw`.
pub(crate) const ADAMWBETA1: f32 = 0.9;
/// The AdamW second-moment decay used for `train --optimizer adamw`.
pub(crate) const ADAMWBETA2: f32 = 0.999;
/// The AdamW denominator offset used for `train --optimizer adamw`.
pub(crate) const ADAMWEPS: f32 = 1.0e-8;

/// The rendered output of one successfully dispatched command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct commandoutput {
    text: String,
}

impl commandoutput {
    /// Builds an output value from rendered text.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }

    /// Returns the rendered text.
    #[must_use]
    pub fn gettext(&self) -> &str {
        &self.text
    }
}

impl core::fmt::Display for commandoutput {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(&self.text)
    }
}

/// Parses `args` into a [`cli`] command tree.
///
/// This wraps `clap::Parser::try_parse_from` so the whole tree can be parsed
/// and dispatched in-process. The first element of `args` is the program name.
///
/// # Errors
///
/// Returns the `clap` diagnostic for an unknown flag, a missing required
/// argument, or a conflicting argument. `--help` and `--version` are returned
/// as a `clap` "display" error, which [`runcli`] writes to stdout and treats as
/// success.
pub fn parse<I, T>(args: I) -> Result<cli, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    <cli as clap::Parser>::try_parse_from(args)
}

/// Dispatches a parsed command to the workflow it selects.
///
/// # Errors
///
/// Returns a [`clierror`] describing a usage, path, artifact, model, inference,
/// training, tokenization, or optimizer failure, or the declared
/// [`clierror::notsupported`] limitation.
pub fn run(parsed: cli) -> Result<commandoutput, clierror> {
    debug!("dispatching command");
    commands::dispatch(parsed.command)
}

/// Installs the process-wide diagnostics subscriber if none is present.
///
/// The filter is read from the `RUST_LOG` environment variable and defaults to
/// `warn`, so the `debug`/`trace` events emitted on hot paths stay discarded
/// unless an operator opts in (`PF2`). A pre-existing global subscriber is kept
/// and the install is skipped explicitly rather than treated as an error.
fn initdiagnostics() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let subscriber = tracing_subscriber::fmt().with_env_filter(filter).finish();
    match tracing::subscriber::set_global_default(subscriber) {
        Ok(()) => {}
        Err(_) => {
            // Diagnostics were already enabled by another component; keep them.
        }
    }
}

/// Runs the full `argv`-to-exit-code adapter, writing output to `out` and
/// diagnostics to `err`.
///
/// This is the composition root the binary calls. It parses, dispatches,
/// writes the report or the diagnostic, and returns the process exit code. It
/// is generic over the writers so tests can capture output in memory.
///
/// # Returns
///
/// [`SUCCESSEXITCODE`] on success (including `--help`/`--version`), or the code
/// from [`exitcode`] for a [`clierror`], or [`USAGEEXITCODE`] for a parse
/// error.
pub fn runcli<A, W, E>(args: A, out: &mut W, err: &mut E) -> u8
where
    A: IntoIterator,
    A::Item: Into<OsString> + Clone,
    W: Write,
    E: Write,
{
    initdiagnostics();
    match parse(args) {
        Ok(parsed) => match run(parsed) {
            Ok(output) => {
                if writeline(out, output.gettext()).is_ok() {
                    SUCCESSEXITCODE
                } else {
                    // Best-effort diagnostic: if stdout is gone there may be no
                    // usable channel to report through, but the code still
                    // signals the failure.
                    let _ = writeline(err, "error: failed to write command output");
                    OPERATIONALEXITCODE
                }
            }
            Err(error) => {
                let text = format!("error: {error}");
                // The diagnostic is best-effort: the exit code below still
                // conveys the failure if stderr cannot be written.
                let _ = writeline(err, &text);
                exitcode(&error)
            }
        },
        Err(error) => {
            let message = error.to_string();
            let text = message.trim_end();
            if error.use_stderr() {
                // A parse error; the code is the failure signal even if the
                // diagnostic cannot be written.
                let _ = writeline(err, text);
                USAGEEXITCODE
            } else {
                // `--help`/`--version`: a request for information, not an error.
                let _ = writeline(out, text);
                SUCCESSEXITCODE
            }
        }
    }
}

/// Writes `text` and a trailing newline to `target`, then flushes.
fn writeline<W: Write>(target: &mut W, text: &str) -> std::io::Result<()> {
    target.write_all(text.as_bytes())?;
    if !text.ends_with('\n') {
        target.write_all(b"\n")?;
    }
    target.flush()
}
