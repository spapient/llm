use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

/// The top-level command-line parser.
///
/// A value is produced only by [`crate::parse`], which wraps
/// `clap::Parser::try_parse_from`, so the tree is testable without a process.
#[derive(Debug, Parser)]
#[command(name = "cli", about = "Command-line application for the LLM runtime")]
pub struct cli {
    /// The selected subcommand and its arguments.
    #[command(subcommand)]
    pub command: command,
}

/// The recognized subcommands.
#[derive(Debug, Subcommand)]
pub enum command {
    /// Inspect the metadata of a config, checkpoint, or safetensors artifact.
    inspect(inspectargs),

    /// Structurally validate a config, checkpoint, or safetensors artifact.
    validate(validateargs),

    /// Generate tokens with the inference engine over a deterministic tiny
    /// model.
    generate(generateargs),

    /// Decode a comma-separated list of token ids to text with a tokenizer.
    decode(decodeargs),

    /// Load and validate a training configuration.
    train(trainargs),
}

/// Arguments for `inspect`.
#[derive(Debug, Args)]
pub struct inspectargs {
    /// The artifact path to inspect.
    #[arg(long, value_name = "PATH")]
    pub path: PathBuf,
}

/// Arguments for `validate`.
#[derive(Debug, Args)]
pub struct validateargs {
    /// The artifact path to validate.
    #[arg(long, value_name = "PATH")]
    pub path: PathBuf,
}

/// Arguments for `generate`.
#[derive(Debug, Args)]
pub struct generateargs {
    /// A prompt as a comma-separated list of token ids.
    #[arg(
        long,
        value_name = "IDS",
        conflicts_with = "text",
        required_unless_present = "text"
    )]
    pub tokens: Option<String>,

    /// A prompt as text, tokenized by the tokenizer at `--tokenizer`.
    #[arg(
        long,
        value_name = "TEXT",
        conflicts_with = "tokens",
        required_unless_present = "tokens",
        requires = "tokenizer"
    )]
    pub text: Option<String>,

    /// A `tokenizer.json` path used to encode `--text`.
    #[arg(long, value_name = "PATH", requires = "text")]
    pub tokenizer: Option<PathBuf>,

    /// The maximum number of new tokens to generate.
    #[arg(long, default_value_t = crate::DEFAULTMAXNEWTOKENS)]
    pub maxnewtokens: usize,

    /// End-of-sequence token ids as a comma-separated list.
    #[arg(long, value_name = "IDS")]
    pub eos: Option<String>,

    /// The seed for the tiny model weights and the sampler.
    #[arg(long, default_value_t = 1)]
    pub seed: u64,

    /// The sampling temperature; `0.0` selects greedy decoding.
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// The top-k sampling bound, if any.
    #[arg(long)]
    pub topk: Option<usize>,

    /// The top-p sampling bound in `(0, 1]`, if any.
    #[arg(long)]
    pub topp: Option<f32>,

    /// The tiny model vocabulary size; the prompt ids must be below it.
    #[arg(long, default_value_t = crate::TINYVOCAB)]
    pub vocab: usize,
}

/// Arguments for `train`.
#[derive(Debug, Args)]
pub struct trainargs {
    /// The training-configuration JSON path to load and validate.
    #[arg(long, value_name = "PATH")]
    pub config: PathBuf,

    /// Validate the configuration and stop, instead of attempting a run.
    #[arg(long)]
    pub check_only: bool,
}

/// Arguments for `decode`.
#[derive(Debug, Args)]
pub struct decodeargs {
    /// A `tokenizer.json` path used to decode `--tokens`.
    #[arg(long, value_name = "PATH")]
    pub tokenizer: PathBuf,

    /// A comma-separated list of token ids to decode.
    #[arg(long, value_name = "IDS")]
    pub tokens: String,
}
