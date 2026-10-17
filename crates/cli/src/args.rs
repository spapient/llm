//! Command-line argument parsing for the `cli` crate.
//!
//! A hand-written parser replaces the previous `clap` derive tree so the crate
//! carries no third-party CLI dependency. The public value types are
//! unchanged; only the production of a [`cli`] from `argv` is rewritten.

use std::ffi::OsString;
use std::path::PathBuf;

/// The top-level command-line value: a parsed subcommand and its arguments.
///
/// A value is produced only by [`crate::parse`], so the tree is testable
/// without a process.
#[derive(Debug, PartialEq)]
pub struct cli {
    /// The selected subcommand and its arguments.
    pub command: command,
}

/// The recognized subcommands.
#[derive(Debug, PartialEq)]
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
#[derive(Debug, PartialEq)]
pub struct inspectargs {
    /// The artifact path to inspect.
    pub path: PathBuf,
}

/// Arguments for `validate`.
#[derive(Debug, PartialEq)]
pub struct validateargs {
    /// The artifact path to validate.
    pub path: PathBuf,
}

/// Arguments for `generate`.
#[derive(Debug, PartialEq)]
pub struct generateargs {
    /// A prompt as a comma-separated list of token ids.
    pub tokens: Option<String>,
    /// A prompt as text, tokenized by the tokenizer at `--tokenizer`.
    pub text: Option<String>,
    /// A `tokenizer.json` path used to encode `--text`.
    pub tokenizer: Option<PathBuf>,
    /// The maximum number of new tokens to generate.
    pub maxnewtokens: usize,
    /// End-of-sequence token ids as a comma-separated list.
    pub eos: Option<String>,
    /// The seed for the tiny model weights and the sampler.
    pub seed: u64,
    /// The sampling temperature; `0.0` selects greedy decoding.
    pub temperature: f32,
    /// The top-k sampling bound, if any.
    pub topk: Option<usize>,
    /// The top-p sampling bound in `(0, 1]`, if any.
    pub topp: Option<f32>,
    /// The tiny model vocabulary size; the prompt ids must be below it.
    pub vocab: usize,
}

/// Arguments for `train`.
#[derive(Debug, PartialEq)]
pub struct trainargs {
    /// The training-configuration JSON path to load and validate.
    pub config: PathBuf,
    /// Validate the configuration and stop, instead of attempting a run.
    pub check_only: bool,
}

/// Arguments for `decode`.
#[derive(Debug, PartialEq)]
pub struct decodeargs {
    /// A `tokenizer.json` path used to decode `--tokens`.
    pub tokenizer: PathBuf,
    /// A comma-separated list of token ids to decode.
    pub tokens: String,
}

/// A parse outcome that is not a value: a usage diagnostic or an information
/// request (`--help` / `--version`).
#[derive(Debug, Clone, PartialEq)]
pub enum parseoutcome {
    /// A usage diagnostic; written to stderr and mapped to `USAGEEXITCODE`.
    usage(String),
    /// The rendered help text; written to stdout and mapped to success.
    help(String),
    /// The rendered version text; written to stdout and mapped to success.
    version(String),
}

impl parseoutcome {
    /// Whether the message should be written to stderr (`true`) or stdout.
    #[must_use]
    pub fn use_stderr(&self) -> bool {
        matches!(self, parseoutcome::usage(_))
    }
}

impl core::fmt::Display for parseoutcome {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            parseoutcome::usage(message)
            | parseoutcome::help(message)
            | parseoutcome::version(message) => formatter.write_str(message),
        }
    }
}

/// Parses the argument vector (including the program name at index 0) into a
/// [`cli`] value.
///
/// The first element of `args` is the program name and is skipped; the
/// remaining elements select a subcommand and its flags.
///
/// # Errors
///
/// Returns a [`parseoutcome`] for a missing or unknown subcommand, an unknown
/// flag, a missing or mistyped flag value, a conflicting or missing required
/// argument, or an information request (`--help` / `--version`).
pub fn parse<I, T>(args: I) -> Result<cli, parseoutcome>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let raw: Vec<String> = args
        .into_iter()
        .map(|item| item.into().to_string_lossy().into_owned())
        .collect();
    if raw.is_empty() {
        return Err(parseoutcome::help(HELPTEXT.to_string()));
    }
    parseargs(&raw[1..])
}

/// Dispatches the first positional token to the matching subcommand parser.
fn parseargs(args: &[String]) -> Result<cli, parseoutcome> {
    let Some(head) = args.first() else {
        return Err(parseoutcome::help(HELPTEXT.to_string()));
    };
    match head.as_str() {
        "--help" | "-h" => Err(parseoutcome::help(HELPTEXT.to_string())),
        "--version" | "-V" => Err(parseoutcome::version(VERSIONTEXT.to_string())),
        "inspect" => Ok(cli {
            command: command::inspect(parseinspect(&args[1..])?),
        }),
        "validate" => Ok(cli {
            command: command::validate(parsevalidate(&args[1..])?),
        }),
        "generate" => Ok(cli {
            command: command::generate(parsegenerate(&args[1..])?),
        }),
        "decode" => Ok(cli {
            command: command::decode(parsedecode(&args[1..])?),
        }),
        "train" => Ok(cli {
            command: command::train(parsetrain(&args[1..])?),
        }),
        other => Err(parseoutcome::usage(format!(
            "error: unrecognized subcommand '{other}'"
        ))),
    }
}

/// The shared flag cursor over the remaining argument slice.
struct flags<'a> {
    args: &'a [String],
    index: usize,
}

impl<'a> flags<'a> {
    fn new(args: &'a [String]) -> Self {
        Self { args, index: 0 }
    }

    /// Returns the next token, if any.
    fn next(&mut self) -> Option<&'a str> {
        let token = self.args.get(self.index).map(String::as_str);
        if token.is_some() {
            self.index += 1;
        }
        token
    }

    /// Returns the next token, if any, without consuming it.
    fn peek(&self) -> Option<&'a str> {
        self.args.get(self.index).map(String::as_str)
    }

    /// Consumes `--name VALUE` or `--name=VALUE` and returns the value.
    ///
    /// # Errors
    ///
    /// Returns a usage diagnostic when the flag is absent, the inline form is
    /// empty, or the spaced value is missing.
    fn value(&mut self, name: &str) -> Result<String, parseoutcome> {
        let Some(token) = self.next() else {
            return Err(parseoutcome::usage(format!(
                "error: the following required argument was not provided: {name}"
            )));
        };
        if let Some(inline) = token.strip_prefix(&format!("{name}=")) {
            if inline.is_empty() {
                return Err(parseoutcome::usage(format!(
                    "error: a value is required for {name}"
                )));
            }
            return Ok(inline.to_string());
        }
        if token == name {
            let Some(value) = self.next() else {
                return Err(parseoutcome::usage(format!(
                    "error: a value is required for {name}"
                )));
            };
            return Ok(value.to_string());
        }
        Err(parseoutcome::usage(format!(
            "error: unexpected argument '{token}'"
        )))
    }

    /// Consumes `--name` as a boolean flag; `false` when the next token is not
    /// the flag.
    fn boolean(&mut self, name: &str) -> bool {
        if self.args.get(self.index).is_some_and(|token| token == name) {
            self.index += 1;
            true
        } else {
            false
        }
    }

    /// Rejects any trailing token.
    ///
    /// # Errors
    ///
    /// Returns a usage diagnostic for a trailing flag or value.
    fn finish(self) -> Result<(), parseoutcome> {
        if let Some(token) = self.args.get(self.index) {
            return Err(parseoutcome::usage(format!(
                "error: unexpected argument '{token}'"
            )));
        }
        Ok(())
    }
}

/// Parses a numeric value with the named flag's diagnostics.
fn number<T: core::str::FromStr>(raw: &str, name: &str) -> Result<T, parseoutcome> {
    raw.parse::<T>().map_err(|_| {
        parseoutcome::usage(format!(
            "error: invalid value '{raw}' for {name}: not a valid number"
        ))
    })
}

/// Parses `inspect` arguments: a required `--path`.
fn parseinspect(args: &[String]) -> Result<inspectargs, parseoutcome> {
    let mut cursor = flags::new(args);
    let path = PathBuf::from(cursor.value("--path")?);
    cursor.finish()?;
    Ok(inspectargs { path })
}

/// Parses `validate` arguments: a required `--path`.
fn parsevalidate(args: &[String]) -> Result<validateargs, parseoutcome> {
    let mut cursor = flags::new(args);
    let path = PathBuf::from(cursor.value("--path")?);
    cursor.finish()?;
    Ok(validateargs { path })
}

/// Parses `generate` arguments.
fn parsegenerate(args: &[String]) -> Result<generateargs, parseoutcome> {
    let mut tokens: Option<String> = None;
    let mut text: Option<String> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut eos: Option<String> = None;
    let mut topk: Option<usize> = None;
    let mut topp: Option<f32> = None;
    let mut maxnewtokens = crate::DEFAULTMAXNEWTOKENS;
    let mut seed = 1u64;
    let mut temperature = 0.0f32;
    let mut vocab = crate::TINYVOCAB;
    let mut cursor = flags::new(args);
    while let Some(token) = cursor.peek() {
        match token {
            "--tokens" => tokens = Some(cursor.value("--tokens")?),
            "--text" => text = Some(cursor.value("--text")?),
            "--tokenizer" => tokenizer = Some(PathBuf::from(cursor.value("--tokenizer")?)),
            "--eos" => eos = Some(cursor.value("--eos")?),
            "--topk" => topk = Some(number(&cursor.value("--topk")?, "--topk")?),
            "--topp" => topp = Some(number(&cursor.value("--topp")?, "--topp")?),
            "--maxnewtokens" => {
                maxnewtokens = number(&cursor.value("--maxnewtokens")?, "--maxnewtokens")?
            }
            "--seed" => seed = number(&cursor.value("--seed")?, "--seed")?,
            "--temperature" => {
                temperature = number(&cursor.value("--temperature")?, "--temperature")?
            }
            "--vocab" => vocab = number(&cursor.value("--vocab")?, "--vocab")?,
            other => {
                return Err(parseoutcome::usage(format!(
                    "error: unexpected argument '{other}'"
                )));
            }
        }
    }
    match (&tokens, &text) {
        (Some(_), Some(_)) => {
            return Err(parseoutcome::usage(
                "error: the argument '--tokens' cannot be used with '--text'".to_string(),
            ));
        }
        (None, None) => {
            return Err(parseoutcome::usage(
                "error: one of '--tokens' or '--text' is required".to_string(),
            ));
        }
        _ => {}
    }
    if text.is_some() && tokenizer.is_none() {
        return Err(parseoutcome::usage(
            "error: the argument '--text' requires '--tokenizer'".to_string(),
        ));
    }
    if tokenizer.is_some() && text.is_none() {
        return Err(parseoutcome::usage(
            "error: the argument '--tokenizer' requires '--text'".to_string(),
        ));
    }
    Ok(generateargs {
        tokens,
        text,
        tokenizer,
        maxnewtokens,
        eos,
        seed,
        temperature,
        topk,
        topp,
        vocab,
    })
}

/// Parses `train` arguments: a required `--config` and an optional
/// `--check-only`.
fn parsetrain(args: &[String]) -> Result<trainargs, parseoutcome> {
    let mut cursor = flags::new(args);
    let config = PathBuf::from(cursor.value("--config")?);
    let check_only = cursor.boolean("--check-only");
    cursor.finish()?;
    Ok(trainargs { config, check_only })
}

/// Parses `decode` arguments: required `--tokenizer` and `--tokens`.
fn parsedecode(args: &[String]) -> Result<decodeargs, parseoutcome> {
    let mut cursor = flags::new(args);
    let tokenizer = PathBuf::from(cursor.value("--tokenizer")?);
    let tokens = cursor.value("--tokens")?;
    cursor.finish()?;
    Ok(decodeargs { tokenizer, tokens })
}

/// The rendered help text; the first line matches the previous `clap` "Usage"
/// banner the tests assert on.
const HELPTEXT: &str = "Usage: cli <COMMAND>

Commands:
  inspect   Inspect the metadata of a config, checkpoint, or safetensors artifact
  validate  Structurally validate a config, checkpoint, or safetensors artifact
  generate  Generate tokens with the inference engine over a deterministic tiny model
  decode    Decode a comma-separated list of token ids to text with a tokenizer
  train     Load and validate a training configuration

Options:
  -h, --help     Print help
  -V, --version  Print version";

/// The rendered version text.
const VERSIONTEXT: &str = "cli 0.1.0";
