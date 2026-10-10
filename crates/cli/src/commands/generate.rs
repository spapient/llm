use std::path::Path;

use context::{executionconfig, executioncontext};
use cpu::cpuexecutor;
use infer::{
    collectingsink, generationengine, generationparams, generationrequest, generationresult,
    samplingparams, stopcondition, stopreason,
};
use lib::{requestid, seed};
use models::{getarchitecture, llamaconfig, llamaweights};
use tokenizer::tokenizer;

use crate::args::generateargs;
use crate::clierror;
use crate::commandoutput;
use crate::commands::readartifact;
use crate::{
    DEFAULTLIMITS, DEFAULTMAXARTIFACTBYTES, DEFAULTMAXWORKERS, DEFAULTMEMORYBYTES,
    MAXNEWTOKENSLIMIT, MAXPROMPTTOKENS, TINYDIM, TINYHEADS, TINYHIDDEN, TINYKVHEADS, TINYLAYERS,
    TINYMAXSEQ, TINYRMSEPS, TINYROPEBASE,
};

/// Generates tokens with the real inference engine over a deterministic tiny
/// model.
///
/// # Errors
///
/// Returns a [`clierror`] for an invalid argument value, a path or read
/// failure, a tokenizer failure, a model construction failure, or a generation
/// failure from the engine.
pub(crate) fn run(args: &generateargs) -> Result<commandoutput, clierror> {
    validatecap(args.maxnewtokens)?;
    let config = configfromvocab(args.vocab)?;
    let prompt = promptfromargs(args, config.getvocab())?;
    let contextlen =
        prompt
            .len()
            .checked_add(args.maxnewtokens)
            .ok_or_else(|| clierror::invalidarguments {
                detail: "prompt length plus maxnewtokens overflows".to_string(),
            })?;
    if contextlen > config.getmaxseq() {
        return Err(clierror::invalidarguments {
            detail: format!(
                "prompt length {} plus maxnewtokens {} exceeds the model maxseq {}",
                prompt.len(),
                args.maxnewtokens,
                config.getmaxseq()
            ),
        });
    }
    let eos = parseeos(args.eos.as_deref(), config.getvocab())?;
    let sampling =
        samplingparams::new(args.temperature, args.topk, args.topp, seed::new(args.seed))?;
    let stop = stopcondition::new(eos, args.maxnewtokens)?;
    let params = generationparams::formodel(&config, stop, sampling);
    let mut request = generationrequest::new(prompt, params)?;

    let weights = llamaweights::random(&config, seed::new(args.seed))?;
    let model = getarchitecture("llama")?.newmodel(config, weights)?;

    let maxsteps = args.maxnewtokens.saturating_add(2);
    let exec = executioncontext::new(
        requestid::new(1),
        executionconfig::new(DEFAULTMEMORYBYTES, DEFAULTMAXWORKERS, maxsteps)?,
    );
    let executor = cpuexecutor::new(DEFAULTLIMITS);
    let engine = generationengine::new(&model, &executor);
    let mut sink = collectingsink::new();
    let result = engine.generate(&exec, &mut request, &mut sink)?;

    Ok(commandoutput::new(formatresult(
        args,
        &config,
        request.getprompt(),
        &result,
    )))
}

/// Rejects a zero or oversized new-token cap.
fn validatecap(maxnewtokens: usize) -> Result<(), clierror> {
    if maxnewtokens == 0 {
        return Err(clierror::invalidarguments {
            detail: "maxnewtokens must be at least 1".to_string(),
        });
    }
    if maxnewtokens > MAXNEWTOKENSLIMIT {
        return Err(clierror::invalidarguments {
            detail: format!("maxnewtokens {maxnewtokens} exceeds the {MAXNEWTOKENSLIMIT} limit"),
        });
    }
    Ok(())
}

/// Builds the validated tiny model configuration.
fn configfromvocab(vocab: usize) -> Result<llamaconfig, clierror> {
    if vocab == 0 {
        return Err(clierror::invalidarguments {
            detail: "vocab must be at least 1".to_string(),
        });
    }
    Ok(llamaconfig::new(
        vocab,
        TINYDIM,
        TINYLAYERS,
        TINYHEADS,
        TINYKVHEADS,
        TINYHIDDEN,
        TINYRMSEPS,
        TINYROPEBASE,
        TINYMAXSEQ,
    )?)
}

/// Builds the prompt token list from `--tokens` or from `--text` via the
/// tokenizer.
fn promptfromargs(args: &generateargs, vocab: usize) -> Result<Vec<usize>, clierror> {
    if let Some(tokens) = args.tokens.as_deref() {
        let parsed = parsetokenlist(tokens, "prompt", vocab)?;
        if parsed.is_empty() {
            return Err(clierror::invalidarguments {
                detail: "prompt must not be empty".to_string(),
            });
        }
        return Ok(parsed);
    }
    let text = args.text.as_deref().unwrap_or("");
    if text.is_empty() {
        return Err(clierror::invalidarguments {
            detail: "text prompt must not be empty".to_string(),
        });
    }
    let path = args
        .tokenizer
        .as_deref()
        .ok_or_else(|| clierror::invalidarguments {
            detail: "text requires --tokenizer".to_string(),
        })?;
    tokenize(path, text, vocab)
}

/// Encodes `text` with the tokenizer loaded from `path`.
fn tokenize(path: &Path, text: &str, vocab: usize) -> Result<Vec<usize>, clierror> {
    let bytes = readartifact(path, DEFAULTMAXARTIFACTBYTES)?;
    let document = std::str::from_utf8(&bytes).map_err(|_| clierror::invalidarguments {
        detail: "tokenizer file is not valid UTF-8".to_string(),
    })?;
    let loaded = tokenizer::fromjson(document)?;
    let ids = loaded.encode(text)?;
    if ids.len() > MAXPROMPTTOKENS {
        return Err(clierror::invalidarguments {
            detail: format!(
                "tokenized prompt has {} tokens, exceeding the {MAXPROMPTTOKENS} limit",
                ids.len()
            ),
        });
    }
    let mut tokens = Vec::with_capacity(ids.len());
    for id in ids {
        let token = id.get() as usize;
        checktoken(token, "prompt", vocab)?;
        tokens.push(token);
    }
    if tokens.is_empty() {
        return Err(clierror::invalidarguments {
            detail: "tokenized prompt is empty".to_string(),
        });
    }
    Ok(tokens)
}

/// Parses the optional `--eos` list.
fn parseeos(text: Option<&str>, vocab: usize) -> Result<Vec<usize>, clierror> {
    match text {
        None => Ok(Vec::new()),
        Some(value) => parsetokenlist(value, "eos", vocab),
    }
}

/// Parses a comma-separated token list and checks every id against `vocab`.
fn parsetokenlist(text: &str, field: &'static str, vocab: usize) -> Result<Vec<usize>, clierror> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let mut tokens = Vec::new();
    for part in trimmed.split(',') {
        let raw = part.trim();
        let value = raw.parse::<u64>().map_err(|_| clierror::invalidarguments {
            detail: format!("{field} contains a non-numeric token {raw:?}"),
        })?;
        let token = usize::try_from(value).map_err(|_| clierror::invalidarguments {
            detail: format!("{field} token {value} does not fit this platform"),
        })?;
        checktoken(token, field, vocab)?;
        tokens.push(token);
        if tokens.len() > MAXPROMPTTOKENS {
            return Err(clierror::invalidarguments {
                detail: format!("{field} has more than {MAXPROMPTTOKENS} tokens"),
            });
        }
    }
    Ok(tokens)
}

/// Rejects a token id outside the model vocabulary.
fn checktoken(token: usize, field: &'static str, vocab: usize) -> Result<(), clierror> {
    if token >= vocab {
        return Err(clierror::invalidarguments {
            detail: format!("{field} token {token} is outside the vocabulary of size {vocab}"),
        });
    }
    Ok(())
}

/// Renders the generation report.
fn formatresult(
    args: &generateargs,
    config: &llamaconfig,
    prompt: &[usize],
    result: &generationresult,
) -> String {
    let mut out = format!(
        "model: llama vocab={} dim={} layers={} heads={} maxseq={}\n",
        config.getvocab(),
        config.getdim(),
        config.getnlayers(),
        config.getnheads(),
        config.getmaxseq()
    );
    out.push_str(&format!("seed: {}\n", args.seed));
    out.push_str(&format!("prompt: {prompt:?}\n"));
    out.push_str(&format!("maxnewtokens: {}\n", args.maxnewtokens));
    out.push_str(&format!("generated: {:?}\n", result.gettokens()));
    out.push_str(&format!(
        "stopreason: {}\n",
        stopreasonname(result.getreason())
    ));
    out.push_str(&format!("steps: {}\n", result.getsteps()));
    out
}

/// Returns the stable name of a stop reason.
fn stopreasonname(reason: stopreason) -> &'static str {
    match reason {
        stopreason::endoftoken => "endoftoken",
        stopreason::maxlength => "maxlength",
        stopreason::stopped => "stopped",
    }
}
