use lib::tokenid;
use tokenizer::tokenizer;

use crate::DEFAULTMAXARTIFACTBYTES;
use crate::args::decodeargs;
use crate::clierror;
use crate::commandoutput;
use crate::commands::readartifact;

/// Decodes a token-id list to text with the tokenizer at `--tokenizer`.
///
/// # Errors
///
/// Returns a [`clierror`] for a non-numeric id, an empty list, a path or read
/// failure, a non-UTF-8 tokenizer file, a tokenizer load failure, or an id
/// absent from the vocabulary.
pub(crate) fn run(args: &decodeargs) -> Result<commandoutput, clierror> {
    let ids = parsetokenids(&args.tokens)?;
    let bytes = readartifact(&args.tokenizer, DEFAULTMAXARTIFACTBYTES)?;
    let document = std::str::from_utf8(&bytes).map_err(|_| clierror::invalidarguments {
        detail: "tokenizer file is not valid UTF-8".to_string(),
    })?;
    let loaded = tokenizer::fromjson(document)?;
    let text = loaded.decode(&ids)?;
    Ok(commandoutput::new(text))
}

/// Parses a comma-separated token-id list into `tokenid`s.
fn parsetokenids(input: &str) -> Result<Vec<tokenid>, clierror> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(clierror::invalidarguments {
            detail: "tokens must not be empty".to_string(),
        });
    }
    let mut ids = Vec::new();
    for part in trimmed.split(',') {
        let raw = part.trim();
        let value = raw.parse::<u32>().map_err(|_| clierror::invalidarguments {
            detail: format!("tokens contains a non-numeric id {raw:?}"),
        })?;
        ids.push(tokenid::new(value));
    }
    Ok(ids)
}
