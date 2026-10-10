use lib::tokenid;

use crate::error::tokenizererror;
use crate::merges::bytepair;
use crate::special::specialtokens;
use crate::vocab::vocabulary;

/// Encodes `text` as ordinary text: every Unicode scalar is an initial symbol,
/// byte-pair merges are applied, and each resulting symbol is resolved to one
/// or more token ids.
///
/// Special-token literals in `text` are not treated as structural here; use
/// [`encodespecial`] for that.
pub(crate) fn encodeplain(
    vocab: &vocabulary,
    bpe: &bytepair,
    bytefallback: bool,
    unk: Option<tokenid>,
    text: &str,
) -> Result<Vec<tokenid>, tokenizererror> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let symbols: Vec<String> = text
        .chars()
        .map(|character| character.to_string())
        .collect();
    let merged = bpe.apply(&symbols);
    let mut out = Vec::with_capacity(merged.len());
    for symbol in &merged {
        resolvetoken(vocab, bytefallback, unk, symbol, &mut out)?;
    }
    Ok(out)
}

/// Encodes `text`, recognizing special-token literals as structural markers.
///
/// The input is scanned left to right. At each position the longest special
/// token whose literal matches is emitted directly; the text between special
/// tokens is encoded with [`encodeplain`].
pub(crate) fn encodespecial(
    vocab: &vocabulary,
    bpe: &bytepair,
    bytefallback: bool,
    unk: Option<tokenid>,
    specials: &specialtokens,
    text: &str,
) -> Result<Vec<tokenid>, tokenizererror> {
    let mut literals: Vec<(&str, tokenid)> = specials.iter().collect();
    // Longest literal first so that overlapping markers resolve deterministically.
    literals.sort_by(|left, right| {
        right
            .0
            .len()
            .cmp(&left.0.len())
            .then_with(|| left.0.cmp(right.0))
    });

    let mut out = Vec::new();
    let mut segment = String::new();
    let mut index = 0;
    while index < text.len() {
        if let Some((literal, id)) = literals
            .iter()
            .find(|(literal, _)| text[index..].starts_with(literal))
        {
            flushsegment(vocab, bpe, bytefallback, unk, &mut segment, &mut out)?;
            out.push(*id);
            index += literal.len();
            continue;
        }
        let character = text[index..]
            .chars()
            .next()
            .ok_or(tokenizererror::invalidutf8)?;
        segment.push(character);
        index += character.len_utf8();
    }
    flushsegment(vocab, bpe, bytefallback, unk, &mut segment, &mut out)?;
    Ok(out)
}

fn flushsegment(
    vocab: &vocabulary,
    bpe: &bytepair,
    bytefallback: bool,
    unk: Option<tokenid>,
    segment: &mut String,
    out: &mut Vec<tokenid>,
) -> Result<(), tokenizererror> {
    if segment.is_empty() {
        return Ok(());
    }
    let mut ids = encodeplain(vocab, bpe, bytefallback, unk, segment)?;
    out.append(&mut ids);
    segment.clear();
    Ok(())
}

/// Resolves one merged symbol to token ids, appending them to `out`.
///
/// Resolution order: an exact vocabulary match; then, if byte fallback is
/// enabled and every UTF-8 byte has a `<0xNN>` token, those byte tokens; then
/// the configured unknown token; otherwise a [`tokenizererror::unknowntoken`].
fn resolvetoken(
    vocab: &vocabulary,
    bytefallback: bool,
    unk: Option<tokenid>,
    symbol: &str,
    out: &mut Vec<tokenid>,
) -> Result<(), tokenizererror> {
    if let Some(id) = vocab.getid(symbol) {
        out.push(id);
        return Ok(());
    }
    if bytefallback {
        let mut bytes = Vec::new();
        let mut complete = true;
        for byte in symbol.bytes() {
            let token = format!("<0x{byte:02X}>");
            match vocab.getid(&token) {
                Some(id) => bytes.push(id),
                None => {
                    complete = false;
                    break;
                }
            }
        }
        if complete {
            out.extend(bytes);
            return Ok(());
        }
    }
    if let Some(id) = unk {
        out.push(id);
        return Ok(());
    }
    Err(tokenizererror::unknowntoken {
        token: symbol.to_string(),
    })
}
