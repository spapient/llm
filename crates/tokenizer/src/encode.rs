use lib::tokenid;

use crate::error::tokenizererror;
use crate::merges::bytepair;
use crate::pretoken::{bytetounicode, gpt2split};
use crate::special::specialtokens;
use crate::vocab::vocabulary;

/// Encodes `text` as ordinary text.
///
/// When `bytelevel` is set the GPT-2 ByteLevel pipeline is used: the text is
/// split with the GPT-2 splitter, each chunk's bytes are mapped to the
/// byte-to-unicode alphabet, and merges run within each chunk without crossing
/// a chunk boundary. Otherwise every Unicode scalar is an initial symbol.
///
/// Special-token literals in `text` are not treated as structural here; use
/// [`encodespecial`] for that.
pub(crate) fn encodeplain(
    vocab: &vocabulary,
    bpe: &bytepair,
    bytefallback: bool,
    unk: Option<tokenid>,
    bytelevel: bool,
    text: &str,
) -> Result<Vec<tokenid>, tokenizererror> {
    if bytelevel {
        return encodebytelevel(vocab, bpe, bytefallback, unk, text);
    }
    encodechars(vocab, bpe, bytefallback, unk, text)
}

/// Char-level encoding: every Unicode scalar is an initial symbol.
fn encodechars(
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

/// Byte-level encoding: split with the GPT-2 splitter, map each chunk's bytes
/// to the byte-to-unicode alphabet, and apply merges within each chunk.
///
/// The per-call cache reuses a chunk's encoding when the same pre-token appears
/// more than once in `text`; it is call-local so it adds no shared state.
fn encodebytelevel(
    vocab: &vocabulary,
    bpe: &bytepair,
    bytefallback: bool,
    unk: Option<tokenid>,
    text: &str,
) -> Result<Vec<tokenid>, tokenizererror> {
    let table = bytetounicode();
    let mut out = Vec::new();
    let mut cache: std::collections::HashMap<&str, Vec<tokenid>> = std::collections::HashMap::new();
    for chunk in gpt2split(text) {
        if let Some(ids) = cache.get(chunk) {
            out.extend_from_slice(ids);
            continue;
        }
        let symbols: Vec<String> = chunk
            .bytes()
            .map(|byte| table[byte as usize].to_string())
            .collect();
        let merged = bpe.apply(&symbols);
        let mut ids = Vec::with_capacity(merged.len());
        for symbol in &merged {
            resolvetoken(vocab, bytefallback, unk, symbol, &mut ids)?;
        }
        out.extend_from_slice(&ids);
        cache.insert(chunk, ids);
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
    bytelevel: bool,
    specials: &specialtokens,
    text: &str,
) -> Result<Vec<tokenid>, tokenizererror> {
    let mut out = Vec::new();
    let mut segment = String::new();
    let mut index = 0;
    while index < text.len() {
        if let Some((literal, id)) = specials
            .iterlongest()
            .find(|(literal, _)| text[index..].starts_with(literal))
        {
            flushsegment(
                vocab,
                bpe,
                bytefallback,
                unk,
                bytelevel,
                &mut segment,
                &mut out,
            )?;
            out.push(id);
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
    flushsegment(
        vocab,
        bpe,
        bytefallback,
        unk,
        bytelevel,
        &mut segment,
        &mut out,
    )?;
    Ok(out)
}

fn flushsegment(
    vocab: &vocabulary,
    bpe: &bytepair,
    bytefallback: bool,
    unk: Option<tokenid>,
    bytelevel: bool,
    segment: &mut String,
    out: &mut Vec<tokenid>,
) -> Result<(), tokenizererror> {
    if segment.is_empty() {
        return Ok(());
    }
    let mut ids = encodeplain(vocab, bpe, bytefallback, unk, bytelevel, segment)?;
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
        let mut ids = Vec::new();
        let mut complete = true;
        for byte in symbol.bytes() {
            match vocab.getbytetokenid(byte) {
                Some(id) => ids.push(id),
                None => {
                    complete = false;
                    break;
                }
            }
        }
        if complete {
            out.extend(ids);
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
