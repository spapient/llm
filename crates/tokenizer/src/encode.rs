use lib::tokenid;

use crate::error::tokenizererror;
use crate::vocab::vocabulary;

/// Resolves one merged symbol to token ids, appending them to `out`.
///
/// Resolution order: an exact vocabulary match; then, if byte fallback is
/// enabled and every UTF-8 byte has a `<0xNN>` token, those byte tokens; then
/// the configured unknown token; otherwise a [`tokenizererror::unknowntoken`].
pub(crate) fn resolvetoken(
    vocab: &vocabulary,
    bytefallback: bool,
    unk: Option<tokenid>,
    symbol: &str,
    bytetokens: &mut Vec<tokenid>,
    out: &mut Vec<tokenid>,
) -> Result<(), tokenizererror> {
    if let Some(id) = vocab.getid(symbol) {
        out.push(id);
        return Ok(());
    }
    resolvenonvocab(vocab, bytefallback, unk, symbol, bytetokens, out)
}

/// Resolves a symbol already known to be absent from the vocabulary through
/// byte fallback, the unknown token, or an error. `bytetokens` is a reusable
/// buffer the caller owns, so a byte-fallback symbol does not allocate per call.
pub(crate) fn resolvenonvocab(
    vocab: &vocabulary,
    bytefallback: bool,
    unk: Option<tokenid>,
    symbol: &str,
    bytetokens: &mut Vec<tokenid>,
    out: &mut Vec<tokenid>,
) -> Result<(), tokenizererror> {
    if bytefallback {
        bytetokens.clear();
        let mut complete = true;
        for byte in symbol.bytes() {
            match vocab.getbytetokenid(byte) {
                Some(id) => bytetokens.push(id),
                None => {
                    complete = false;
                    break;
                }
            }
        }
        if complete {
            out.extend_from_slice(bytetokens);
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
