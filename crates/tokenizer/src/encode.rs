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
