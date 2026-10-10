use lib::tokenid;

use crate::error::tokenizererror;
use crate::vocab::vocabulary;

/// Decodes token ids to a string.
///
/// Every id is validated against `vocab`; an id that is not present produces
/// [`tokenizererror::unknowntokenid`] rather than being silently skipped. A
/// token of the byte-fallback form `<0xNN>` contributes the byte `NN` to a
/// pending byte buffer; any other token flushes that buffer (as UTF-8) and is
/// appended verbatim. A pending buffer that is not valid UTF-8 at a flush point
/// produces [`tokenizererror::invalidutf8`].
pub(crate) fn decodetokens(vocab: &vocabulary, ids: &[tokenid]) -> Result<String, tokenizererror> {
    let mut out = String::new();
    let mut bytes: Vec<u8> = Vec::new();
    for id in ids {
        let content = vocab
            .getcontent(*id)
            .ok_or(tokenizererror::unknowntokenid { id: id.get() })?;
        match parsebytetoken(content) {
            Some(byte) => bytes.push(byte),
            None => {
                flushbytes(&mut bytes, &mut out)?;
                out.push_str(content);
            }
        }
    }
    flushbytes(&mut bytes, &mut out)?;
    Ok(out)
}

/// Parses a `<0xNN>` byte-fallback token to its byte value, if it is well formed.
fn parsebytetoken(content: &str) -> Option<u8> {
    let inner = content.strip_prefix("<0x")?.strip_suffix('>')?;
    if inner.len() != 2 {
        return None;
    }
    u8::from_str_radix(inner, 16).ok()
}

fn flushbytes(bytes: &mut Vec<u8>, out: &mut String) -> Result<(), tokenizererror> {
    if bytes.is_empty() {
        return Ok(());
    }
    let taken = core::mem::take(bytes);
    let decoded = String::from_utf8(taken).map_err(|_| tokenizererror::invalidutf8)?;
    out.push_str(&decoded);
    Ok(())
}
