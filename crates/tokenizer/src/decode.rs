use lib::tokenid;

use crate::error::tokenizererror;
use crate::pretoken::unicodetobyte;
use crate::vocab::vocabulary;

/// Byte-level decoding: every character of every token maps through the
/// byte-to-unicode alphabet back to a byte, and the bytes reconstruct UTF-8.
pub(crate) fn decodebytelevel(
    vocab: &vocabulary,
    ids: &[tokenid],
) -> Result<String, tokenizererror> {
    let table = unicodetobyte();
    let mut bytes = Vec::new();
    for id in ids {
        let content = vocab
            .getcontent(*id)
            .ok_or(tokenizererror::unknowntokenid { id: id.get() })?;
        for character in content.chars() {
            let codepoint = character as usize;
            if codepoint < table.len()
                && let Some(byte) = table[codepoint]
            {
                bytes.push(byte);
                continue;
            }
            // A character outside the byte-to-unicode alphabet (e.g. from a
            // full-width special token) is decoded literally.
            let mut buffer = [0u8; 4];
            bytes.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
        }
    }
    String::from_utf8(bytes).map_err(|_| tokenizererror::invalidutf8)
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

/// A decoder: converts token contents back to text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum decoder {
    /// ByteLevel inverse mapping (byte-to-unicode → bytes).
    bytelevel,
    /// Byte-fallback (`<0xNN>` tokens) with optional literal replacement and
    /// leading-whitespace strip.
    bytefallback {
        /// A literal replace applied to non-byte tokens (e.g. `▁` → space).
        replace: Option<(String, String)>,
        /// Number of leading spaces to strip.
        stripleading: usize,
    },
    /// Metaspace (SentencePiece): map `replacement` back to a space in every
    /// token, then strip `stripleading` leading spaces.
    metaspace {
        /// The space replacement string (usually `▁`).
        replacement: String,
        /// Number of leading spaces to strip.
        stripleading: usize,
    },
}

impl decoder {
    /// Decodes `ids` to text.
    pub(crate) fn decode(
        &self,
        vocab: &vocabulary,
        ids: &[tokenid],
    ) -> Result<String, tokenizererror> {
        match self {
            Self::bytelevel => decodebytelevel(vocab, ids),
            Self::bytefallback {
                replace,
                stripleading,
            } => {
                let replace = replace
                    .as_ref()
                    .map(|(pattern, content)| (pattern.as_str(), content.as_str()));
                let mut out = decodebytefallback(vocab, ids, replace)?;
                for _ in 0..*stripleading {
                    if out.starts_with(' ') {
                        out.remove(0);
                    }
                }
                Ok(out)
            }
            Self::metaspace {
                replacement,
                stripleading,
            } => {
                let mut out = String::new();
                for id in ids {
                    let content = vocab
                        .getcontent(*id)
                        .ok_or(tokenizererror::unknowntokenid { id: id.get() })?;
                    out.push_str(&content.replace(replacement, " "));
                }
                for _ in 0..*stripleading {
                    if out.starts_with(' ') {
                        out.remove(0);
                    }
                }
                Ok(out)
            }
        }
    }
}

/// Byte-fallback decoding with an optional literal replacement on non-byte
/// tokens.
fn decodebytefallback(
    vocab: &vocabulary,
    ids: &[tokenid],
    replace: Option<(&str, &str)>,
) -> Result<String, tokenizererror> {
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
                match replace {
                    Some((pattern, replacement)) => {
                        out.push_str(&content.replace(pattern, replacement));
                    }
                    None => out.push_str(content),
                }
            }
        }
    }
    flushbytes(&mut bytes, &mut out)?;
    Ok(out)
}

/// Parses the `decoder` section, if present.
pub(crate) fn parsedecoder(
    root: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<decoder>, tokenizererror> {
    let Some(value) = root.get("decoder") else {
        return Ok(None);
    };
    Ok(Some(parsedecoderone(value)?))
}

/// Parses one decoder object, flattening the common `Sequence` of
/// `Replace` + `ByteFallback` (+ `Fuse`) + `Strip` into one value.
fn parsedecoderone(value: &serde_json::Value) -> Result<decoder, tokenizererror> {
    use serde_json::Value;

    let object = value.as_object().ok_or(tokenizererror::unexpectedtype {
        field: "decoder",
        expected: "object",
    })?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or(tokenizererror::missingfield {
            field: "decoder.type",
        })?;
    match kind {
        "ByteLevel" => Ok(decoder::bytelevel),
        "Metaspace" => Ok(parsemetaspace(object)),
        "Sequence" => {
            let children = object.get("decoders").and_then(Value::as_array).ok_or(
                tokenizererror::missingfield {
                    field: "decoder.decoders",
                },
            )?;
            parsesedecodersequence(children)
        }
        other => Err(tokenizererror::unsupporteddecoder {
            decodertype: other.to_string(),
        }),
    }
}

/// Parses a `Metaspace` decoder.
fn parsemetaspace(object: &serde_json::Map<String, serde_json::Value>) -> decoder {
    let replacement = object
        .get("replacement")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("▁")
        .to_string();
    let add_prefix_space = object
        .get("add_prefix_space")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    decoder::metaspace {
        replacement,
        stripleading: if add_prefix_space { 1 } else { 0 },
    }
}

/// Parses a `Sequence` of `Replace` + `ByteFallback` (+ `Fuse`) + `Strip`.
fn parsesedecodersequence(children: &[serde_json::Value]) -> Result<decoder, tokenizererror> {
    use serde_json::Value;

    let mut replace = None;
    let mut stripleading = 0usize;
    for child in children {
        let childtype = child
            .as_object()
            .and_then(|object| object.get("type"))
            .and_then(Value::as_str)
            .ok_or(tokenizererror::unsupporteddecoder {
                decodertype: "malformed decoder entry".to_string(),
            })?;
        match childtype {
            "Replace" => {
                let pattern = child
                    .as_object()
                    .and_then(|object| object.get("pattern"))
                    .and_then(Value::as_object)
                    .and_then(|pattern| pattern.get("String"))
                    .and_then(Value::as_str);
                let content = child
                    .as_object()
                    .and_then(|object| object.get("content"))
                    .and_then(Value::as_str);
                if let (Some(pattern), Some(content)) = (pattern, content) {
                    replace = Some((pattern.to_string(), content.to_string()));
                }
            }
            "ByteFallback" | "Fuse" => {}
            "Strip" => {
                let start = child
                    .as_object()
                    .and_then(|object| object.get("start"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                stripleading = start as usize;
            }
            other => {
                return Err(tokenizererror::unsupporteddecoder {
                    decodertype: other.to_string(),
                });
            }
        }
    }
    Ok(decoder::bytefallback {
        replace,
        stripleading,
    })
}
