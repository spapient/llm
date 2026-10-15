//! GPT-2 ByteLevel pre-tokenization: the byte-to-unicode alphabet and the
//! GPT-2 whitespace/contraction splitter.
//!
//! The GPT-2 model maps raw bytes to printable Unicode codepoints before
//! byte-pair encoding, and splits text with a fixed regex before that mapping.
//! This module owns those two fixed pieces as pure functions; wiring them into
//! `encode`/`decode` is the caller's concern.

/// The GPT-2 byte-to-unicode alphabet: `table[byte]` is the Unicode character
/// representing `byte`.
///
/// The 188 printable bytes `0x21..=0x7E`, `0xA1..=0xAC`, and `0xAE..=0xFF` map
/// to themselves; the remaining 68 bytes map to `U+0100..=U+0143` in ascending
/// byte order. The mapping is a bijection.
#[must_use]
pub fn bytetounicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 0x0100u32;
    for (byte, slot) in table.iter_mut().enumerate() {
        let codepoint = if directbyte(byte) {
            byte as u32
        } else {
            let mapped = next;
            next += 1;
            mapped
        };
        // `codepoint` is always in 33..=323, so it is a valid scalar value.
        *slot = char::from_u32(codepoint).unwrap_or('\u{fffd}');
    }
    table
}

/// Reports whether `byte` maps to itself rather than a private-use codepoint.
fn directbyte(byte: usize) -> bool {
    (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte)
}

/// The inverse of [`bytetounicode`]: `table[codepoint]` is the byte the
/// codepoint represents, for codepoints up to `U+0143`.
#[must_use]
pub fn unicodetobyte() -> [Option<u8>; 324] {
    let mut table = [None; 324];
    for (byte, character) in bytetounicode().iter().enumerate() {
        table[*character as usize] = Some(byte as u8);
    }
    table
}

/// Splits `text` with the GPT-2 pre-tokenizer regex
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`.
///
/// The splitter is hand-written (no regex engine): Unicode scalars are
/// classified with the standard library's `is_alphabetic`, `is_numeric`, and
/// `is_whitespace`, so `\p{L}` here is the `Alphabetic` derived property, a
/// superset of the Unicode `Letter` category. The trailing-whitespace
/// alternative `\s+(?!\S)` collapses to the same token length as `\s+`, so the
/// splitter need not distinguish the two.
#[must_use]
pub fn gpt2split(text: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let length = gpt2toklen(&text[start..]);
        tokens.push(&text[start..start + length]);
        start += length;
    }
    tokens
}

/// Returns the byte length of the next GPT-2 token in the non-empty `rest`.
fn gpt2toklen(rest: &str) -> usize {
    for contraction in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        if rest.starts_with(contraction) {
            return contraction.len();
        }
    }
    let Some(first) = rest.chars().next() else {
        return 0; // unreachable: `rest` is non-empty
    };
    if first.is_alphabetic() {
        return classrun(rest, |ch| ch.is_alphabetic());
    }
    if first.is_numeric() {
        return classrun(rest, |ch| ch.is_numeric());
    }
    if first == ' ' {
        let tail = &rest[1..];
        match tail.chars().next() {
            Some(ch) if ch.is_alphabetic() => return 1 + classrun(tail, |c| c.is_alphabetic()),
            Some(ch) if ch.is_numeric() => return 1 + classrun(tail, |c| c.is_numeric()),
            Some(ch) if !ch.is_whitespace() => return 1 + classrun(tail, otherchar),
            _ => {} // whitespace or end: fall through to a whitespace run
        }
    }
    if !first.is_whitespace() {
        return classrun(rest, otherchar);
    }
    // `\s+(?!\S)` is tried before `\s+`: a whitespace run followed by a
    // non-space backtracks to exclude its last whitespace, which the
    // ` ?\p{L}+`/` ?\p{N}+`/` ?[^...]+` alternatives then absorb as the
    // optional leading space of the next token.
    let runlen = classrun(rest, |ch| ch.is_whitespace());
    match rest[runlen..].chars().next() {
        Some(_) => runlen.saturating_sub(1).max(1),
        None => runlen,
    }
}

/// Returns the byte length of the leading run of characters satisfying `pred`.
fn classrun(rest: &str, pred: impl Fn(char) -> bool) -> usize {
    rest.chars()
        .take_while(|&ch| pred(ch))
        .map(char::len_utf8)
        .sum()
}

/// Reports whether `ch` is neither whitespace, a letter, nor a number.
fn otherchar(ch: char) -> bool {
    !ch.is_whitespace() && !ch.is_alphabetic() && !ch.is_numeric()
}

/// A pre-tokenizer: splits and/or transforms text before byte-pair encoding.
///
/// The result of [`pretokenize`](pretokenizer::pretokenize) is a list of
/// pre-token strings. A ByteLevel pre-tokenizer maps each byte to a
/// byte-to-unicode character, so the BPE then treats each of those characters
/// as one initial symbol; a Split or raw sequence leaves the text unmodified
/// so the BPE treats each Unicode scalar as one initial symbol.
#[derive(Clone, Debug)]
pub enum pretokenizer {
    /// ByteLevel: map bytes to the byte-to-unicode alphabet; with `useregex`
    /// also split by the GPT-2 regex first.
    bytelevel {
        /// Whether to apply the GPT-2 regex before byte-mapping.
        useregex: bool,
    },
    /// Split: isolate each regex match as its own segment.
    split {
        /// The compiled split pattern.
        pattern: fancy_regex::Regex,
    },
    /// Sequence: apply the children in order.
    sequence(Vec<pretokenizer>),
}

impl pretokenizer {
    /// Splits `text` into pre-token strings.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::invalidregex`](crate::tokenizererror::invalidregex)
    /// when a Split pattern hits a runtime matching error.
    pub fn pretokenize(&self, text: &str) -> Result<Vec<String>, crate::error::tokenizererror> {
        match self {
            Self::bytelevel { useregex } => {
                if *useregex {
                    Ok(gpt2split(text).into_iter().map(bytemap).collect())
                } else {
                    Ok(vec![bytemap(text)])
                }
            }
            Self::split { pattern } => splitisolated(pattern, text)
                .map(|segments| segments.into_iter().map(str::to_string).collect()),
            Self::sequence(children) => {
                let mut chunks = vec![text.to_string()];
                for child in children {
                    let mut next = Vec::new();
                    for chunk in &chunks {
                        next.extend(child.pretokenize(chunk)?);
                    }
                    chunks = next;
                }
                Ok(chunks)
            }
        }
    }
}

/// Maps `text`'s bytes to the byte-to-unicode alphabet.
fn bytemap(text: &str) -> String {
    let table = bytetounicode();
    text.bytes().map(|byte| table[byte as usize]).collect()
}

/// Splits `text`, isolating each regex match as its own segment.
///
/// Uses `fancy-regex`, which supports the look-around in the Split patterns
/// (the Rust `regex` crate does not).
fn splitisolated<'a>(
    pattern: &fancy_regex::Regex,
    text: &'a str,
) -> Result<Vec<&'a str>, crate::error::tokenizererror> {
    let mut out = Vec::new();
    let mut last = 0;
    for found in pattern.find_iter(text) {
        let found = found.map_err(|error| crate::error::tokenizererror::invalidregex {
            detail: error.to_string(),
        })?;
        if found.start() > last {
            out.push(&text[last..found.start()]);
        }
        out.push(&text[found.start()..found.end()]);
        last = found.end();
    }
    if last < text.len() {
        out.push(&text[last..]);
    }
    Ok(out)
}

/// Parses the `pre_tokenizer` section, if present.
pub(crate) fn parsepretokenizer(
    root: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<pretokenizer>, crate::error::tokenizererror> {
    let Some(value) = root.get("pre_tokenizer") else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    Ok(Some(parseone(value)?))
}

/// Parses one pre-tokenizer object.
fn parseone(value: &serde_json::Value) -> Result<pretokenizer, crate::error::tokenizererror> {
    use crate::error::tokenizererror;
    use serde_json::Value;

    let object = value.as_object().ok_or(tokenizererror::unexpectedtype {
        field: "pre_tokenizer",
        expected: "object",
    })?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or(tokenizererror::missingfield {
            field: "pre_tokenizer.type",
        })?;
    match kind {
        "ByteLevel" => Ok(pretokenizer::bytelevel {
            useregex: object
                .get("use_regex")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        }),
        "Split" => {
            let pattern = object
                .get("pattern")
                .and_then(|p| p.as_object())
                .and_then(|p| p.get("Regex"))
                .and_then(Value::as_str)
                .ok_or(tokenizererror::unsupportedpretokenizer {
                    pretype: "Split without Regex pattern".to_string(),
                })?;
            let compiled =
                fancy_regex::Regex::new(pattern).map_err(|error| tokenizererror::invalidregex {
                    detail: error.to_string(),
                })?;
            Ok(pretokenizer::split { pattern: compiled })
        }
        "Sequence" => {
            let children = object
                .get("pretokenizers")
                .and_then(Value::as_array)
                .ok_or(tokenizererror::missingfield {
                    field: "pre_tokenizer.pretokenizers",
                })?;
            let mut parsed = Vec::with_capacity(children.len());
            for child in children {
                parsed.push(parseone(child)?);
            }
            Ok(pretokenizer::sequence(parsed))
        }
        other => Err(tokenizererror::unsupportedpretokenizer {
            pretype: other.to_string(),
        }),
    }
}
