//! GPT-2 ByteLevel pre-tokenization: the byte-to-unicode alphabet and the
//! GPT-2 whitespace/contraction splitter.
//!
//! The GPT-2 model maps raw bytes to printable Unicode codepoints before
//! byte-pair encoding, and splits text with a fixed regex before that mapping.
//! This module owns those two fixed pieces as pure functions; wiring them into
//! `encode`/`decode` is the caller's concern.

use std::sync::LazyLock;

/// The GPT-2 byte-to-unicode alphabet, computed once on first use.
static BYTEALPHABET: LazyLock<[char; 256]> = LazyLock::new(|| {
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
});

/// The GPT-2 byte-to-unicode alphabet: `table[byte]` is the Unicode character
/// representing `byte`.
///
/// The 188 printable bytes `0x21..=0x7E`, `0xA1..=0xAC`, and `0xAE..=0xFF` map
/// to themselves; the remaining 68 bytes map to `U+0100..=U+0143` in ascending
/// byte order. The mapping is a bijection.
#[must_use]
pub fn bytetounicode() -> &'static [char; 256] {
    &BYTEALPHABET
}

/// Reports whether `byte` maps to itself rather than a private-use codepoint.
fn directbyte(byte: usize) -> bool {
    (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte)
}

/// The inverse of [`bytetounicode`], computed once on first use.
static UNICODEBYTES: LazyLock<[Option<u8>; 324]> = LazyLock::new(|| {
    let mut table = [None; 324];
    for (byte, character) in BYTEALPHABET.iter().enumerate() {
        table[*character as usize] = Some(byte as u8);
    }
    table
});

/// The inverse of [`bytetounicode`]: `table[codepoint]` is the byte the
/// codepoint represents, for codepoints up to `U+0143`.
#[must_use]
pub fn unicodetobyte() -> &'static [Option<u8>; 324] {
    &UNICODEBYTES
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

/// Splits `text` with the o200k pre-tokenizer regex (from tiktoken's
/// `openai_public` module):
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`.
///
/// Like [`gpt2split`], this is hand-written (no regex engine) and classifies
/// Unicode scalars with the standard library's `is_alphabetic`, `is_numeric`,
/// and `is_whitespace`.
#[must_use]
pub fn o200ksplit(text: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let length = o200ktoklen(&text[start..]);
        tokens.push(&text[start..start + length]);
        start += length;
    }
    tokens
}

/// Returns the byte length of the next o200k token in the non-empty `rest`.
fn o200ktoklen(rest: &str) -> usize {
    // 1. Case-insensitive contractions `'s|'t|'re|'ve|'m|'ll|'d` (ASCII-only).
    let bytes = rest.as_bytes();
    for contraction in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        if bytes.len() >= contraction.len()
            && bytes[..contraction.len()].eq_ignore_ascii_case(contraction.as_bytes())
        {
            return contraction.len();
        }
    }
    let Some(first) = rest.chars().next() else {
        return 0; // unreachable: `rest` is non-empty
    };
    let firstlen = first.len_utf8();
    // 3. `\p{N}{1,3}`: one to three digits, with no leading space.
    if first.is_numeric() {
        let mut total = 0;
        let mut count = 0;
        for ch in rest.chars() {
            if ch.is_numeric() && count < 3 {
                total += ch.len_utf8();
                count += 1;
            } else {
                break;
            }
        }
        return total;
    }
    // 2. `[^\r\n\p{L}\p{N}]?\p{L}+`: an optional single non-CR/LF, non-letter,
    //    non-number character followed by one or more letters.
    if first.is_alphabetic() {
        return classrun(rest, |ch| ch.is_alphabetic());
    }
    if first != '\r'
        && first != '\n'
        && let Some(second) = rest[firstlen..].chars().next()
        && second.is_alphabetic()
    {
        return firstlen + classrun(&rest[firstlen..], |ch| ch.is_alphabetic());
    }
    // 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*`: optional space + 1+ punctuation + 0+ CR/LF.
    if first == ' ' {
        let tail = &rest[firstlen..];
        if let Some(ch) = tail.chars().next()
            && otherchar(ch)
        {
            let run = classrun(tail, otherchar);
            return firstlen + run + crlfrun(&tail[run..]);
        }
    }
    if otherchar(first) {
        let run = classrun(rest, otherchar);
        return run + crlfrun(&rest[run..]);
    }
    // 5-7. Whitespace alternatives.
    if first.is_whitespace() {
        let runlen = classrun(rest, |ch| ch.is_whitespace());
        // 5. `\s*[\r\n]+`: whitespace up to and including the last CR/LF.
        if let Some(last) = rest[..runlen]
            .bytes()
            .rposition(|byte| byte == b'\r' || byte == b'\n')
        {
            return last + 1;
        }
        // 6. `\s+(?!\S)` then 7. `\s+`: trailing whitespace backs off one.
        return match rest[runlen..].chars().next() {
            Some(_) => runlen.saturating_sub(1).max(1),
            None => runlen,
        };
    }
    firstlen // unreachable: every character is classified above
}

/// Returns the byte length of the leading run of CR/LF bytes.
fn crlfrun(rest: &str) -> usize {
    rest.bytes()
        .take_while(|&byte| byte == b'\r' || byte == b'\n')
        .count()
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

/// Reports whether `ch` is a Han ideograph, approximating the Unicode `\p{Han}`
/// property over its main blocks (basic, extensions A–H, compatibility). The
/// standard library exposes no `\p{Han}` query, and the splitter is hand-written
/// without a regex engine.
fn is_han(ch: char) -> bool {
    matches!(u32::from(ch),
        0x3400..=0x4DBF   // Extension A
        | 0x4E00..=0x9FFF // Unified
        | 0xF900..=0xFAFF // Compatibility
        | 0x20000..=0x2A6DF // Extension B
        | 0x2A700..=0x2B73F // Extension C
        | 0x2B740..=0x2B81F // Extension D
        | 0x2B820..=0x2CEAF // Extension E
        | 0x2CEB0..=0x2EBEF // Extension F
        | 0x30000..=0x3134F // Extension G
        | 0x31350..=0x323AF // Extension H
    )
}

/// Returns the byte length of an optional Kimi contraction suffix
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)` at the start of `rest`.
fn contractionsuffix(rest: &str) -> usize {
    for contraction in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        if rest.len() >= contraction.len()
            && rest.is_char_boundary(contraction.len())
            && rest[..contraction.len()].eq_ignore_ascii_case(contraction)
        {
            return contraction.len();
        }
    }
    0
}

/// Returns `(len, end)` of the letter run at the start of `rest` under the Kimi
/// two-class split: an upper/title-case start consumes the upper run then the
/// lower run; a lower-case start consumes only the lower run. Han is excluded
/// (already consumed by the Han-first branch).
fn hanletterrun(rest: &str) -> (usize, usize) {
    let Some(first) = rest.chars().next() else {
        return (0, 0); // unreachable: callers pass non-empty text
    };
    if first.is_uppercase() {
        let upper = classrun(rest, |ch| ch.is_uppercase() && !is_han(ch));
        let lower = classrun(&rest[upper..], |ch| ch.is_lowercase() && !is_han(ch));
        (upper + lower, upper + lower)
    } else {
        let lower = classrun(rest, |ch| ch.is_lowercase() && !is_han(ch));
        (lower, lower)
    }
}

/// Splits `text` with the Kimi K2 pre-tokenizer regex (DeepSeek-style
/// `pat_str`), hand-written like [`o200ksplit`]:
/// `[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`.
///
/// Han ideographs form a single run first (whereas [`o200ksplit`] folds them
/// into the letter class), and letter runs split upper/lower with an optional
/// contraction suffix. The digit, punctuation, newline and whitespace
/// alternatives match [`o200ksplit`].
#[must_use]
pub fn hanfirstsplit(text: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let length = hanfirsttoklen(&text[start..]);
        tokens.push(&text[start..start + length]);
        start += length;
    }
    tokens
}

/// Returns the byte length of the next Kimi pre-token in the non-empty `rest`.
fn hanfirsttoklen(rest: &str) -> usize {
    let Some(first) = rest.chars().next() else {
        return 0; // unreachable: `rest` is non-empty
    };
    let firstlen = first.len_utf8();
    // 1. `[\p{Han}]+`: a whole run of Han ideographs.
    if is_han(first) {
        return classrun(rest, is_han);
    }
    // 2/3. Letter runs (Han excluded), each with an optional single non-letter,
    //    non-number prefix and an optional contraction suffix.
    if first.is_alphabetic() {
        let (body, end) = hanletterrun(rest);
        return body + contractionsuffix(&rest[end..]);
    }
    if first != '\r'
        && first != '\n'
        && let Some(second) = rest[firstlen..].chars().next()
        && second.is_alphabetic()
        && !is_han(second)
    {
        let (body, end) = hanletterrun(&rest[firstlen..]);
        return firstlen + body + contractionsuffix(&rest[firstlen + end..]);
    }
    // 4. `\p{N}{1,3}`: one to three digits, with no leading space.
    if first.is_numeric() {
        let mut total = 0;
        let mut count = 0;
        for ch in rest.chars() {
            if ch.is_numeric() && count < 3 {
                total += ch.len_utf8();
                count += 1;
            } else {
                break;
            }
        }
        return total;
    }
    // 5. ` ?[^\s\p{L}\p{N}]+[\r\n]*`: optional space + 1+ punctuation + 0+ CR/LF.
    if first == ' ' {
        let tail = &rest[firstlen..];
        if let Some(ch) = tail.chars().next()
            && otherchar(ch)
        {
            let run = classrun(tail, otherchar);
            return firstlen + run + crlfrun(&tail[run..]);
        }
    }
    if otherchar(first) {
        let run = classrun(rest, otherchar);
        return run + crlfrun(&rest[run..]);
    }
    // 6-8. Whitespace alternatives, identical to `o200ktoklen`.
    if first.is_whitespace() {
        let runlen = classrun(rest, |ch| ch.is_whitespace());
        if let Some(last) = rest[..runlen]
            .bytes()
            .rposition(|byte| byte == b'\r' || byte == b'\n')
        {
            return last + 1;
        }
        return match rest[runlen..].chars().next() {
            Some(_) => runlen.saturating_sub(1).max(1),
            None => runlen,
        };
    }
    firstlen // unreachable: every character is classified above
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
    /// Metaspace (SentencePiece): map ASCII spaces to `replacement` (usually
    /// `▁`), prepend one at the start when `prepend` holds and the text does not
    /// begin with a space, and isolate each replacement as its own segment when
    /// `split` holds (BPE consumes the replacement as a vocabulary symbol).
    metaspace {
        /// The space replacement string.
        replacement: String,
        /// Prepend one replacement unless the text begins with a space.
        prepend: bool,
        /// Isolate each replacement as its own segment.
        split: bool,
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
            Self::metaspace {
                replacement,
                prepend,
                split,
            } => {
                let mut mapped = String::with_capacity(text.len() + replacement.len());
                if *prepend && !text.starts_with(' ') {
                    mapped.push_str(replacement);
                }
                for ch in text.chars() {
                    if ch == ' ' {
                        mapped.push_str(replacement);
                    } else {
                        mapped.push(ch);
                    }
                }
                if *split {
                    Ok(metaspacesplit(&mapped, replacement)
                        .into_iter()
                        .map(str::to_string)
                        .collect())
                } else {
                    Ok(vec![mapped])
                }
            }
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

/// Splits `mapped` at every occurrence of `replacement`; each segment starts at
/// a replacement and ends before the next one. Text with no replacement forms
/// one segment. Mirrors the SentencePiece Metaspace `split = true` behavior.
fn metaspacesplit<'a>(mapped: &'a str, replacement: &str) -> Vec<&'a str> {
    let mut positions = Vec::new();
    let mut scan = 0;
    while scan < mapped.len() {
        match mapped[scan..].find(replacement) {
            Some(offset) => {
                positions.push(scan + offset);
                scan += offset + replacement.len();
            }
            None => break,
        }
    }
    if positions.is_empty() {
        return if mapped.is_empty() {
            Vec::new()
        } else {
            vec![mapped]
        };
    }
    positions
        .iter()
        .enumerate()
        .map(|(index, &start)| {
            let end = positions.get(index + 1).copied().unwrap_or(mapped.len());
            &mapped[start..end]
        })
        .collect()
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
        "Metaspace" => {
            let replacement = object
                .get("replacement")
                .and_then(Value::as_str)
                .unwrap_or("▁")
                .to_string();
            let add_prefix_space = object
                .get("add_prefix_space")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let prepend = object
                .get("prepend_scheme")
                .and_then(Value::as_str)
                .is_some_and(|scheme| scheme != "never")
                || add_prefix_space;
            let split = object
                .get("split")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            Ok(pretokenizer::metaspace {
                replacement,
                prepend,
                split,
            })
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
