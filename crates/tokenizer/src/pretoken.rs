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
    classrun(rest, |ch| ch.is_whitespace())
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
