//! The OpenAI tiktoken `.tiktoken` format (GPT-4o `o200k_base` and friends).
//!
//! Unlike a HuggingFace `tokenizer.json`, a `.tiktoken` file holds only the
//! mergeable vocabulary — one `<base64 token bytes> <rank>` per line — with no
//! merges table, no pre-tokenizer, and no decoder. Encoding runs the fixed
//! o200k regex to split, then greedily merges adjacent bytes whose
//! concatenation is a vocabulary token (lowest resulting rank first). Special
//! tokens are supplied by the caller because they are not in the file.

use std::collections::HashMap;

use lib::tokenid;

use crate::error::tokenizererror;

/// The o200k_base pre-tokenizer regex (from tiktoken's openai_public module).
const O200KREGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// A tiktoken-format encoder.
#[derive(Clone, Debug)]
pub struct tiktokenizer {
    /// Token bytes → rank (mergeable vocabulary).
    ranks: HashMap<Vec<u8>, u32>,
    /// Rank → token bytes (mergeable and special).
    tokens: Vec<Vec<u8>>,
    /// Special token content → rank, sorted longest first.
    specials: Vec<(String, u32)>,
    /// The pre-tokenizer regex.
    regex: fancy_regex::Regex,
}

impl tiktokenizer {
    /// Parses a `.tiktoken` document. `specials` supplies the special-token
    /// `(content, rank)` pairs that the file does not contain.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::invalidtiktoken`] for a malformed line or a
    /// base64 token, and [`tokenizererror::invalidregex`] if the built-in
    /// regex fails to compile.
    pub fn fromfile(content: &str, specials: &[(String, u32)]) -> Result<Self, tokenizererror> {
        let mut ranks = HashMap::new();
        for (index, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let (encoded, rank) =
                line.split_once(' ')
                    .ok_or_else(|| tokenizererror::invalidtiktoken {
                        detail: format!("line {index} is not `token rank`"),
                    })?;
            let bytes = decodebase64(encoded).map_err(|_| tokenizererror::invalidtiktoken {
                detail: format!("line {index} has invalid base64"),
            })?;
            let rank = rank
                .parse::<u32>()
                .map_err(|_| tokenizererror::invalidtiktoken {
                    detail: format!("line {index} has a non-numeric rank"),
                })?;
            ranks.insert(bytes, rank);
        }

        let maxrank = ranks
            .values()
            .copied()
            .chain(specials.iter().map(|(_, rank)| *rank))
            .max()
            .unwrap_or(0);
        let mut tokens = vec![Vec::new(); maxrank as usize + 1];
        for (bytes, rank) in &ranks {
            tokens[*rank as usize] = bytes.clone();
        }
        let mut specials_sorted: Vec<(String, u32)> = specials.to_vec();
        specials_sorted.sort_by(|left, right| {
            right
                .0
                .len()
                .cmp(&left.0.len())
                .then_with(|| left.0.cmp(&right.0))
        });
        for (content, rank) in &specials_sorted {
            tokens[*rank as usize] = content.as_bytes().to_vec();
        }
        let regex =
            fancy_regex::Regex::new(O200KREGEX).map_err(|error| tokenizererror::invalidregex {
                detail: error.to_string(),
            })?;
        Ok(Self {
            ranks,
            tokens,
            specials: specials_sorted,
            regex,
        })
    }

    /// Encodes `text`, recognizing the supplied special-token literals.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::invalidregex`] for a runtime regex failure.
    pub fn encode(&self, text: &str) -> Result<Vec<tokenid>, tokenizererror> {
        let mut out = Vec::new();
        let mut segment = String::new();
        let mut index = 0;
        while index < text.len() {
            if let Some((length, rank)) = self.matching_special(&text[index..]) {
                self.encode_segment(&segment, &mut out)?;
                segment.clear();
                out.push(tokenid::new(rank));
                index += length;
                continue;
            }
            let character = text[index..]
                .chars()
                .next()
                .ok_or(tokenizererror::invalidutf8)?;
            segment.push(character);
            index += character.len_utf8();
        }
        self.encode_segment(&segment, &mut out)?;
        Ok(out)
    }

    /// Decodes token ids to text.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::unknowntokenid`] for an absent id and
    /// [`tokenizererror::invalidutf8`] when the bytes do not form UTF-8.
    pub fn decode(&self, ids: &[tokenid]) -> Result<String, tokenizererror> {
        let mut bytes = Vec::new();
        for id in ids {
            let token = self
                .tokens
                .get(id.get() as usize)
                .ok_or(tokenizererror::unknowntokenid { id: id.get() })?;
            bytes.extend_from_slice(token);
        }
        String::from_utf8(bytes).map_err(|_| tokenizererror::invalidutf8)
    }

    /// Returns the longest special token `text` starts with, if any.
    fn matching_special(&self, text: &str) -> Option<(usize, u32)> {
        self.specials
            .iter()
            .find(|(content, _)| text.starts_with(content))
            .map(|(content, rank)| (content.len(), *rank))
    }

    /// Encodes one non-special segment through the regex and the byte merge.
    fn encode_segment(&self, segment: &str, out: &mut Vec<tokenid>) -> Result<(), tokenizererror> {
        if segment.is_empty() {
            return Ok(());
        }
        let mut last = 0;
        for found in self.regex.find_iter(segment) {
            let found = found.map_err(|error| tokenizererror::invalidregex {
                detail: error.to_string(),
            })?;
            if found.start() > last {
                self.encode_chunk(&segment[last..found.start()], out)?;
            }
            self.encode_chunk(&segment[found.start()..found.end()], out)?;
            last = found.end();
        }
        if last < segment.len() {
            self.encode_chunk(&segment[last..], out)?;
        }
        Ok(())
    }

    /// Encodes one chunk by greedily merging adjacent bytes.
    fn encode_chunk(&self, chunk: &str, out: &mut Vec<tokenid>) -> Result<(), tokenizererror> {
        let bytes = chunk.as_bytes();
        if bytes.is_empty() {
            return Ok(());
        }
        let mut tokens: Vec<Vec<u8>> = bytes.iter().map(|byte| vec![*byte]).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for index in 0..tokens.len().saturating_sub(1) {
                let mut merged = tokens[index].clone();
                merged.extend_from_slice(&tokens[index + 1]);
                if let Some(rank) = self.ranks.get(&merged)
                    && best.is_none_or(|(bestrank, _)| *rank < bestrank)
                {
                    best = Some((*rank, index));
                }
            }
            match best {
                Some((_, index)) => {
                    let right = tokens.remove(index + 1);
                    tokens[index].extend_from_slice(&right);
                }
                None => break,
            }
        }
        for token in &tokens {
            let rank = self
                .ranks
                .get(token)
                .ok_or_else(|| tokenizererror::invalidtiktoken {
                    detail: "merged token is absent from the vocabulary".to_string(),
                })?;
            out.push(tokenid::new(*rank));
        }
        Ok(())
    }
}

/// Decodes standard base64 (with `=` padding) to bytes.
fn decodebase64(input: &str) -> Result<Vec<u8>, ()> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buffer = 0u32;
    let mut bits = 0u32;
    for byte in input.bytes() {
        if byte == b'=' {
            break;
        }
        let value = ALPHABET
            .iter()
            .position(|alphabet| *alphabet == byte)
            .ok_or(())?;
        buffer = (buffer << 6) | value as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buffer >> bits) as u8);
        }
    }
    Ok(out)
}
