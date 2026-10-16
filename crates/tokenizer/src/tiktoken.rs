//! The OpenAI tiktoken `.tiktoken` format (GPT-4o `o200k_base` and friends).
//!
//! Unlike a HuggingFace `tokenizer.json`, a `.tiktoken` file holds only the
//! mergeable vocabulary — one `<base64 token bytes> <rank>` per line — with no
//! merges table, no pre-tokenizer, and no decoder. Encoding runs the fixed
//! hand-written o200k splitter to split, then greedily merges adjacent bytes
//! whose concatenation is a vocabulary token (lowest resulting rank first).
//! Special tokens are supplied by the caller because they are not in the file.

use std::collections::{BinaryHeap, HashMap};

use lib::tokenid;

use crate::error::tokenizererror;
use crate::pretoken::o200ksplit;
use crate::scratch::{mergescratch, mergestep, resetlinks};

/// A tiktoken-format encoder.
#[derive(Clone, Debug)]
pub struct tiktokenizer {
    /// Token bytes → rank (mergeable vocabulary).
    ranks: HashMap<Vec<u8>, u32>,
    /// Rank → token bytes (mergeable and special).
    tokens: Vec<Vec<u8>>,
    /// Special token content → rank, sorted longest first.
    specials: Vec<(String, u32)>,
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
        Ok(Self {
            ranks,
            tokens,
            specials: specials_sorted,
        })
    }

    /// Encodes `text`, recognizing the supplied special-token literals.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::invalidregex`] for a runtime regex failure.
    pub fn encode(&self, text: &str) -> Result<Vec<tokenid>, tokenizererror> {
        let mut scratch = mergescratch::default();
        let mut out = Vec::new();
        self.encode_into(text, &mut scratch, &mut out)?;
        Ok(out)
    }

    /// Encodes `text`, recognizing special-token literals, appending to `out`
    /// and reusing `scratch` for the merge-loop buffers.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::invalidregex`] for a runtime regex failure.
    pub fn encode_into(
        &self,
        text: &str,
        scratch: &mut mergescratch,
        out: &mut Vec<tokenid>,
    ) -> Result<(), tokenizererror> {
        let mut segment = String::new();
        let mut index = 0;
        while index < text.len() {
            if let Some((length, rank)) = self.matching_special(&text[index..]) {
                self.encode_segment(&segment, scratch, out)?;
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
        self.encode_segment(&segment, scratch, out)?;
        Ok(())
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

    /// Encodes one non-special segment through the hand-written splitter and
    /// the byte merge.
    fn encode_segment(
        &self,
        segment: &str,
        scratch: &mut mergescratch,
        out: &mut Vec<tokenid>,
    ) -> Result<(), tokenizererror> {
        for chunk in o200ksplit(segment) {
            self.encode_chunk(chunk, scratch, out)?;
        }
        Ok(())
    }

    /// Encodes one chunk by greedily merging adjacent bytes.
    ///
    /// The next merge is selected from a binary heap and symbols live in a
    /// linked list, so each merge touches only its two neighbours instead of
    /// rescanning the whole chunk. A live symbol at index `i` is the byte slice
    /// `bytes[i..next[i]]` (its start offset equals its index), so no per-symbol
    /// buffer is materialized and each candidate probes the vocabulary with a
    /// borrowed slice.
    fn encode_chunk(
        &self,
        chunk: &str,
        scratch: &mut mergescratch,
        out: &mut Vec<tokenid>,
    ) -> Result<(), tokenizererror> {
        let bytes = chunk.as_bytes();
        let count = bytes.len();
        if count == 0 {
            return Ok(());
        }
        if count == 1 {
            let rank = self
                .ranks
                .get(bytes)
                .ok_or_else(|| tokenizererror::invalidtiktoken {
                    detail: "single byte is absent from the vocabulary".to_string(),
                })?;
            out.push(tokenid::new(*rank));
            return Ok(());
        }

        let merge = &mut scratch.merge;
        resetlinks(merge, count);
        merge.heap.clear();
        for index in 0..count - 1 {
            self.queue_pair(bytes, &merge.next, &merge.alive, &mut merge.heap, index);
        }

        while let Some(step) = merge.heap.pop() {
            self.trymerge(
                bytes,
                &mut merge.prev,
                &mut merge.next,
                &mut merge.alive,
                &mut merge.heap,
                step,
            );
        }

        let mut cursor = 0usize;
        loop {
            let end = merge.next[cursor].unwrap_or(count);
            let rank = self.ranks.get(&bytes[cursor..end]).ok_or_else(|| {
                tokenizererror::invalidtiktoken {
                    detail: "merged token is absent from the vocabulary".to_string(),
                }
            })?;
            out.push(tokenid::new(*rank));
            match merge.next[cursor] {
                Some(following) => cursor = following,
                None => break,
            }
        }
        Ok(())
    }

    /// Applies one heap-selected merge when it is still live and its rank is
    /// still current; otherwise it is a stale candidate and is skipped.
    fn trymerge(
        &self,
        bytes: &[u8],
        prev: &mut [Option<usize>],
        next: &mut [Option<usize>],
        alive: &mut [bool],
        heap: &mut BinaryHeap<mergestep>,
        step: mergestep,
    ) {
        let Some(right) = next[step.left] else {
            return;
        };
        if !alive[step.left] || !alive[right] {
            return;
        }
        let end = next[right].unwrap_or(bytes.len());
        let Some(rank) = self.ranks.get(&bytes[step.left..end]) else {
            return;
        };
        if *rank as usize != step.rank {
            return;
        }
        alive[right] = false;
        let following = next[right];
        next[step.left] = following;
        if let Some(after) = following {
            prev[after] = Some(step.left);
        }
        if let Some(before) = prev[step.left] {
            self.queue_pair(bytes, next, alive, heap, before);
        }
        self.queue_pair(bytes, next, alive, heap, step.left);
    }

    /// Queues the mergeable pair starting at `left`, if its concatenation is a
    /// vocabulary token and both symbols are live.
    fn queue_pair(
        &self,
        bytes: &[u8],
        next: &[Option<usize>],
        alive: &[bool],
        heap: &mut BinaryHeap<mergestep>,
        left: usize,
    ) {
        if !alive[left] {
            return;
        }
        let Some(right) = next[left] else {
            return;
        };
        if !alive[right] {
            return;
        }
        let end = next[right].unwrap_or(bytes.len());
        if let Some(rank) = self.ranks.get(&bytes[left..end]) {
            heap.push(mergestep {
                rank: *rank as usize,
                left,
            });
        }
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
