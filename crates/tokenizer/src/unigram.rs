//! SentencePiece Unigram model: a `(piece, score)` vocabulary encoded with a
//! Viterbi lattice over each pre-token (lowest cumulative score wins).
//!
//! The HF `tokenizer.json` form is `"model": {"type": "Unigram", "vocab":
//! [["piece", score], ...], "unk_id": N}`. The vocabulary also feeds the
//! tokenizer's `vocabulary` table so the shared decoder can resolve contents.

use std::collections::HashMap;

use lib::tokenid;

/// A Unigram model.
#[derive(Clone, Debug)]
pub struct unigrammodel {
    /// Piece → `(id, score, charlen)`. `charlen` is the piece length in
    /// Unicode scalars, used to bound the Viterbi prefix probes.
    ranks: HashMap<String, (tokenid, f32, usize)>,
    /// Unk token id and score (from the unk piece when present).
    unk: tokenid,
    unkscore: f32,
    /// Longest piece length in scalars.
    maxcharlen: usize,
}

/// Builds the piece table: `[piece, score]` arrays become `(String, f32)`
/// entries in id order.
pub fn parseunigramvocab(
    value: &serde_json::Value,
) -> Result<Vec<(String, f32)>, crate::tokenizererror> {
    use crate::tokenizererror;
    let array = value.as_array().ok_or(tokenizererror::unexpectedtype {
        field: "model.vocab",
        expected: "array",
    })?;
    let mut out = Vec::with_capacity(array.len());
    for entry in array.iter() {
        let pair = entry.as_array().ok_or(tokenizererror::unexpectedtype {
            field: "model.vocab entry",
            expected: "array",
        })?;
        if pair.len() != 2 {
            return Err(tokenizererror::unexpectedtype {
                field: "model.vocab entry",
                expected: "[piece, score]",
            });
        }
        let piece = pair[0].as_str().ok_or(tokenizererror::unexpectedtype {
            field: "model.vocab entry piece",
            expected: "string",
        })?;
        let score = pair[1].as_f64().ok_or(tokenizererror::unexpectedtype {
            field: "model.vocab entry score",
            expected: "number",
        })?;
        out.push((piece.to_string(), score as f32));
    }
    Ok(out)
}

impl unigrammodel {
    /// Builds a model from the `[piece, score]` vocabulary array and `unk_id`.
    /// The caller has already validated the array shape.
    #[must_use]
    pub fn fromvocab(vocab: &[(String, f32)], unk_id: u32) -> Self {
        let mut ranks = HashMap::with_capacity(vocab.len());
        let mut maxcharlen = 0usize;
        for (index, (piece, score)) in vocab.iter().enumerate() {
            let id = tokenid::new(index as u32);
            let charlen = piece.chars().count();
            maxcharlen = maxcharlen.max(charlen);
            ranks.insert(piece.clone(), (id, *score, charlen));
        }
        let unk = tokenid::new(unk_id);
        let unkscore = vocab
            .get(unk_id as usize)
            .map(|(_, score)| *score)
            .unwrap_or(0.0);
        Self {
            ranks,
            unk,
            unkscore,
            maxcharlen,
        }
    }

    /// Encodes `text` with the Viterbi lattice: `dp[i]` holds the best
    /// (lowest-score) path to scalar index `i`; each position probes every
    /// vocabulary piece that starts there. A scalar with no matching piece
    /// falls back to the unk token.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<tokenid> {
        if text.is_empty() {
            return Vec::new();
        }
        eprintln!("DBG unigram.encode input: {text:?} ranks={}", self.ranks.len());
        let charbyte: Vec<usize> = text
            .char_indices()
            .map(|(byte, _)| byte)
            .chain(std::iter::once(text.len()))
            .collect();
        let scalars = text.chars().count();
        let mut dp: Vec<Option<(f32, usize, tokenid)>> = vec![None; scalars + 1];
        dp[0] = Some((0.0, usize::MAX, self.unk));
        let maxlen = self.maxcharlen.min(scalars);
        for index in 0..scalars {
            let Some((score, _, _)) = dp[index] else {
                continue;
            };
            let mut matched = false;
            let available = maxlen.min(scalars - index);
            for length in 1..=available {
                let end = charbyte[index + length];
                let start = charbyte[index];
                if let Some((id, piece_score, _)) =
                    self.ranks.get(&text[start..end])
                {
                    let candidate = score + piece_score;
                    let current = &mut dp[index + length];
                    if current.is_none_or(|(best, _, _)| candidate > best) {
                        *current = Some((candidate, index, *id));
                    }
                    matched = true;
                }
            }
            if !matched {
                let candidate = score + self.unkscore;
                let current = &mut dp[index + 1];
                if current.is_none_or(|(best, _, _)| candidate > best) {
                    *current = Some((candidate, index, self.unk));
                }
            }
        }
        let mut out = Vec::new();
        let mut pos = scalars;
        while let Some((_, prev, id)) = dp[pos] {
            if prev == usize::MAX {
                break; // dp[0] is the virtual start, not a token
            }
            out.push(id);
            pos = prev;
        }
        out.reverse();
        out
    }
}
