use std::collections::BTreeMap;

use crate::error::tokenizererror;

/// A byte-pair-encoding merge table.
///
/// A `bytepair` owns an ordered table of adjacent-symbol merges. Each entry is
/// the pair `(left, right)` with the index at which it appeared in the
/// `merges` array; lower indices take precedence, which is the HuggingFace
/// convention. The table is applied to a symbol sequence by repeatedly merging
/// the adjacent pair with the lowest rank until no rule applies.
///
/// The application is intentionally straightforward: worst-case time is
/// quadratic in the number of initial symbols and each candidate lookup
/// allocates a key pair. This is a correctness-first baseline; no performance
/// improvement is claimed and no benchmark is provided because byte-pair
/// encoding is not a computational operator under
/// `AGENTS.md` section 7.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct bytepair {
    ranks: BTreeMap<(String, String), usize>,
}

impl bytepair {
    /// Builds a merge table from raw `"left right"` merge strings.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::malformedmerge`] with the entry index when an
    /// entry does not contain exactly one separating space, or when either
    /// side of the split is empty.
    pub fn frommerges(merges: &[String]) -> Result<Self, tokenizererror> {
        let mut ranks = BTreeMap::new();
        for (index, merge) in merges.iter().enumerate() {
            let (left, right) = merge
                .split_once(' ')
                .ok_or(tokenizererror::malformedmerge { index })?;
            if left.is_empty() || right.is_empty() {
                return Err(tokenizererror::malformedmerge { index });
            }
            ranks
                .entry((left.to_string(), right.to_string()))
                .or_insert(index);
        }
        Ok(Self { ranks })
    }

    /// Returns the number of distinct merge rules.
    #[must_use]
    pub fn getcount(&self) -> usize {
        self.ranks.len()
    }

    /// Returns the rank of the pair `(left, right)`, if the rule exists.
    #[must_use]
    pub fn getrank(&self, left: &str, right: &str) -> Option<usize> {
        self.ranks
            .get(&(left.to_string(), right.to_string()))
            .copied()
    }

    /// Applies the merge rules to `symbols`, returning the merged sequence.
    ///
    /// The result is deterministic: at each step the adjacent pair with the
    /// lowest rank is merged, and ties are resolved toward the leftmost pair.
    #[must_use]
    pub fn apply(&self, symbols: &[String]) -> Vec<String> {
        let mut work: Vec<String> = symbols.to_vec();
        loop {
            let mut best: Option<(usize, usize)> = None;
            for (index, pair) in work.windows(2).enumerate() {
                if let Some(rank) = self.getrank(&pair[0], &pair[1])
                    && best.is_none_or(|(bestrank, _)| rank < bestrank)
                {
                    best = Some((rank, index));
                }
            }
            match best {
                Some((_, index)) => {
                    let merged = format!("{}{}", work[index], work[index + 1]);
                    work[index] = merged;
                    work.remove(index + 1);
                }
                None => break,
            }
        }
        work
    }
}
