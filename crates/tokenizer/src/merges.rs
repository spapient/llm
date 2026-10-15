use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use crate::error::tokenizererror;

/// A candidate byte-pair merge: the rank of the pair starting at `left` and
/// that pair's left-symbol position.
#[derive(PartialEq, Eq)]
struct mergestep {
    rank: usize,
    left: usize,
}

impl Ord for mergestep {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` pops the greatest element, so both keys are reversed to
        // pop the smallest rank first, then the leftmost position.
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for mergestep {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A byte-pair-encoding merge table.
///
/// A `bytepair` owns an ordered table of adjacent-symbol merges. Each entry is
/// the pair `(left, right)` with the index at which it appeared in the
/// `merges` array; lower indices take precedence, which is the HuggingFace
/// convention. The table is applied to a symbol sequence by repeatedly merging
/// the adjacent pair with the lowest rank until no rule applies.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct bytepair {
    ranks: BTreeMap<String, BTreeMap<String, usize>>,
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
        let mut ranks: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
        for (index, merge) in merges.iter().enumerate() {
            let (left, right) = merge
                .split_once(' ')
                .ok_or(tokenizererror::malformedmerge { index })?;
            if left.is_empty() || right.is_empty() {
                return Err(tokenizererror::malformedmerge { index });
            }
            ranks
                .entry(left.to_string())
                .or_default()
                .entry(right.to_string())
                .or_insert(index);
        }
        Ok(Self { ranks })
    }

    /// Returns the number of distinct merge rules.
    #[must_use]
    pub fn getcount(&self) -> usize {
        self.ranks.values().map(|inner| inner.len()).sum()
    }

    /// Returns the rank of the pair `(left, right)`, if the rule exists.
    ///
    /// The lookup is allocation-free: both levels are probed by borrowed
    /// `str` keys.
    #[must_use]
    pub fn getrank(&self, left: &str, right: &str) -> Option<usize> {
        self.ranks
            .get(left)
            .and_then(|inner| inner.get(right))
            .copied()
    }

    /// Applies the merge rules to `symbols`, returning the merged sequence.
    ///
    /// The result is deterministic: at each step the adjacent pair with the
    /// lowest rank is merged, and ties are resolved toward the leftmost pair.
    /// Symbols are held in a linked list and the next merge is selected from a
    /// binary heap, so each merge touches only its two neighbours instead of
    /// rescanning the whole sequence. The rank lookup itself is still a
    /// string-keyed `BTreeMap` probe, so adversarial inputs whose merged keys
    /// grow unboundedly add a total key-length term on top of the O(n log n)
    /// heap operations.
    #[must_use]
    pub fn apply(&self, symbols: &[String]) -> Vec<String> {
        let count = symbols.len();
        if count < 2 {
            return symbols.to_vec();
        }

        let mut work: Vec<String> = symbols.to_vec();
        let mut prev = Vec::with_capacity(count);
        let mut next = Vec::with_capacity(count);
        for index in 0..count {
            prev.push(index.checked_sub(1));
            next.push(if index + 1 < count {
                Some(index + 1)
            } else {
                None
            });
        }
        let mut alive = vec![true; count];

        let mut heap = BinaryHeap::new();
        for index in 0..count - 1 {
            self.queuecandidate(&work, &next, &alive, &mut heap, index);
        }

        while let Some(step) = heap.pop() {
            let Some(right) = next[step.left] else {
                continue;
            };
            if !alive[step.left] || !alive[right] {
                continue;
            }
            if self.getrank(&work[step.left], &work[right]) != Some(step.rank) {
                continue;
            }

            let absorbed = std::mem::take(&mut work[right]);
            work[step.left].push_str(&absorbed);
            alive[right] = false;
            let following = next[right];
            next[step.left] = following;
            if let Some(after) = following {
                prev[after] = Some(step.left);
            }
            if let Some(before) = prev[step.left] {
                self.queuecandidate(&work, &next, &alive, &mut heap, before);
            }
            self.queuecandidate(&work, &next, &alive, &mut heap, step.left);
        }

        let mut out = Vec::with_capacity(count);
        let mut cursor = 0usize;
        loop {
            out.push(std::mem::take(&mut work[cursor]));
            match next[cursor] {
                Some(following) => cursor = following,
                None => break,
            }
        }
        out
    }

    /// Queues the mergeable pair starting at `left`, if one exists and both of
    /// its symbols are still live.
    fn queuecandidate(
        &self,
        work: &[String],
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
        if let Some(rank) = self.getrank(&work[left], &work[right]) {
            heap.push(mergestep { rank, left });
        }
    }
}
