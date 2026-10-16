use std::cell::OnceCell;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

use lib::tokenid;

use crate::error::tokenizererror;
use crate::scratch::{mergescratch, mergestep, mergework, resetlinks};
use crate::vocab::vocabulary;

/// A byte-pair-encoding merge table.
///
/// A `bytepair` owns an ordered table of adjacent-symbol merges. Each entry is
/// the pair `(left, right)` with the index at which it appeared in the
/// `merges` array; lower indices take precedence, which is the HuggingFace
/// convention. The table is applied to a symbol sequence by repeatedly merging
/// the adjacent pair with the lowest rank until no rule applies.
///
/// The string-keyed form (`ranks`) is built lazily on first use by
/// [`getrank`](bytepair::getrank)/[`apply`](bytepair::apply), because the
/// encoding path only needs the integer-keyed form (`byids`) built by
/// [`frommergesandvocab`](bytepair::frommergesandvocab).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct bytepair {
    merges: Vec<String>,
    ranks: OnceCell<BTreeMap<String, BTreeMap<String, usize>>>,
    byids: HashMap<u64, (usize, u32)>,
    count: usize,
}

/// Builds the string-keyed merge table from pre-validated merge strings.
fn ranksfrommerges(merges: &[String]) -> BTreeMap<String, BTreeMap<String, usize>> {
    let mut ranks: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for (index, merge) in merges.iter().enumerate() {
        if let Some((left, right)) = merge.split_once(' ') {
            ranks
                .entry(left.to_string())
                .or_default()
                .entry(right.to_string())
                .or_insert(index);
        }
    }
    ranks
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
        for (index, merge) in merges.iter().enumerate() {
            let (left, right) = merge
                .split_once(' ')
                .ok_or(tokenizererror::malformedmerge { index })?;
            if left.is_empty() || right.is_empty() {
                return Err(tokenizererror::malformedmerge { index });
            }
        }
        let count = merges
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>()
            .len();
        Ok(Self {
            merges: merges.to_vec(),
            ranks: OnceCell::new(),
            byids: HashMap::new(),
            count,
        })
    }

    /// Builds a merge table with both its string-keyed and integer-keyed forms.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::malformedmerge`] for the same malformed
    /// entries as [`frommerges`](bytepair::frommerges), and
    /// [`tokenizererror::mergenotinvocab`] when a merge's left, right, or
    /// concatenated result is absent from `vocab` and so has no id.
    pub fn frommergesandvocab(
        merges: &[String],
        vocab: &vocabulary,
    ) -> Result<Self, tokenizererror> {
        let mut byids: HashMap<u64, (usize, u32)> = HashMap::with_capacity(merges.len());
        for (index, merge) in merges.iter().enumerate() {
            let (left, right) = merge
                .split_once(' ')
                .ok_or(tokenizererror::malformedmerge { index })?;
            if left.is_empty() || right.is_empty() {
                return Err(tokenizererror::malformedmerge { index });
            }
            let leftid = vocab
                .getid(left)
                .ok_or_else(|| tokenizererror::mergenotinvocab {
                    symbol: left.to_string(),
                })?;
            let rightid = vocab
                .getid(right)
                .ok_or_else(|| tokenizererror::mergenotinvocab {
                    symbol: right.to_string(),
                })?;
            let mut merged = String::with_capacity(left.len() + right.len());
            merged.push_str(left);
            merged.push_str(right);
            let mergedid = vocab
                .getid(&merged)
                .ok_or(tokenizererror::mergenotinvocab { symbol: merged })?;
            let key = (u64::from(leftid.get()) << 32) | u64::from(rightid.get());
            byids.entry(key).or_insert((index, mergedid.get()));
        }
        let count = byids.len();
        Ok(Self {
            merges: merges.to_vec(),
            ranks: OnceCell::new(),
            byids,
            count,
        })
    }

    /// Returns the number of distinct merge rules.
    #[must_use]
    pub fn getcount(&self) -> usize {
        self.count
    }

    /// Returns the string-keyed merge table, building it lazily on first use.
    fn ranks(&self) -> &BTreeMap<String, BTreeMap<String, usize>> {
        self.ranks.get_or_init(|| ranksfrommerges(&self.merges))
    }

    /// Returns the rank of the pair `(left, right)`, if the rule exists.
    ///
    /// The lookup is allocation-free: both levels are probed by borrowed
    /// `str` keys.
    #[must_use]
    pub fn getrank(&self, left: &str, right: &str) -> Option<usize> {
        self.ranks()
            .get(left)
            .and_then(|inner| inner.get(right))
            .copied()
    }

    /// Applies the integer-keyed merge rules to a sequence of vocabulary ids,
    /// returning the merged id sequence.
    ///
    /// This is a convenience wrapper that allocates a fresh
    /// [`mergescratch`] and output buffer.
    #[must_use]
    pub fn applyids(&self, symbols: &[tokenid]) -> Vec<tokenid> {
        let mut scratch = mergescratch::default();
        let mut out = Vec::with_capacity(symbols.len());
        self.applyids_into(symbols, &mut scratch.merge, &mut out);
        out
    }

    /// Applies the integer-keyed merge rules to a sequence of vocabulary ids,
    /// appending the merged ids to `out` and reusing `merge` for the linked
    /// list, symbol buffer, and candidate heap instead of allocating them per
    /// call.
    ///
    /// Each merge step probes the packed `(left_id, right_id)` key, so the
    /// per-candidate cost is a constant-size hash lookup with no allocation.
    /// The result is identical to [`apply`](bytepair::apply) followed by an id
    /// resolution when every merge rule's symbols are in the vocabulary; a
    /// table built by [`frommerges`](bytepair::frommerges) has an empty id map,
    /// so the method then appends `symbols` unchanged.
    pub(crate) fn applyids_into(
        &self,
        symbols: &[tokenid],
        merge: &mut mergework,
        out: &mut Vec<tokenid>,
    ) {
        let count = symbols.len();
        if count < 2 {
            out.extend_from_slice(symbols);
            return;
        }

        resetlinks(merge, count);
        merge.ids.clear();
        merge.ids.extend(symbols.iter().map(|id| id.get()));
        merge.heap.clear();
        for index in 0..count - 1 {
            self.queueidcandidate(
                &merge.ids,
                &merge.next,
                &merge.alive,
                &mut merge.heap,
                index,
            );
        }

        while let Some(step) = merge.heap.pop() {
            self.tryidmerge(
                &mut merge.ids,
                &mut merge.prev,
                &mut merge.next,
                &mut merge.alive,
                &mut merge.heap,
                step,
            );
        }

        let mut cursor = 0usize;
        loop {
            out.push(tokenid::new(merge.ids[cursor]));
            match merge.next[cursor] {
                Some(following) => cursor = following,
                None => break,
            }
        }
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
            self.trymerge(&mut work, &mut prev, &mut next, &mut alive, &mut heap, step);
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

    /// Applies one heap-selected merge step when it is still live and its rank
    /// is still current; otherwise it is a stale candidate and is skipped.
    fn trymerge(
        &self,
        work: &mut [String],
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
        if self.getrank(&work[step.left], &work[right]) != Some(step.rank) {
            return;
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
            self.queuecandidate(&*work, &*next, &*alive, heap, before);
        }
        self.queuecandidate(&*work, &*next, &*alive, heap, step.left);
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

    /// Applies one heap-selected id merge when it is still live and its rank is
    /// still current; otherwise it is a stale candidate and is skipped.
    fn tryidmerge(
        &self,
        work: &mut [u32],
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
        let key = (u64::from(work[step.left]) << 32) | u64::from(work[right]);
        let Some((rank, merged)) = self.byids.get(&key) else {
            return;
        };
        if *rank != step.rank {
            return;
        }
        work[step.left] = *merged;
        alive[right] = false;
        let following = next[right];
        next[step.left] = following;
        if let Some(after) = following {
            prev[after] = Some(step.left);
        }
        if let Some(before) = prev[step.left] {
            self.queueidcandidate(work, next, alive, heap, before);
        }
        self.queueidcandidate(work, next, alive, heap, step.left);
    }

    /// Queues the mergeable id pair starting at `left`, if one exists and both
    /// of its symbols are still live.
    fn queueidcandidate(
        &self,
        work: &[u32],
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
        let key = (u64::from(work[left]) << 32) | u64::from(work[right]);
        if let Some((rank, _)) = self.byids.get(&key) {
            heap.push(mergestep { rank: *rank, left });
        }
    }
}
