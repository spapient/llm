use std::cmp::Ordering;
use std::collections::BinaryHeap;

use lib::tokenid;

/// A merge candidate: the rank of the pair starting at `left` and that pair's
/// left-symbol position. The reverse ordering pops the smallest rank first,
/// then the leftmost position.
#[derive(PartialEq, Eq)]
pub(crate) struct mergestep {
    pub(crate) rank: usize,
    pub(crate) left: usize,
}

impl Ord for mergestep {
    fn cmp(&self, other: &Self) -> Ordering {
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

/// The linked-list arrays, the symbol-id buffer, and the candidate heap shared
/// by the byte-pair merge loops. Grouped so a caller can hand out disjoint
/// borrows to the run accumulator and these buffers at once.
#[derive(Default)]
pub(crate) struct mergework {
    pub(crate) prev: Vec<Option<usize>>,
    pub(crate) next: Vec<Option<usize>>,
    pub(crate) alive: Vec<bool>,
    pub(crate) ids: Vec<u32>,
    pub(crate) heap: BinaryHeap<mergestep>,
}

/// Reusable working buffers for tokenizer encoding. Owned by the caller so
/// repeated encodes reuse the merge-loop buffers, the run accumulator, and the
/// character-offset table instead of reallocating per call. Buffers only grow
/// to the largest input seen and are never shrunk.
#[derive(Default)]
pub struct mergescratch {
    pub(crate) merge: mergework,
    pub(crate) run: Vec<tokenid>,
    pub(crate) charstart: Vec<usize>,
}

/// Resets `prev`/`next`/`alive` to the initial linked list of `count` symbols.
pub(crate) fn resetlinks(merge: &mut mergework, count: usize) {
    merge.prev.clear();
    merge.next.clear();
    merge.alive.clear();
    merge
        .prev
        .extend((0..count).map(|index| index.checked_sub(1)));
    merge
        .next
        .extend((0..count).map(|index| (index + 1 < count).then_some(index + 1)));
    merge.alive.resize(count, true);
}
