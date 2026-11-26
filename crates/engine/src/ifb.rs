//! IFB 一次迭代的可行域：`B ≤ B_max`、`Σ T_i ≤ T_max`（第三条 `N_need ≤ N_free` 由页账给）。

/// 一次迭代的两条预算；`usize::MAX` 表示该维不限。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Budget {
    seqs: usize,
    tokens: usize,
}

impl Budget {
    /// 两条上限都显式给出。
    #[must_use]
    pub const fn new(seqs: usize, tokens: usize) -> Self {
        Self { seqs, tokens }
    }

    /// 不限：桩、测试与"先跑通再说容量"的场景用得上。
    #[must_use]
    pub const fn unlimited() -> Self {
        Self::new(usize::MAX, usize::MAX)
    }

    /// `B_max`：本轮最多几条序列。
    #[must_use]
    pub const fn seqs(&self) -> usize {
        self.seqs
    }

    /// `T_max`：本轮最多几行 token。
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }
}

/// 本轮相位：生成只算一行，上下文按块推进。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    /// 解码：一条序列一行。
    Generation,
    /// 提示段或前缀补齐：一条序列可占多行。
    Prefill,
}

/// 一条候选：本轮要算几行、属于哪个相位。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Candidate {
    id: u64,
    phase: Phase,
    rows: usize,
}

impl Candidate {
    /// 一条生成行：解码一轮就是一行。
    #[must_use]
    pub const fn generation(id: u64) -> Self {
        Self {
            id,
            phase: Phase::Generation,
            rows: 1,
        }
    }

    /// 一条提示段的块：`rows` 行由 [`chunk`] 给出，不是调用方拍脑袋的数。
    #[must_use]
    pub const fn prefill(id: u64, rows: usize) -> Self {
        Self {
            id,
            phase: Phase::Prefill,
            rows,
        }
    }

    /// 序列号。
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// 相位。
    #[must_use]
    pub const fn phase(&self) -> Phase {
        self.phase
    }

    /// 本轮行数。
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }
}

/// 排布结论：装进本轮，或哪条约束先撞上。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// 这些行按序进本轮，总行数 `tokens`。
    Packed {
        /// 进本轮的行，生成在前、上下文在后。
        rows: Vec<Candidate>,
        /// 总行数。
        tokens: usize,
    },
    /// 候选序列数超过 `B_max`。
    OverSeqs {
        /// 候选序列数。
        seqs: usize,
        /// `B_max`。
        max: usize,
    },
    /// 生成行自己就超过 `T_max`。
    OverTokens {
        /// 生成行数。
        tokens: usize,
        /// `T_max`。
        max: usize,
    },
    /// 没有候选：本轮空转。
    Idle,
}

/// 按 `budget` 排这一轮的行：生成优先，上下文按输入序补到预算为止。
#[must_use]
pub fn pack(budget: Budget, candidates: &[Candidate]) -> Verdict {
    if candidates.is_empty() {
        return Verdict::Idle;
    }
    if candidates.len() > budget.seqs {
        return Verdict::OverSeqs {
            seqs: candidates.len(),
            max: budget.seqs,
        };
    }
    let mut ordered = candidates.to_vec();
    // 生成优先：`Phase` 的 `Ord` 就是这条次序，稳定排序保留同相位的输入序。
    ordered.sort_by_key(|candidate| candidate.phase);
    let generation = generation_rows(&ordered);
    if generation > budget.tokens {
        return Verdict::OverTokens {
            tokens: generation,
            max: budget.tokens,
        };
    }
    let (rows, tokens) = fill(ordered, budget.tokens);
    Verdict::Packed { rows, tokens }
}

/// 生成行的合计：只有它们进 `OverTokens` 判定——上下文行超预算能从下一轮补，生成行不能。
fn generation_rows(ordered: &[Candidate]) -> usize {
    ordered
        .iter()
        .filter(|candidate| candidate.phase == Phase::Generation)
        .map(Candidate::rows)
        .sum()
}

/// 按已排好的次序装箱：整行进或不进，不切割、不留 padding；返回选中的行与实占 token。
fn fill(ordered: Vec<Candidate>, tokens_max: usize) -> (Vec<Candidate>, usize) {
    let mut rows = Vec::with_capacity(ordered.len());
    let mut tokens = 0;
    for candidate in ordered {
        let next = tokens + candidate.rows;
        if next <= tokens_max {
            tokens = next;
            rows.push(candidate);
        }
    }
    (rows, tokens)
}

/// 本轮上下文能推进几行：除末块外 `C ≡ 0 (mod page_size)`。
#[must_use]
pub fn chunk(remaining: usize, page_size: usize, budget: usize) -> usize {
    if remaining == 0 || page_size == 0 || budget == 0 {
        return 0;
    }
    if remaining <= budget {
        // 末块：算得完就整段算完，不必补齐到整页。
        return remaining;
    }
    (budget / page_size) * page_size
}

#[cfg(test)]
mod tests {
    //! `ifb` 的准入预算与分块打包夹具。

    use super::*;

    testkit::test_cases! {
    fn generation_goes_before_ctx() {
        let candidates = [
            Candidate::prefill(1, 4),
            Candidate::generation(2),
            Candidate::prefill(3, 2),
        ];
        let Verdict::Packed { rows, tokens } = pack(Budget::unlimited(), &candidates) else {
            panic!("不限预算必须装得下");
        };
        assert_eq!(tokens, 7);
        assert_eq!(
            rows.iter().map(Candidate::id).collect::<Vec<_>>(),
            [2, 1, 3],
            "生成在前，上下文保持输入序"
        );
    }

    /// 挤压时只挤上下文：生成行一条都不许丢，上下文整条推到下一轮。
    fn a_tight_budget_squeezes_only_ctx() {
        let candidates = [
            Candidate::generation(1),
            Candidate::generation(2),
            Candidate::prefill(3, 8),
            Candidate::prefill(4, 4),
        ];
        let Verdict::Packed { rows, tokens } = pack(Budget::new(4, 6), &candidates) else {
            panic!("应当能装下");
        };
        assert_eq!(tokens, 6);
        assert_eq!(
            rows.iter().map(Candidate::id).collect::<Vec<_>>(),
            [1, 2, 4],
            "8 行的上下文挤不进 4 行余量，整条推到下一轮"
        );
    }

    fn over_seqs_names_the_sequence_limit() {
        let candidates = [Candidate::generation(1), Candidate::generation(2)];
        assert_eq!(
            pack(Budget::new(1, 8), &candidates),
            Verdict::OverSeqs { seqs: 2, max: 1 }
        );
    }

    /// 生成行自己超预算：不是"这轮挤一挤"，重试也不会有用。
    fn generation_over_budget_is_not_a_squeeze() {
        let candidates = [Candidate::generation(1), Candidate::generation(2)];
        assert_eq!(
            pack(Budget::new(8, 1), &candidates),
            Verdict::OverTokens { tokens: 2, max: 1 }
        );
    }

    fn no_candidates_is_idle() {
        assert_eq!(pack(Budget::unlimited(), &[]), Verdict::Idle);
    }

    /// 除末块外整页对齐；末块不补齐，省下的预算留给下一轮。
    fn ctx_chunks_align_to_whole_pages_except_the_tail() {
        assert_eq!(chunk(1000, 16, 100), 96, "100 行里只有 96 是整页");
        assert_eq!(chunk(100, 16, 100), 100, "末块不必对齐");
        assert_eq!(chunk(100, 16, 0), 0, "没有预算就不做上下文");
        assert_eq!(chunk(100, 16, 8), 0, "不足一整页也不做");
        assert_eq!(chunk(0, 16, 8), 0);
        assert_eq!(chunk(100, 0, 8), 0, "页宽为零算不出对齐");
    }

    fn a_zero_budget_is_never_read_as_unlimited() {
        assert_eq!(Budget::new(0, 0).seqs(), 0);
        assert_eq!(
            pack(Budget::new(0, 0), &[Candidate::generation(1)]),
            Verdict::OverSeqs { seqs: 1, max: 0 }
        );
    }
    }
}
