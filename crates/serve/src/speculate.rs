//! 投机解码的接受判定：草稿一次猜 `K` 个 token，目标模型按同一批位置出分布，取最长一致前缀。

use diagnostic::{Code, Error, Result};

/// 一次投机的结论。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verdict {
    /// 最长一致前缀的长度：等于草稿长度即全部接受。
    pub accepted: usize,
    /// 第一个不一致位置上**目标模型**要的 token；全接受时为 `None`。
    pub fallback: Option<u32>,
}

impl Verdict {
    /// 是否整段草稿都被接受。
    #[must_use]
    pub const fn all(&self) -> bool {
        self.fallback.is_none()
    }

    /// 本次投机真正前进了几格：接受的草稿 token，加上被拒位置补的那一个。
    #[must_use]
    pub const fn advanced(&self) -> usize {
        if self.fallback.is_some() {
            self.accepted + 1
        } else {
            self.accepted
        }
    }

    /// 这一步真正走出来的 token：被接受的草稿前缀 + 第一个不一致处目标模型的补位。
    #[must_use]
    pub fn produced(&self, draft: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.accepted + usize::from(self.fallback.is_some()));
        out.extend(draft.iter().take(self.accepted).copied());
        out.extend(self.fallback);
        out
    }
}

/// 一次投机：`draft` 是草稿猜的 token，`rows[i]` 是目标模型在第 `i` 个草稿位置的分布。
///
/// 要求 `rows.len() == draft.len()`：位置 `i` 的分布决定 `draft[i]` 接不接受，多一行少一行都是
/// 调用方算错了偏移，不是"可以忽略的差"。
///
/// # Errors
///
/// 草稿为空（投机至少猜一个）、行数与草稿不一致、或某一行 logits 不可用（空 / NaN / 无穷）。
pub fn accept(draft: &[u32], rows: &[Vec<f32>]) -> Result<Verdict> {
    agree(draft, rows)?;
    for (position, (&guess, row)) in draft.iter().zip(rows).enumerate() {
        let target = crate::generate::best(row)?;
        if target != guess {
            return Ok(Verdict {
                accepted: position,
                fallback: Some(target),
            });
        }
    }
    Ok(Verdict {
        accepted: draft.len(),
        fallback: None,
    })
}

/// 提示查找草稿器：拿历史里**最近一次**的相同后缀续写，不训草稿模型、也不引入第二个 argmax。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Drafter {
    order: usize,
    max: usize,
}

impl Drafter {
    /// `order` 是匹配后缀的长度（≥ 2：1-gram 只是把上一个 token 抄一遍），`max` 是最多猜几个。
    ///
    /// # Errors
    ///
    /// `order < 2` 或 `max == 0`——后者会让"投机"退化成必然空转。
    pub fn new(order: usize, max: usize) -> Result<Self> {
        if order < 2 {
            return Err(bad(format!("匹配后缀长度必须 ≥ 2（order = {order}）")));
        }
        if max == 0 {
            return Err(bad("草稿上限必须为正：0 个草稿的投机是必然空转".to_owned()));
        }
        Ok(Self { order, max })
    }

    /// 匹配后缀长度。
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
    }

    /// 草稿上限。
    #[must_use]
    pub const fn max(&self) -> usize {
        self.max
    }

    /// 按历史出草稿：最近一次相同后缀之后的那一段，最多 [`max`](Self::max) 个 token。
    #[must_use]
    pub fn draft(&self, history: &[u32]) -> Vec<u32> {
        let Some(tail) = history.len().checked_sub(self.order) else {
            return Vec::new();
        };
        let pattern = &history[tail..];
        let Some(start) = (0..tail)
            .rev()
            .find(|&at| history[at..at + self.order] == *pattern)
        else {
            return Vec::new();
        };
        let from = start + self.order;
        let to = history.len().min(from + self.max);
        history[from..to].to_vec()
    }

    /// 一步投机：按历史出草稿，按目标分布在草稿位置上判定，返回这一步真正走出来的 token。
    ///
    /// 行数与草稿保持一致由调用方负责（行是"按这份草稿算出来的"）。草稿为空时要求没有行，
    /// 返回空——这一层只回答"投机带来了什么"，常规前向仍归调用方。
    ///
    /// # Errors
    ///
    /// 行数与草稿不一致，或 [`accept`] 的失败条件。
    pub fn walk(&self, history: &[u32], rows: &[Vec<f32>]) -> Result<Vec<u32>> {
        let draft = self.draft(history);
        if rows.len() != draft.len() {
            return Err(bad(format!(
                "校验行数 {} 与草稿长度 {} 不一致：行要按这份草稿算",
                rows.len(),
                draft.len()
            )));
        }
        if draft.is_empty() {
            return Ok(Vec::new());
        }
        Ok(accept(&draft, rows)?.produced(&draft))
    }
}

/// 草稿与校验行必须一一对上：位置错了就判不了「第几个被接受」。
fn agree(draft: &[u32], rows: &[Vec<f32>]) -> Result<()> {
    if draft.is_empty() {
        return Err(bad("草稿为空：每次投机至少猜一个 token".to_owned()));
    }
    if rows.len() != draft.len() {
        return Err(bad(format!(
            "校验行数 {} 与草稿长度 {} 不一致：位置对不上就判不了接受",
            rows.len(),
            draft.len()
        )));
    }
    Ok(())
}

fn bad(message: String) -> Error {
    Error::new(Code::Invalid, message).with("serve::speculate")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一行里只有第 `idx` 个位置是高分：argmax 可读。
    fn row(idx: usize, width: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; width];
        logits[idx] = 9.0;
        logits
    }

    testkit::test_cases! {
    fn a_fully_accepted_draft_advances_by_its_length() {
        let verdict = accept(&[1, 2, 3], &[row(1, 4), row(2, 4), row(3, 4)]).expect("判定");
        assert_eq!(
            verdict,
            Verdict {
                accepted: 3,
                fallback: None
            }
        );
        assert!(verdict.all());
        assert_eq!(verdict.advanced(), 3);
    }

    /// 第一个就猜错：接受 0 个，用目标模型的 token 补一格——这一格照样是前进。
    fn a_first_mismatch_falls_back_to_the_target_token() {
        let verdict = accept(&[0, 2], &[row(3, 4), row(2, 4)]).expect("判定");
        assert_eq!(
            verdict,
            Verdict {
                accepted: 0,
                fallback: Some(3)
            }
        );
        assert!(!verdict.all());
        assert_eq!(verdict.advanced(), 1);
    }

    /// 中途猜错：只有一致前缀被接受，后面的草稿一律作废（它们是在错的上下文里猜的）。
    fn a_middle_mismatch_keeps_only_the_agreeing_prefix() {
        let verdict = accept(&[1, 0, 3], &[row(1, 4), row(2, 4), row(3, 4)]).expect("判定");
        assert_eq!(
            verdict,
            Verdict {
                accepted: 1,
                fallback: Some(2)
            }
        );
    }

    fn a_length_mismatch_is_an_error() {
        assert_eq!(
            accept(&[1, 2], &[row(1, 4)]).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(
            accept(&[], &[]).unwrap_err().code(),
            Code::Invalid,
            "空草稿不成立，不是「全部接受」"
        );
    }

    fn unusable_logits_are_errors() {
        assert!(accept(&[1], &[Vec::new()]).is_err());
        assert!(accept(&[1], &[vec![f32::NAN, 1.0]]).is_err());
    }

    /// 草稿器只认"之前出现过同一段后缀"：历史不够长、或后缀只出现这一次，都不猜。
    ///
    /// 空草稿不是失败——它让这一步退化成常规前向，比拿最后两个 token 硬抄一遍诚实。
    fn a_drafter_without_a_previous_occurrence_drafts_nothing() {
        let drafter = Drafter::new(2, 3).expect("草稿器");
        assert!(drafter.draft(&[]).is_empty());
        assert!(drafter.draft(&[7]).is_empty(), "历史不足 order");
        assert!(drafter.draft(&[1, 2]).is_empty(), "后缀只出现这一次");
    }

    /// 续写取**最近一次**出现的后面那几个 token：更早的一次不算。
    fn a_drafter_continues_the_latest_matching_suffix() {
        let drafter = Drafter::new(2, 1).expect("草稿器");
        assert_eq!(
            drafter.draft(&[1, 2, 7, 1, 2, 8, 1, 2]),
            [8],
            "末尾是 [1,2]：最近一次之前那次 [1,2] 后面是 7，最近一次后面是 8"
        );
    }

    /// 草稿长度受 `max` 约束：一块上下文里的重复不会一次猜出一整段。
    fn a_draft_never_exceeds_the_ceiling() {
        let history: Vec<u32> = (0..12).map(|at| at % 3).collect();
        for max in 1..=5 {
            let drafter = Drafter::new(2, max).expect("草稿器");
            assert!(drafter.draft(&history).len() <= max);
        }
    }

    /// 全接受时这一步走出的就是整段草稿；部分接受时只留前缀加补位，后面的草稿作废。
    fn what_was_accepted_is_what_the_step_produces() {
        let all = accept(&[1, 2], &[row(1, 4), row(2, 4)]).expect("判定");
        assert_eq!(all.produced(&[1, 2]), [1, 2]);

        let partial = accept(&[1, 0, 3], &[row(1, 4), row(2, 4), row(3, 4)]).expect("判定");
        assert_eq!(
            partial.produced(&[1, 0, 3]),
            [1, 2],
            "第 2 个位置目标要 2：后面的草稿是在错的上下文里猜的"
        );
    }

    /// 一步投机串起来：草稿由历史给出，行必须按这份草稿算；对不上是调用方的偏移错了。
    fn a_speculative_step_walks_the_drafted_suffix() {
        let drafter = Drafter::new(2, 2).expect("草稿器");
        let history = [0, 1, 2, 0, 1];
        assert_eq!(
            drafter.draft(&history),
            [2, 0],
            "末尾是 [0,1]：上一次出现之后是 [2,0]"
        );

        let produced = drafter
            .walk(&history, &[row(2, 4), row(0, 4)])
            .expect("全接受");
        assert_eq!(produced, [2, 0]);
        assert_eq!(produced.len(), 2, "一步走出两格");
    }

    /// 行数与草稿不一致时拒绝：位置错了就判不了"第几个被接受"。
    fn a_step_whose_rows_do_not_match_the_draft_is_refused() {
        let drafter = Drafter::new(2, 3).expect("草稿器");
        let history = [0, 1, 2, 0, 1];

        assert_eq!(
            drafter.walk(&history, &[row(2, 4)]).unwrap_err().code(),
            Code::Invalid,
            "草稿两位、只给了 1 行"
        );
        assert!(
            drafter.walk(&[1, 2], &[row(1, 4)]).is_err(),
            "没有草稿就不该有校验行"
        );
    }

    /// 退化参数在构造期拒绝：1-gram 等于抄上一个 token，零上限等于必然空转。
    fn a_degenerate_drafter_is_refused() {
        assert_eq!(Drafter::new(1, 4).unwrap_err().code(), Code::Invalid);
        assert_eq!(Drafter::new(2, 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(Drafter::new(3, 2).expect("合法").order(), 3);
    }
    }
}
