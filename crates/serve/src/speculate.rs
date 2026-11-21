//! 投机解码的接受判定：草稿一次猜 `K` 个 token，目标模型按同一批位置出分布，取最长一致前缀。
//!
//! 口径只有一处：目标的 argmax 走 [`crate::generate`] 的 `best`（等值取编号最小），与 [`Greedy`]
//! 同源。因此"草稿的 token 与目标的一致"在两边是同一个判据，不会出现草稿按一种 argmax 猜、
//! 校验按另一种判的错配。
//!
//! 贪心目标下这条规则**严格无损**：被接受的每个 token 都是目标模型自己会给出的那一个；第一个
//! 不一致的位置由目标模型补一个，然后从那里继续——于是接受数只影响"这一步走了多远"，不影响
//! "走出来的序列是不是目标模型的序列"。
//!
//! 这一层只判"接受几个"，不写 KV：接受几个由分布决定，写几行由调用方按接受数写。两件事分开，
//! "接受了 3 个却写了 4 行"这类错就不会从这一层漏出去。
//!
//! [`Greedy`]: crate::Greedy

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
    ///
    /// 取 `accepted` 个而不是"草稿里等于目标的那几个"：接受判定是**前缀**语义，被接受的一定是
    /// 从头连续的一段，中间一旦断开，后面的草稿即使碰巧对上也不再采信。
    #[must_use]
    pub fn produced(&self, draft: &[u32]) -> Vec<u32> {
        let mut out: Vec<u32> = draft.iter().take(self.accepted).copied().collect();
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
///
/// 只看已经算出来的历史：草稿要么来自"之前出现过同一段后缀"，要么干脆不给（空草稿）。于是它
/// 不会把还没发生的 token 当成已知——与 [`nn::slot_of`]、`nn::Emitter` 是同一句判据。
///
/// 它不承诺"猜得准"：接受几个由 [`accept`] 按目标模型的 argmax 判，被拒的那一段一律作废。
/// 草稿器的质量只影响"这一步走多远"，不影响"走出来的是不是目标模型自己的序列"。
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
    ///
    /// 当前位置自己不算"上一次"：只在 `[0, len − order)` 里找，否则永远只匹配到自己。
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
#[path = "../tests/speculate_tests.rs"]
mod tests;
