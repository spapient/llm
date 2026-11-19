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
        self.accepted + usize::from(self.fallback.is_some())
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

fn bad(message: String) -> Error {
    Error::new(Code::Invalid, message).with("serve::speculate")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一行里只有第 `index` 个位置是高分：argmax 可读。
    fn row(index: usize, width: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; width];
        logits[index] = 9.0;
        logits
    }

    #[test]
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
    #[test]
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
    #[test]
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

    #[test]
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

    #[test]
    fn unusable_logits_are_errors() {
        assert!(accept(&[1], &[Vec::new()]).is_err());
        assert!(accept(&[1], &[vec![f32::NAN, 1.0]]).is_err());
    }
}
