use super::*;

/// 一行里只有第 `idx` 个位置是高分：argmax 可读。
fn row(idx: usize, width: usize) -> Vec<f32> {
    let mut logits = vec![0.0f32; width];
    logits[idx] = 9.0;
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

/// 草稿器只认"之前出现过同一段后缀"：历史不够长、或后缀只出现这一次，都不猜。
///
/// 空草稿不是失败——它让这一步退化成常规前向，比拿最后两个 token 硬抄一遍诚实。
#[test]
fn a_drafter_without_a_previous_occurrence_drafts_nothing() {
    let drafter = Drafter::new(2, 3).expect("草稿器");
    assert!(drafter.draft(&[]).is_empty());
    assert!(drafter.draft(&[7]).is_empty(), "历史不足 order");
    assert!(drafter.draft(&[1, 2]).is_empty(), "后缀只出现这一次");
}

/// 续写取**最近一次**出现的后面那几个 token：更早的一次不算。
#[test]
fn a_drafter_continues_the_latest_matching_suffix() {
    let drafter = Drafter::new(2, 1).expect("草稿器");
    assert_eq!(
        drafter.draft(&[1, 2, 7, 1, 2, 8, 1, 2]),
        [8],
        "末尾是 [1,2]：最近一次之前那次 [1,2] 后面是 7，最近一次后面是 8"
    );
}

/// 草稿长度受 `max` 约束：一块上下文里的重复不会一次猜出一整段。
#[test]
fn a_draft_never_exceeds_the_ceiling() {
    let history: Vec<u32> = (0..12).map(|at| at % 3).collect();
    for max in 1..=5 {
        let drafter = Drafter::new(2, max).expect("草稿器");
        assert!(drafter.draft(&history).len() <= max);
    }
}

/// 全接受时这一步走出的就是整段草稿；部分接受时只留前缀加补位，后面的草稿作废。
#[test]
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
#[test]
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
#[test]
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
#[test]
fn a_degenerate_drafter_is_refused() {
    assert_eq!(Drafter::new(1, 4).unwrap_err().code(), Code::Invalid);
    assert_eq!(Drafter::new(2, 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Drafter::new(3, 2).expect("合法").order(), 3);
}
