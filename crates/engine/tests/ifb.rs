//! `ifb` 的准入预算与分块打包夹具。

use super::*;

#[test]
fn generation_goes_before_ctx() {
    let candidates = [
        Candidate::ctx(1, 4),
        Candidate::generation(2),
        Candidate::ctx(3, 2),
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
#[test]
fn a_tight_budget_squeezes_only_ctx() {
    let candidates = [
        Candidate::generation(1),
        Candidate::generation(2),
        Candidate::ctx(3, 8),
        Candidate::ctx(4, 4),
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

#[test]
fn over_seqs_names_the_sequence_limit() {
    let candidates = [Candidate::generation(1), Candidate::generation(2)];
    assert_eq!(
        pack(Budget::new(1, 8), &candidates),
        Verdict::OverSeqs { seqs: 2, max: 1 }
    );
}

/// 生成行自己超预算：不是"这轮挤一挤"，重试也不会有用。
#[test]
fn generation_over_budget_is_not_a_squeeze() {
    let candidates = [Candidate::generation(1), Candidate::generation(2)];
    assert_eq!(
        pack(Budget::new(8, 1), &candidates),
        Verdict::OverTokens { tokens: 2, max: 1 }
    );
}

#[test]
fn no_candidates_is_idle() {
    assert_eq!(pack(Budget::unlimited(), &[]), Verdict::Idle);
}

/// 除末块外整页对齐；末块不补齐，省下的预算留给下一轮。
#[test]
fn ctx_chunks_align_to_whole_pages_except_the_tail() {
    assert_eq!(chunk(1000, 16, 100), 96, "100 行里只有 96 是整页");
    assert_eq!(chunk(100, 16, 100), 100, "末块不必对齐");
    assert_eq!(chunk(100, 16, 0), 0, "没有预算就不做上下文");
    assert_eq!(chunk(100, 16, 8), 0, "不足一整页也不做");
    assert_eq!(chunk(0, 16, 8), 0);
    assert_eq!(chunk(100, 0, 8), 0, "页宽为零算不出对齐");
}

#[test]
fn a_zero_budget_is_never_read_as_unlimited() {
    assert_eq!(Budget::new(0, 0).seqs(), 0);
    assert_eq!(
        pack(Budget::new(0, 0), &[Candidate::generation(1)]),
        Verdict::OverSeqs { seqs: 1, max: 0 }
    );
}
