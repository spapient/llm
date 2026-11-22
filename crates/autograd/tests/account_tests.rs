use diagnostic::Code;

use super::*;
use crate::graph::Requirement;
use crate::tape::Tape;

/// 菱形图：两条路径的贡献都到齐，x 的梯度才算齐全。
#[test]
fn a_diamond_waits_for_every_contribution() {
    let (x, left, right, joined, _, schedule) = crate::testutil::diamond();
    let mut account = schedule.account();
    assert_eq!(account.pending(x), 2);
    let steps = schedule.steps();
    assert_eq!(account.settle(&steps[0]).expect("loss"), [joined]);
    assert_eq!(account.settle(&steps[1]).expect("joined"), [left, right]);
    assert!(account.settle(&steps[3]).expect("left").is_empty());
    assert!(!account.ready(x), "还有一条路径没回传");
    assert_eq!(account.settle(&steps[2]).expect("right"), [x]);
    assert!(account.finished());
}

/// 同一个变量在一步里出现两次，就要收两份贡献，且只报一次齐全。
#[test]
fn a_repeated_input_receives_one_contribution_per_position() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let doubled = tape.record("add", &[x, x]).expect("记录");
    let scaled = tape.record("mul", &[doubled, x]).expect("记录");
    let loss = tape.record("reduce_sum", &[scaled]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    let mut account = schedule.account();
    assert_eq!(account.pending(x), 3, "add 里两次 + mul 里一次");
    let steps = schedule.steps();
    assert_eq!(account.settle(&steps[0]).expect("loss"), [scaled]);
    assert_eq!(account.settle(&steps[1]).expect("scaled"), [doubled]);
    assert!(!account.ready(x), "x 还差两份");
    assert_eq!(
        account.settle(&steps[2]).expect("doubled"),
        [x],
        "两个位置属于同一步，只报一次齐全"
    );
    assert!(account.ready(x));
}

/// 乱序执行、重复结算、外来步骤都必须失败，而不是悄悄算错。
#[test]
fn settling_out_of_order_or_twice_is_rejected() {
    let (mut tape, x, _, _, schedule) = crate::testutil::linear();
    let steps = schedule.steps();
    let mut account = schedule.account();
    assert_eq!(
        account.settle(&steps[1]).expect_err("y 还没收满").code(),
        Code::Invalid
    );
    account.settle(&steps[0]).expect("loss");
    assert_eq!(
        account.settle(&steps[0]).expect_err("已结算").code(),
        Code::Invalid
    );

    let pruned = tape.record("relu", &[x]).expect("记录");
    let foreign = tape.node(pruned).expect("节点").clone();
    assert_eq!(
        account.settle(&foreign).expect_err("不在计划里").code(),
        Code::Invalid
    );
    assert_eq!(account.settled(), 1);
}

/// 账本按计划重建：结算进度清零，扇出重新数一遍。
#[test]
fn rebuilding_starts_the_ledger_over() {
    let (_, _, y, _, schedule) = crate::testutil::linear();
    let mut account = Account::new();
    account.rebuild(&schedule);
    assert_eq!(account.settled(), 0);
    assert_eq!(account.pending(y), 1);
    let step = &schedule.steps()[0];
    account.settle(step).expect("loss");
    assert_eq!(account.settled(), 1);
    account.rebuild(&schedule);
    assert_eq!(account.settled(), 0, "重建丢弃进度");
    assert_eq!(account.pending(y), 1, "扇出回到初始值");
    assert!(!account.finished());
}

/// `settle_into` 与 `settle` 结果一致，复用调用方的缓冲。
#[test]
fn settling_into_a_reused_buf_reports_the_same_ready_vars() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let y = tape.parameter();
    let joined = tape.record("add", &[x, y]).expect("记录");
    let loss = tape.record("reduce_sum", &[joined]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    let mut account = schedule.account();
    let mut ready = Vec::with_capacity(8);
    let capacity = ready.capacity();
    account
        .settle_into(&schedule.steps()[0], &mut ready)
        .expect("loss");
    assert_eq!(ready, [joined]);
    account
        .settle_into(&schedule.steps()[1], &mut ready)
        .expect("joined");
    assert_eq!(ready, [x, y], "先清空再写，不残留上一步的就绪变量");
    assert_eq!(ready.capacity(), capacity, "缓冲没有重新分配");
    assert!(account.finished());
}

/// 一个账本只管一份调度：空账本拒绝一切，`finished` 也只是平凡真。
#[test]
fn an_empty_account_accepts_nothing() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let recorded = tape.record("relu", &[x]).expect("记录");
    let node = tape.node(recorded).expect("节点");
    let mut account = Account::new();
    assert!(account.finished());
    assert_eq!(
        account.settle(node).expect_err("没有调度").code(),
        Code::Invalid
    );
    assert!(
        account
            .settle(node)
            .expect_err("没有调度")
            .to_string()
            .contains("不是本次调度")
    );
}

/// 每个输入都有扇出，图上变量的计数与独立统计一致；不进图的变量不收账。
#[test]
fn fan_out_matches_an_independent_count() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let constant = tape.leaf();
    let a = tape.record("add", &[x, constant]).expect("记录");
    let b = tape.record("mul", &[a, a]).expect("记录");
    let loss = tape.record("reduce_sum", &[b, constant]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    let account = schedule.account();
    assert_eq!(account.pending(loss), 0, "loss 自己不是任何步骤的输入");
    assert_eq!(account.pending(b), 1);
    assert_eq!(account.pending(a), 2, "mul 里出现两次");
    assert_eq!(account.pending(x), 1);
    assert_eq!(account.pending(constant), 0, "常量不进图");
    assert_eq!(
        tape.requirement(constant),
        Some(Requirement::None),
        "常量确实在图外"
    );
    assert!(!account.ready(constant), "图外变量永远不会'就绪'");
}
