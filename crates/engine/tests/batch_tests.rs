use super::*;
use crate::exec::Stream;
use crate::kv::Cache;

fn pool(total: usize, page_size: usize) -> Pager {
    Pager::new(Cache::new(total, page_size).expect("页池"), 2).expect("行宽")
}

fn sequence(id: u64, prompt: usize, max_new: usize, page_size: usize) -> Sequence {
    let prompt = vec![0u32; prompt];
    Sequence::new(id, &prompt, max_new, page_size, Stream::new(0)).expect("序列")
}

#[test]
fn an_admitted_sequence_is_owned_by_the_batch() {
    let mut batch = Batch::new(pool(4, 4));
    assert!(batch.is_empty());
    assert_eq!(
        batch.admit(sequence(7, 4, 4, 4)).expect("准入"),
        Admission::Admitted
    );
    assert_eq!(batch.len(), 1);
    assert_eq!(batch.ids().collect::<Vec<u64>>(), [7]);
    let admitted = batch.sequence(7).expect("在批里");
    assert_eq!(admitted.state(), crate::State::Waiting);
    assert_eq!(batch.scheduler().cache().in_use(), 1);
}

/// "现在满"必须原样退回序列：调用方要能把它排队，而不是丢掉一条合法请求。
#[test]
fn a_full_batch_hands_the_sequence_back() {
    let mut batch = Batch::new(pool(1, 4));
    let rejected = match batch.admit(sequence(1, 4, 4, 4)).expect("准入判定") {
        Admission::Full(sequence) => sequence,
        Admission::Admitted => panic!("页池只有 1 页，装不下 2 页的最坏情况"),
    };
    assert_eq!(rejected.id(), 1);
    assert!(batch.is_empty());
    assert_eq!(batch.scheduler().cache().in_use(), 0, "退回不能占页");
}

#[test]
fn step_advances_every_sequence_in_admission_order() {
    let mut batch = Batch::new(pool(8, 4));
    batch.admit(sequence(1, 0, 4, 4)).expect("准入");
    batch.admit(sequence(2, 0, 4, 4)).expect("准入");
    assert_eq!(
        batch.step().expect("一步"),
        [Progress::Running, Progress::Running]
    );
    assert_eq!(batch.sequence(1).expect("在批里").produced(), 1);
    assert_eq!(batch.sequence(2).expect("在批里").produced(), 1);
    assert_eq!(batch.ids().collect::<Vec<u64>>(), [1, 2]);
}

/// 页用尽时只有"下一步确实要新页"的那条被挡住：其余序列照常推进，批不会整体停摆。
#[test]
fn a_blocked_sequence_does_not_stop_the_batch() {
    let mut batch = Batch::new(pool(2, 4));
    batch.admit(sequence(1, 0, 8, 4)).expect("准入");
    batch.admit(sequence(2, 4, 4, 4)).expect("准入");
    assert_eq!(batch.scheduler().cache().free_pages(), 0);
    assert_eq!(batch.blocked(), [2], "只有第 2 条需要新页");

    assert_eq!(
        batch.step().expect("一步"),
        [Progress::Running, Progress::Blocked]
    );
    assert_eq!(batch.sequence(1).expect("在批里").produced(), 1);
    assert_eq!(
        batch.sequence(2).expect("在批里").produced(),
        0,
        "被挡住的那条什么都不该改变"
    );
}

#[test]
fn finishing_returns_pages_and_removes_the_sequence() {
    let mut batch = Batch::new(pool(4, 4));
    batch.admit(sequence(3, 8, 4, 4)).expect("准入");
    assert_eq!(batch.scheduler().cache().in_use(), 2);
    assert!(batch.finish(3).expect("结束"));
    assert!(batch.is_empty());
    assert_eq!(batch.scheduler().cache().in_use(), 0, "页全部归还");
    assert!(!batch.finish(3).expect("重复结束是恒等"), "重复结束不算错");
    assert_eq!(
        batch.advance(3).unwrap_err().code(),
        Code::Invalid,
        "已经离开批的序列不能再推进"
    );
}

#[test]
fn a_stopped_sequence_cannot_be_advanced() {
    let mut batch = Batch::new(pool(4, 4));
    batch.admit(sequence(5, 0, 8, 4)).expect("准入");
    batch.sequence_mut(5).expect("在批里").stop().expect("停止");
    assert!(batch.sequence(5).expect("在批里").is_done());
    assert_eq!(batch.advance(5).unwrap_err().code(), Code::Invalid);
    // 页仍然占着，直到 `finish`：结束本身不碰页账。
    assert_eq!(batch.scheduler().cache().in_use(), 1);
}

#[test]
fn a_duplicate_id_is_refused_rather_than_silently_merged() {
    let mut batch = Batch::new(pool(4, 4));
    batch.admit(sequence(1, 0, 4, 4)).expect("准入");
    assert_eq!(
        batch.admit(sequence(1, 0, 4, 4)).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(batch.len(), 1);
}

/// 记账与推进成对：`emit` 之后计数与历史各长一格，顺序与采样顺序一致。
#[test]
fn emit_records_the_token_in_the_sequence() {
    let mut batch = Batch::new(pool(4, 4));
    batch.admit(sequence(1, 0, 3, 4)).expect("准入");
    assert!(batch.sequence(1).expect("在批里").tokens().is_empty());

    assert_eq!(batch.emit(1, 7).expect("生成"), Progress::Running);
    assert_eq!(batch.emit(1, 8).expect("生成"), Progress::Running);
    let admitted = batch.sequence(1).expect("在批里");
    assert_eq!(admitted.tokens(), [7, 8], "历史按生成顺序");
    assert_eq!(admitted.produced(), 2, "计数与历史同步");

    assert_eq!(batch.emit(1, 9).expect("最后一步"), Progress::Done);
    assert_eq!(
        batch.sequence(1).expect("在批里").tokens(),
        [7, 8, 9],
        "走到上限的那一步也要记账"
    );
}

/// 被挡的那一步没有生成：历史里不能多一个 token，否则重试会看见一个不存在的输出。
#[test]
fn a_blocked_step_does_not_record_a_token() {
    // 两页、每页 2 个 token：两条序列各占一页，各自下一步都要新页。
    let mut batch = Batch::new(pool(2, 2));
    batch.admit(sequence(1, 2, 1, 2)).expect("准入");
    batch.admit(sequence(2, 2, 1, 2)).expect("准入");
    assert_eq!(batch.blocked(), [1, 2], "两条都在等页");

    assert_eq!(batch.emit(1, 5).expect("被挡不是错误"), Progress::Blocked);
    let blocked = batch.sequence(1).expect("在批里");
    assert!(blocked.tokens().is_empty(), "没有生成就没有历史");
    assert_eq!(blocked.produced(), 0, "被挡不推进计数");

    // 腾出一页之后重试成功：同一个 token 这时才被记住。
    assert!(batch.finish(2).expect("腾页"));
    assert_eq!(batch.emit(1, 5).expect("重试"), Progress::Done);
    assert_eq!(batch.sequence(1).expect("在批里").tokens(), [5]);
}

/// 记满之后再记是调用方多记了一次：上限只有 `max_new` 一处。
#[test]
fn recording_past_the_limit_is_refused() {
    let mut batch = Batch::new(pool(4, 4));
    batch.admit(sequence(1, 0, 2, 4)).expect("准入");
    assert_eq!(batch.emit(1, 1).expect("第一步"), Progress::Running);
    assert_eq!(batch.emit(1, 2).expect("第二步"), Progress::Done);
    let sequence = batch.sequence_mut(1).expect("在批里");
    assert_eq!(
        sequence.push_token(3).unwrap_err().code(),
        Code::Invalid,
        "第三个 token 超出 max_new"
    );
    assert_eq!(sequence.tokens(), [1, 2], "失败的记账不留痕");
}
