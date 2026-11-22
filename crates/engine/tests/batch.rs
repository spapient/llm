//! 连续批的行为夹具：准入、推进、KV 记账与前缀复用。

use super::*;
use crate::exec::Stream;
use crate::ifb::{Budget, Verdict};
use crate::kv::{Cache, Page, Pager};
use crate::{Progress, Sequence};
use diagnostic::Code;

fn pool(total: usize, page_size: usize) -> Pager {
    Pager::new(Cache::new(total, page_size).expect("页池"), 2).expect("行宽")
}

fn sequence(id: u64, prompt: usize, max_new: usize, page_size: usize) -> Sequence {
    let prompt = vec![0u32; prompt];
    Sequence::new(id, &prompt, max_new, page_size, Stream::new(0)).expect("序列")
}

/// 指定提示 token 的序列：用于需要避开前缀整页共享的夹具（两条提示不同就不会采纳同一页）。
fn sequence_with(id: u64, tokens: &[u32], max_new: usize, page_size: usize) -> Sequence {
    Sequence::new(id, tokens, max_new, page_size, Stream::new(0)).expect("序列")
}

/// 混跑一轮的夹具：8 页、页宽 4 的批里一条 8 token 的提示段 + 一条 0 token 的纯解码。
fn prompt_and_decode_batch() -> Batch {
    let mut batch = Batch::new(pool(8, 4));
    batch.admit(sequence(1, 8, 4, 4)).expect("准入");
    batch.admit(sequence(2, 0, 4, 4)).expect("准入");
    batch
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
    // 两页、每页 2 个 token：两条序列各占一页，各自下一步都要新页。提示刻意不同，
    // 不触发前缀整页共享，才能复现"两条各独占一页、池被占满"的阻塞账。
    let mut batch = Batch::new(pool(2, 2));
    batch.admit(sequence_with(1, &[0, 0], 1, 2)).expect("准入");
    batch.admit(sequence_with(2, &[1, 1], 1, 2)).expect("准入");
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

/// 把一条序列的前若干满块算完：逐位置写 KV，写满一页就登记进前缀树。
fn prefill(batch: &mut Batch, id: u64, blocks: usize, width: usize) -> Vec<Page> {
    let row = vec![0.0f32; width];
    for token in 0..blocks * width {
        batch.write_kv(id, token, &row).expect("写 KV");
    }
    batch
        .scheduler()
        .pages_of(id)
        .expect("页表")
        .pages()
        .to_vec()
}

/// 已算好的块被下一个同前缀请求采纳：同一页号进新表，池里只占一个页位。
#[test]
fn a_written_block_is_adopted_by_the_next_identical_request() {
    let mut batch = Batch::new(pool(4, 2));
    batch.admit(sequence_with(1, &[0, 0], 1, 2)).expect("准入");
    let shared = prefill(&mut batch, 1, 1, 2)[0];

    batch.admit(sequence_with(2, &[0, 0], 1, 2)).expect("准入");
    assert_eq!(
        batch.scheduler().pages_of(2).expect("新表").pages()[0],
        shared,
        "已算好的块被采纳进新表"
    );
    assert_eq!(
        batch.scheduler().cache().reference_count(shared),
        3,
        "源表、树、新表各持一份"
    );
    assert_eq!(
        batch.scheduler().cache().in_use(),
        1,
        "三份引用只占一个池页"
    );
}

/// 源序列结束后前缀仍命中：树持着那一页，新请求照样接手，页不会回池清零。
#[test]
fn a_written_block_outlives_the_sequence_that_computed_it() {
    let mut batch = Batch::new(pool(4, 2));
    batch.admit(sequence_with(1, &[3, 3], 1, 2)).expect("准入");
    let shared = prefill(&mut batch, 1, 1, 2)[0];
    batch.write_kv(1, 0, &[9.0, 9.0]).expect("改写前缀行");

    assert!(batch.finish(1).expect("源序列结束"));
    assert_eq!(
        batch.scheduler().cache().reference_count(shared),
        1,
        "只剩树这一份"
    );
    assert!(batch.scheduler().cache().is_live(shared), "页不在池里");

    batch.admit(sequence_with(2, &[3, 3], 1, 2)).expect("准入");
    assert_eq!(
        batch.scheduler().pages_of(2).expect("新表").pages()[0],
        shared,
        "跨请求命中：源序列已经不在了"
    );
    assert_eq!(batch.kv(2, 0).expect("读共享前缀行"), [9.0, 9.0]);

    assert!(batch.finish(2).expect("共享方结束"));
    assert!(batch.scheduler().cache().is_live(shared), "树仍持着前缀页");
}

/// 引用计数跨结束顺序正确：每张表一份、树一份，谁先结束都不提前回池。
#[test]
fn a_shared_block_is_released_only_when_the_last_holder_goes() {
    let mut batch = Batch::new(pool(4, 2));
    batch
        .admit(sequence_with(1, &[5, 5], 1, 2))
        .expect("准入源序列");
    let shared = prefill(&mut batch, 1, 1, 2)[0];
    batch
        .admit(sequence_with(2, &[5, 5], 1, 2))
        .expect("准入共享序列");

    assert_eq!(
        batch.scheduler().cache().reference_count(shared),
        3,
        "源表、树、共享表各一份"
    );
    assert!(batch.finish(1).expect("源序列先结束"));
    assert_eq!(batch.scheduler().cache().reference_count(shared), 2);
    assert!(batch.scheduler().cache().is_live(shared));
    assert_eq!(batch.scheduler().cache().in_use(), 1, "共享页只占一个池页");

    assert!(batch.finish(2).expect("共享方结束"));
    assert_eq!(
        batch.scheduler().cache().reference_count(shared),
        1,
        "只剩树这一份"
    );
    assert!(
        !batch.finish(2).expect("重复结束恒等"),
        "重复结束不再次归还"
    );
}

/// 三方可共享同一前缀，谁先结束都只减自己那一份。
#[test]
fn three_requests_share_one_block() {
    let mut batch = Batch::new(pool(6, 2));
    batch
        .admit(sequence_with(1, &[4, 4], 1, 2))
        .expect("第一条");
    let shared = prefill(&mut batch, 1, 1, 2)[0];
    batch
        .admit(sequence_with(2, &[4, 4], 1, 2))
        .expect("第二条");
    batch
        .admit(sequence_with(3, &[4, 4], 1, 2))
        .expect("第三条");

    assert_eq!(batch.scheduler().cache().reference_count(shared), 4);
    assert_eq!(batch.scheduler().cache().in_use(), 1, "四方共享一个池页");

    assert!(batch.finish(2).expect("第二条先结束"));
    assert_eq!(batch.scheduler().cache().reference_count(shared), 3);
    assert!(batch.finish(1).expect("第一条结束"));
    assert_eq!(batch.scheduler().cache().reference_count(shared), 2);
    assert!(batch.finish(3).expect("第三条结束"));
    assert_eq!(
        batch.scheduler().cache().reference_count(shared),
        1,
        "只剩树这一份"
    );
}

/// 首页内发散的前缀不共享：新序列另取新页，两条各占一个池页位。
#[test]
fn diverging_prefix_takes_fresh_pages() {
    let mut batch = Batch::new(pool(4, 2));
    batch.admit(sequence_with(1, &[0, 0], 1, 2)).expect("准入");
    batch.admit(sequence_with(2, &[0, 1], 1, 2)).expect("准入");

    let first = batch.scheduler().pages_of(1).expect("源表");
    let second = batch.scheduler().pages_of(2).expect("新表");
    assert_ne!(second.pages()[0], first.pages()[0], "首页内发散，不共享");
    assert_eq!(batch.scheduler().cache().in_use(), 2, "两条各独占一页");
}

/// 公共前缀不满一页时不拆页：整页共享只对齐到页边界。
#[test]
fn prefix_shorter_than_one_page_is_not_split() {
    let mut batch = Batch::new(pool(4, 4));
    batch.admit(sequence_with(1, &[0, 0], 1, 4)).expect("准入");
    batch.admit(sequence_with(2, &[0, 0], 1, 4)).expect("准入");

    let first = batch.scheduler().pages_of(1).expect("源表");
    let second = batch.scheduler().pages_of(2).expect("新表");
    assert_ne!(
        second.pages()[0],
        first.pages()[0],
        "前缀 2 token < 页宽 4，落在新页"
    );
    assert_eq!(batch.scheduler().cache().in_use(), 2);
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

/// 批内两条以上、不同长度与不同前缀的序列同批跑多步：各自历史与进度对拍，不串台。
///
/// 三条序列：一条短提示、一条长提示、一条与首条共享整页前缀；每轮同批 `step`，再按 id
/// 把采样结果 `emit` 记进各自历史。轮次结束后逐条核对生成数、生成历史与上下文长度，
/// 共享前缀那条的提示段不能被另两条的生成 token 顶替。已跑满的序列不再参与记账。
#[test]
fn multi_sequence_batch_runs_their_own_history_and_progress() {
    // 页宽 2：前缀整页共享；页池 12 页，三条各自提示与跨页增长都放得下，不互相挡页。
    let mut batch = Batch::new(pool(12, 2));
    batch
        .admit(sequence_with(1, &[7, 7], 3, 2))
        .expect("准入第一条");
    batch
        .admit(sequence_with(2, &[8, 8, 9, 9], 2, 2))
        .expect("准入第二条");
    batch
        .admit(sequence_with(3, &[7, 7], 2, 2))
        .expect("准入第三条（与第一条共享整页前缀）");

    // 每轮三条各走一步；采样 token 按序列各自编号，记账与推进成对（`emit` 内部先预留位再记账）。
    for round in 0..3 {
        let ids: Vec<u64> = batch.ids().collect();
        let mut recorded: Vec<Progress> = Vec::with_capacity(ids.len());
        for id in ids {
            if batch.sequence(id).expect("在批里").is_done() {
                recorded.push(Progress::Done);
                continue;
            }
            let token = id as u32 * 10 + round as u32;
            let progress = batch.emit(id, token).expect("记账推进");
            recorded.push(progress);
        }
        // 批量视图把本轮结局按准入顺序对齐到每条序列：不串台。
        let view = batch.outcomes(&recorded);
        assert_eq!(view.len(), 3, "批量视图与准入序列一一对应");
    }

    let first = batch.sequence(1).expect("第一条在批里");
    let second = batch.sequence(2).expect("第二条在批里");
    let third = batch.sequence(3).expect("第三条在批里");

    assert_eq!(first.tokens(), [10, 11, 12], "第一条历史按自己的生成顺序");
    assert_eq!(second.tokens(), [20, 21], "第二条先跑满 max_new");
    assert_eq!(third.tokens(), [30, 31], "第三条历史不与第一条串");
    assert_eq!(first.produced(), 3);
    assert_eq!(second.produced(), 2);
    assert_eq!(third.produced(), 2);
    assert!(first.is_done(), "第一条跑满上限");
    assert!(second.is_done(), "第二条跑满上限");
    assert!(third.is_done(), "第三条跑满上限");

    // 上下文 = 提示段 + 各自生成段：共享前缀那条的提示仍是 [7, 7]，没被生成历史冲掉。
    assert_eq!(first.ctx(), [7, 7, 10, 11, 12]);
    assert_eq!(second.ctx(), [8, 8, 9, 9, 20, 21]);
    assert_eq!(third.ctx(), [7, 7, 30, 31]);
}

/// 批量结果视图：`outcomes` 按准入顺序把每条序列的历史与本步进度对齐，视图间互不重叠。
#[test]
fn batch_outcomes_pair_each_progress_with_its_own_sequence() {
    let mut batch = Batch::new(pool(4, 4));
    batch.admit(sequence(1, 0, 2, 4)).expect("准入第一条");
    batch.admit(sequence(2, 0, 1, 4)).expect("准入第二条");

    // 第一步走 `emit`（预留位 + 记账成对）；结局按准入顺序收进批量视图。
    let first = batch.emit(1, 5).expect("第一条第一步");
    let second = batch.emit(2, 6).expect("第二条第一步");
    let outcomes = [first, second];
    let view = batch.outcomes(&outcomes);
    assert_eq!(view.len(), 2);

    assert_eq!(view[0].id(), 1);
    assert_eq!(view[0].progress(), Progress::Running);
    assert_eq!(view[0].produced(), 1);
    assert_eq!(view[0].tokens(), [5], "第一条历史挂在自己序列上");
    assert_eq!(view[1].id(), 2);
    // 第二条 `max_new = 1`，这一步即跑满：结局是 `Done`，不是 `Running`。
    assert_eq!(view[1].progress(), Progress::Done);
    assert_eq!(view[1].tokens(), [6], "第二条历史挂在自己序列上");
    assert!(view[1].is_done());

    // 第二条已结束不再推进；第一条再走一步，视图仍按准入位置对齐到各自序列。
    let third = batch.emit(1, 7).expect("第一条第二步");
    let view = batch.outcomes(&[third, Progress::Done]);
    assert_eq!(view[0].tokens(), [5, 7], "第一条两步历史不串");
    assert_eq!(view[1].tokens(), [6], "第二条历史仍是第一步那一笔");

    // 按序号取单条视图，越界返回 `None`。
    assert_eq!(batch.outcome(0, third).expect("首条").id(), 1);
    assert!(batch.outcome(2, third).is_none(), "越界序号无视图");
    assert_eq!(batch.sequence_at(1).expect("按序号取序列").id(), 2);
    assert!(batch.sequence_at(2).is_none());
}

/// 准入满批退回：页不够时序列原样退回、不占页，重试前页账与退回前一致。
#[test]
fn admission_full_handles_back_without_touching_the_page_accounts() {
    // 池 2 页、页宽 2：第一条提示 2 占 1 页；第二条提示 2、`max_new = 4`，最坏情况要 3 页，
    // 既超过池总页数、空闲也只剩 1 页，准入补不出差额，整页退回，不留半个分配。
    let mut batch = Batch::new(pool(2, 2));
    batch
        .admit(sequence_with(1, &[7, 7], 1, 2))
        .expect("准入第一条");
    let rejected = match batch
        .admit(sequence_with(2, &[8, 8], 4, 2))
        .expect("准入判定")
    {
        Admission::Full(sequence) => sequence,
        Admission::Admitted => panic!("空闲页不够，第二条应原样退回"),
    };
    assert_eq!(rejected.id(), 2);
    assert_eq!(batch.len(), 1);
    assert_eq!(batch.scheduler().cache().in_use(), 1, "退回不占页");
    assert_eq!(batch.scheduler().cache().free_pages(), 1);

    // 第一条结束腾出页位后：第二条最坏情况仍要 3 页，池只有 2 页，`fits` 不通过，
    // 继续按"现在装不下"退回，而不是被部分准入；它的页账原样不动。
    assert!(batch.finish(1).expect("结束第一条"));
    assert_eq!(batch.scheduler().cache().in_use(), 0);
    match batch.admit(rejected).expect("腾页后重试仍判容量") {
        Admission::Full(sequence) => {
            assert_eq!(sequence.id(), 2, "装不下的序列原样退回");
            assert_eq!(batch.scheduler().cache().in_use(), 0, "重试失败不留分配");
        }
        Admission::Admitted => panic!("池只有 2 页，第二条最坏情况要 3 页，不该准入"),
    }
}

/// 结束序列的页随序列回收：独占新页全部回池，增长出来的页也一并归还，不残留占用。
#[test]
fn finishing_a_sequence_releases_all_of_its_pages() {
    let mut batch = Batch::new(pool(4, 2));
    batch.admit(sequence_with(1, &[3, 3], 3, 2)).expect("准入");
    // 提示长度 2 占 1 页；再生成三步：长度 4 跨进第二页，长度 6 跨进第三页，末步跑满上限。
    for (idx, token) in (10..13).enumerate() {
        let expected = if idx == 2 {
            Progress::Done
        } else {
            Progress::Running
        };
        assert_eq!(batch.emit(1, token).expect("生成"), expected);
    }
    assert_eq!(batch.scheduler().cache().in_use(), 3, "增长后占三页");

    assert!(batch.finish(1).expect("结束"));
    assert!(batch.is_empty());
    assert_eq!(batch.scheduler().cache().in_use(), 0, "全部页归还");
    assert_eq!(
        batch.scheduler().cache().free_pages(),
        batch.scheduler().cache().total(),
        "归还后空闲等于池总页"
    );
}

/// 预填充只登记"行已落定"的整块：推进到 3 个位置只登记第 0 块，满 4 个位置才有第 1 块。
#[test]
fn marking_a_prompt_prefilled_publishes_only_whole_blocks() {
    let mut batch = Batch::new(pool(8, 2));
    assert_eq!(
        batch
            .admit(sequence_with(1, &[1, 2, 3, 4, 5], 2, 2))
            .expect("准入"),
        Admission::Admitted
    );
    assert!(!batch.sequence(1).expect("在批里").prefilled_ready());
    batch.mark_prefilled(1, 3).expect("算完三个位置");
    assert_eq!(batch.scheduler().cached_blocks(), 1, "只有第 0 块写满了");
    assert!(!batch.sequence(1).expect("在批里").prefilled_ready());
    batch.mark_prefilled(1, 5).expect("算完提示段");
    assert!(batch.sequence(1).expect("在批里").prefilled_ready());
    assert_eq!(
        batch.scheduler().cached_blocks(),
        2,
        "第 2 块只有一行，不够一块，不许登记"
    );
}

/// 预填充位置只许往前：倒退会把已登记的块当成没算过，越过提示段则索引不存在的行。
#[test]
fn a_prefill_position_cannot_go_backwards_or_past_the_prompt() {
    let mut batch = Batch::new(pool(4, 2));
    batch
        .admit(sequence_with(1, &[1, 2, 3, 4], 2, 2))
        .expect("准入");
    batch.mark_prefilled(1, 3).expect("算完三个位置");
    assert_eq!(
        batch.mark_prefilled(1, 2).unwrap_err().code(),
        Code::Invalid,
        "倒退不许"
    );
    assert_eq!(
        batch.mark_prefilled(1, 5).unwrap_err().code(),
        Code::Invalid,
        "越过提示段不许"
    );
    assert_eq!(
        batch.mark_prefilled(9, 1).unwrap_err().code(),
        Code::Invalid,
        "不在批里的序列不许"
    );
}

/// 解码写满一块也登记：两条路径（预填充、解码）走的是同一份"落定即登记"。
#[test]
fn writing_the_last_row_of_a_block_publishes_it() {
    let mut batch = Batch::new(pool(8, 2));
    batch.admit(sequence_with(1, &[1, 2], 4, 2)).expect("准入");
    assert_eq!(batch.scheduler().cached_blocks(), 0);
    batch.step().expect("推进一步：页表随长度长到两页");
    batch.write_kv(1, 2, &[0.5; 2]).expect("写第 2 行");
    assert_eq!(
        batch.scheduler().cached_blocks(),
        1,
        "第 2 行落定即第 0 块写完"
    );
}

/// 预填充过的一次提示，下一个同前缀请求只补差额页：共享的页就是同一批页号。
#[test]
fn a_prefilled_prompt_is_shared_with_the_next_request() {
    let mut batch = Batch::new(pool(8, 2));
    let prompt = [1, 2, 3, 4];
    batch.admit(sequence_with(1, &prompt, 1, 2)).expect("准入");
    let first: Vec<Page> = batch
        .scheduler()
        .pages_of(1)
        .expect("页表")
        .pages()
        .to_vec();
    assert_eq!(first.len(), 2, "提示 4 个 token、页宽 2：正好两页");
    batch.mark_prefilled(1, prompt.len()).expect("算完提示段");
    assert_eq!(batch.scheduler().cached_blocks(), 2);
    batch.finish(1).expect("结束");
    assert_eq!(
        batch.scheduler().cache().in_use(),
        2,
        "块由前缀树持有，请求结束不还"
    );

    batch.admit(sequence_with(2, &prompt, 1, 2)).expect("准入");
    let second: Vec<Page> = batch
        .scheduler()
        .pages_of(2)
        .expect("页表")
        .pages()
        .to_vec();
    assert_eq!(&second[..2], &first[..2], "前两页整块接走，页号必须同源");
    assert_eq!(second.len(), 2, "整段提示都接走，一页也不用新开");
}

/// 召回按整块走：块登记进去后，页已被别的请求共享也不会被当作空闲页摘走。
#[test]
fn a_published_block_keeps_its_page_out_of_the_free_list() {
    let mut batch = Batch::new(pool(2, 2));
    batch.admit(sequence_with(1, &[1, 2], 1, 2)).expect("准入");
    batch.mark_prefilled(1, 2).expect("算完提示段");
    let page = batch.scheduler().pages_of(1).expect("页表").pages()[0];
    batch.finish(1).expect("结束");
    assert!(
        batch.scheduler().cache().is_live(page),
        "块还在树上，页不能回池"
    );
    assert_eq!(
        batch.scheduler().cache().free_pages(),
        1,
        "两页池只空闲一页"
    );
}

/// 这一轮的行由相位定序：提示段还没算完的在后面，能解码的在前。
///
/// 相位不是调用方给的标签，而是序列自己的事实（`prefilled < prompt_len`），所以"谁先进批"
/// 与"谁先跑"是两件事。
#[test]
fn a_round_runs_generation_before_ctx() {
    let batch = prompt_and_decode_batch();

    let Verdict::Packed { rows, tokens } = batch.round(Budget::unlimited()) else {
        panic!("不限预算必须装得下");
    };
    assert_eq!(
        rows.iter().map(|row| row.id()).collect::<Vec<_>>(),
        [2, 1],
        "能解码的在前，提示段在后"
    );
    assert_eq!(rows[1].rows(), 8, "8 个 token 的提示段一轮算完");
    assert_eq!(tokens, 9);
}

/// 挤压时只挤上下文：生成行留着，上下文整块推到下一轮。
///
/// 块大小由 [`crate::ifb::chunk`] 按页对齐给出，所以"少跑一点"只能是整页少跑，不会出现
/// 半页的块——那种块既不进前缀树，也换不来一次干净的对齐。
#[test]
fn a_tight_budget_pushes_the_ctx_block_to_the_next_round() {
    let batch = prompt_and_decode_batch();

    let Verdict::Packed { rows, tokens } = batch.round(Budget::new(2, 2)) else {
        panic!("生成行只要 1 行，装得下");
    };
    assert_eq!(
        rows.iter().map(|row| row.id()).collect::<Vec<_>>(),
        [2],
        "2 个 token 的预算连一整页上下文都放不下，只跑生成"
    );
    assert_eq!(tokens, 1);

    let Verdict::Packed { rows, tokens } = batch.round(Budget::new(2, 5)) else {
        panic!("1 生成 + 4 上下文正好 5 行");
    };
    assert_eq!(
        rows.iter()
            .map(|row| (row.id(), row.rows()))
            .collect::<Vec<_>>(),
        [(2, 1), (1, 4)],
        "预算够了就按整页推进一块"
    );
    assert_eq!(tokens, 5);
}

/// 两类"装不下"各报各的：序列数超了可等一条结束，生成行自己超预算重试无用。
#[test]
fn a_round_separates_the_two_over_budget_cases() {
    let mut batch = Batch::new(pool(8, 4));
    batch.admit(sequence(1, 0, 4, 4)).expect("准入");
    batch.admit(sequence(2, 0, 4, 4)).expect("准入");

    assert_eq!(
        batch.round(Budget::new(1, 4)),
        Verdict::OverSeqs { seqs: 2, max: 1 }
    );
    assert_eq!(
        batch.round(Budget::new(2, 1)),
        Verdict::OverTokens { tokens: 2, max: 1 }
    );
}

/// 空批不排任何行：`Idle` 与"被预算挤掉"是两件事，调用方据此决定要不要继续等。
#[test]
fn an_empty_batch_has_nothing_to_round() {
    let batch = Batch::new(pool(4, 4));
    assert_eq!(batch.round(Budget::unlimited()), Verdict::Idle);
}
