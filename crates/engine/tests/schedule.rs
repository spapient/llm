//! 黑盒契约：调度器作为"第三方"使用时的准入、推进、腾页与错误类别。
//!
//! 内联测试直接摸页表与序列内部字段；这里只经 `Scheduler` / `Sequence` / `Progress` 的公开面
//! 断言对外承诺：准入回答"能不能进"，推进回答"生成没生成"，`Blocked` 不修改状态且可原样重试，
//! 三个状态不会互相越界。

use llm_engine::exec::Stream;
use llm_engine::kv::Cache;
use llm_engine::{Progress, Scheduler, Sequence};

/// 页池：`pages` 个页，每页 `page_size` 个 token。
fn scheduler(pages: usize, page_size: usize) -> Scheduler {
    Scheduler::new(Cache::new(pages, page_size).expect("页池参数合法"))
}

fn sequence(id: u64, prompt: usize, max_new: usize, page_size: usize) -> Sequence {
    Sequence::new(id, prompt, max_new, page_size, Stream::new(id as u32)).expect("序列参数合法")
}

#[test]
fn a_sequence_generates_one_token_per_step_until_done() {
    let mut scheduler = scheduler(4, 16);
    let mut sequence = sequence(1, 4, 2, 16);
    assert!(
        scheduler.admit(&sequence).expect("准入"),
        "空页池必须能放下短序列"
    );
    assert_eq!(sequence.produced(), 0);

    assert_eq!(
        scheduler.advance(&mut sequence).expect("推进"),
        Progress::Running,
        "两步里只走了一步"
    );
    assert_eq!(
        scheduler.advance(&mut sequence).expect("推进"),
        Progress::Done
    );
    assert_eq!(sequence.produced(), 2);
    assert!(sequence.is_done(), "`Done` 与 `is_done` 必须同一个判据");
    assert!(scheduler.finish(sequence.id()).expect("释放"));
}

#[test]
fn a_sequence_that_cannot_run_to_completion_is_never_admitted() {
    let mut scheduler = scheduler(2, 8);
    // 最坏情况：提示 1 + 生成 64 个 token，需要 9 页，而池只有 2 页。
    let sequence = sequence(1, 1, 64, 8);
    assert!(
        !scheduler.fits(&sequence),
        "独占页池也跑不完的请求不算放得下"
    );
    assert!(
        !scheduler.admit(&sequence).expect("准入"),
        "放不下就不能进批"
    );
    assert_eq!(scheduler.running(), 0);
}

#[test]
fn a_sequence_occupies_only_the_pages_its_current_length_needs() {
    let mut scheduler = scheduler(4, 4);
    let mut sequence = sequence(1, 4, 8, 4);
    assert!(scheduler.admit(&sequence).expect("准入"));
    assert_eq!(
        scheduler.cache().in_use(),
        1,
        "长度 4 正好一页，不为最坏情况预占"
    );

    // 生成到长度 5：跨进第二页。
    assert_eq!(
        scheduler.advance(&mut sequence).expect("推进"),
        Progress::Running
    );
    assert_eq!(sequence.len(), 5);
    assert_eq!(scheduler.cache().in_use(), 2, "跨页即增长一页");
}

#[test]
fn a_blocked_step_neither_generates_nor_moves_pages() {
    // 两个页、每页 2 个 token：两条序列各需 2 页时池刚好占满，第三步必然有人要不到页。
    let mut scheduler = scheduler(2, 2);
    let mut first = sequence(1, 2, 2, 2);
    let second = sequence(2, 2, 2, 2);
    assert!(scheduler.admit(&first).expect("准入"));
    assert!(scheduler.admit(&second).expect("准入"));

    // 推进到某一条进入 Done 或 Blocked 为止：两条都占满页池时，任意一条继续都要新页。
    let before = scheduler.cache().in_use();
    let outcome = scheduler.advance(&mut first).expect("推进");
    let after = scheduler.cache().in_use();
    match outcome {
        Progress::Blocked => {
            assert_eq!(first.produced(), 0, "`Blocked` 不许生成 token");
            assert_eq!(after, before, "`Blocked` 不许改页账");
            assert!(
                scheduler.blocked().contains(&first.id()),
                "刚被阻塞的序列必须在等待清单里，调用方据此挑牺牲者"
            );
        }
        Progress::Running | Progress::Done => {
            assert_eq!(first.produced(), 1, "非阻塞结局必须真的生成一个 token");
            assert_eq!(
                after,
                before.max(scheduler.cache().in_use()),
                "页账只增不减"
            );
        }
    }
    assert!(scheduler.finish(second.id()).expect("腾页"));
    assert!(scheduler.finish(first.id()).expect("腾页"));
    assert_eq!(scheduler.cache().in_use(), 0, "两条都结束后页必须全部归还");
}

#[test]
fn parameters_that_cannot_work_are_rejected_with_categories() {
    let error = Sequence::new(0, 1, 0, 4, Stream::new(0)).expect_err("max_new 为 0");
    assert_eq!(error.code().as_str(), "invalid");
    let error = Sequence::new(0, 1, 1, 0, Stream::new(0)).expect_err("页宽为 0");
    assert_eq!(error.code().as_str(), "invalid");
    let error = Cache::new(0, 4).expect_err("零页池");
    assert_eq!(error.code().as_str(), "invalid");
}

#[test]
fn finishing_an_unknown_sequence_reports_absence_without_moving_pages() {
    // `finish` 的对外契约是"返回它是否在批里"，不是"未知 id 就报错"：把不在批里的 id 当成错误，
    // 会让幂等收尾（两条路径都调一次 finish）变成必须先查再删。
    let mut scheduler = scheduler(2, 4);
    let sequence = sequence(1, 2, 1, 4);
    assert!(scheduler.admit(&sequence).expect("准入"));
    assert!(scheduler.finish(sequence.id()).expect("在批里"));
    let after_first = scheduler.cache().in_use();
    assert_eq!(after_first, 0, "结束即归还，页数回到零");

    assert!(
        !scheduler.finish(42).expect("未知 id 不报错"),
        "从未准入的 id 必须如实回答「不在批里」"
    );
    assert_eq!(
        scheduler.cache().in_use(),
        after_first,
        "未知 id 不得动页账"
    );
    assert_eq!(scheduler.running(), 0);
}

#[test]
fn admitting_the_same_id_twice_is_rejected_with_a_category() {
    let mut scheduler = scheduler(4, 4);
    let sequence = sequence(7, 2, 2, 4);
    assert!(scheduler.admit(&sequence).expect("首次准入"));
    let error = scheduler
        .admit(&sequence)
        .expect_err("重复 id 必须拒绝，而不是覆盖前一条");
    assert_eq!(
        error.code().as_str(),
        "invalid",
        "重复 id 是请求自己不成立，重试不会好"
    );
}
