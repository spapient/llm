use super::*;
use crate::kv::{Cache, Pager};
use diagnostic::Code;
use testkit::Rng;

fn sequence(id: u64, prompt: usize, max_new: usize) -> Sequence {
    let prompt = vec![0u32; prompt];
    Sequence::new(id, &prompt, max_new, 16, crate::exec::Stream::new(0)).expect("序列")
}

/// 页池 + 行宽 4：这些测试只关心页账，行里的数由 `kv_tests` 覆盖。
fn pool(total: usize, page_size: usize) -> Pager {
    Pager::new(Cache::new(total, page_size).expect("页池"), 4).expect("行宽")
}

/// 准入判定：`None` 是"现在装不下"，`Some` 是能开始（共享前缀长度不影响这些用例）。
fn admitted(scheduler: &mut Scheduler, sequence: &Sequence, why: &str) -> bool {
    scheduler
        .admit(sequence)
        .unwrap_or_else(|_| panic!("{why}"))
        .is_some()
}

#[test]
fn ctx_holds_the_full_prompt_plus_generated_tokens() {
    let mut sequence =
        Sequence::new(1, &[4u32, 5], 2, 16, crate::exec::Stream::new(0)).expect("序列");
    assert_eq!(sequence.prompt_len(), 2);
    assert_eq!(sequence.ctx(), [4, 5]);
    assert!(sequence.tokens().is_empty(), "提示段不算生成历史");

    sequence.push_token(6).expect("记账");
    assert_eq!(sequence.ctx(), [4, 5, 6]);
    assert_eq!(sequence.tokens(), [6]);
}

#[test]
fn a_sequence_generates_until_its_limit_then_stops() {
    let mut scheduler = Scheduler::new(pool(4, 16));
    let mut sequence = sequence(1, 4, 2);
    assert_eq!(sequence.state(), State::Waiting);
    assert!(admitted(&mut scheduler, &sequence, "准入"));
    assert_eq!(
        scheduler.advance(&mut sequence).expect("第一步"),
        Progress::Running
    );
    assert_eq!((sequence.produced(), sequence.state()), (1, State::Running));
    assert_eq!(
        scheduler.advance(&mut sequence).expect("第二步"),
        Progress::Done
    );
    assert!(sequence.is_done());
    assert_eq!(sequence.len(), 6);
    assert_eq!(
        scheduler.advance(&mut sequence).unwrap_err().code(),
        Code::Invalid,
        "结束之后不能再推进"
    );
    assert_eq!(
        scheduler.cache().in_use(),
        1,
        "结束的序列仍然占着页，直到 `finish`"
    );
}

#[test]
fn pages_are_rounded_up_and_never_zero() {
    let empty = sequence(0, 0, 1);
    assert!(empty.is_empty());
    assert_eq!(empty.pages_needed(), 1, "至少一页");
    let long = sequence(1, 17, 1);
    assert_eq!(long.pages_needed(), 2);
    assert_eq!(long.kv_shape().dims(), [17, 16]);
}

/// 最坏情况的上界与"当前需要"是两个数：准入要的是前者。
#[test]
fn pages_required_covers_the_whole_run_not_just_the_prompt() {
    let short = Sequence::new(1, &[0; 4], 4, 4, crate::exec::Stream::new(0)).expect("序列");
    assert_eq!(short.pages_needed(), 1, "当前长度 4 刚好一页");
    assert_eq!(short.pages_required(), 2, "跑满后长度 8 要两页");
    let long = Sequence::new(2, &[0; 1], 39, 4, crate::exec::Stream::new(0)).expect("序列");
    assert_eq!(long.pages_needed(), 1);
    assert_eq!(long.pages_required(), 10, "(1 + 39).div_ceil(4)");
}

/// 独占整个页池也跑不完的序列不进批：否则整批会卡在 `Blocked` 上，而没人能结束。
#[test]
fn a_sequence_that_can_never_finish_is_refused() {
    let mut scheduler = Scheduler::new(pool(4, 4));
    let endless = Sequence::new(1, &[0; 4], 40, 4, crate::exec::Stream::new(0)).expect("序列");
    assert_eq!(endless.pages_needed(), 1, "当前只占一页");
    assert!(
        !admitted(&mut scheduler, &endless, "页池放不下整段生成"),
        "跑不完的序列必须被拒绝，而不是先放进来再卡住"
    );
    assert_eq!(scheduler.cache().in_use(), 0, "拒绝不能留下任何分配");
    assert_eq!(scheduler.running(), 0);
}

#[test]
fn admission_occupies_and_finish_returns_the_pages() {
    let mut scheduler = Scheduler::new(pool(2, 16));
    let sequence = sequence(1, 20, 4);
    assert!(admitted(&mut scheduler, &sequence, "准入"));
    assert_eq!(scheduler.running(), 1);
    assert_eq!(scheduler.cache().in_use(), 2);
    assert_eq!(scheduler.pages_of(1).map(crate::kv::Table::len), Some(2));
    assert!(scheduler.finish(1).expect("结束"));
    assert_eq!(scheduler.cache().in_use(), 0);
    assert!(
        !scheduler.finish(1).expect("重复结束"),
        "重复结束返回 false"
    );
    assert_eq!(scheduler.pages_of(1), None);
}

#[test]
fn a_sequence_that_does_not_fit_is_refused_not_partially_admitted() {
    let mut scheduler = Scheduler::new(pool(2, 16));
    let big = sequence(7, 64, 4);
    assert!(
        !admitted(&mut scheduler, &big, "页不够"),
        "页不够时应返回 false"
    );
    assert_eq!(scheduler.cache().in_use(), 0, "拒绝不能留下半个分配");
    assert_eq!(scheduler.running(), 0);
}

#[test]
fn admitting_the_same_sequence_twice_is_a_bug() {
    let mut scheduler = Scheduler::new(pool(2, 16));
    let sequence = sequence(3, 8, 4);
    assert!(admitted(&mut scheduler, &sequence, "准入"));
    assert_eq!(
        scheduler.admit(&sequence).unwrap_err().code(),
        Code::Invalid
    );
}

#[test]
fn impossible_sequence_parameters_are_rejected() {
    assert_eq!(
        Sequence::new(0, &[0; 1], 1, 0, crate::exec::Stream::new(0))
            .unwrap_err()
            .code(),
        Code::Invalid
    );
    assert_eq!(
        Sequence::new(0, &[0; 1], 0, 16, crate::exec::Stream::new(0))
            .unwrap_err()
            .code(),
        Code::Invalid
    );
    // 上下文长度在构造期就校验：否则 `len() + 1` 会在某次推进里回绕成一个很小的页数。
    // 切片长度受内存约束，理论极值造不出切片；溢出校验仍保留在构造的 `checked_add` 里。
}

/// 生成会跨页：页表按"要写的位置"增长，而不是停在准入那一刻。
#[test]
fn a_growing_sequence_takes_a_page_each_time_it_crosses_one() {
    let mut scheduler = Scheduler::new(pool(4, 4));
    let mut sequence = Sequence::new(1, &[0; 3], 10, 4, crate::exec::Stream::new(0)).expect("序列");
    assert!(admitted(&mut scheduler, &sequence, "准入"));
    assert_eq!(scheduler.cache().in_use(), 1, "提示长度 3 只占一页");
    // 长度 4 仍在第一页，写到第 5 个位置才需要第二页，第 9 个位置才需要第三页。
    for (length, pages) in [(4, 1), (5, 2), (6, 2), (7, 2), (8, 2), (9, 3)] {
        assert_eq!(
            scheduler.advance(&mut sequence).expect("生成"),
            Progress::Running
        );
        assert_eq!(sequence.len(), length);
        assert_eq!(scheduler.cache().in_use(), pages, "长度 {length}");
    }
    assert_eq!(scheduler.pages_of(1).map(crate::kv::Table::len), Some(3));
    assert!(scheduler.finish(1).expect("结束"));
    assert_eq!(scheduler.cache().in_use(), 0, "增长出来的页也要还");
}

/// 池被别人占满时这一步不生成：序列、页表、页池三者都不动；等对方结束后原样重试即可。
#[test]
fn a_step_that_cannot_get_a_page_is_blocked_without_side_effects() {
    let mut scheduler = Scheduler::new(pool(3, 4));
    let mut a = Sequence::new(1, &[0; 4], 8, 4, crate::exec::Stream::new(0)).expect("序列");
    let b = Sequence::new(2, &[0; 4], 8, 4, crate::exec::Stream::new(0)).expect("序列");
    assert!(admitted(&mut scheduler, &b, "准入"));
    assert!(admitted(&mut scheduler, &a, "准入"));
    assert_eq!(scheduler.cache().in_use(), 2, "各自只占提示那一页");
    for length in 5..=8 {
        assert_eq!(scheduler.advance(&mut a).expect("生成"), Progress::Running);
        assert_eq!(a.len(), length);
    }
    assert_eq!(scheduler.cache().in_use(), 3, "长度 8 用掉最后一张空闲页");
    assert_eq!(
        scheduler.advance(&mut a).expect("页不够"),
        Progress::Blocked
    );
    assert_eq!(a.len(), 8, "阻塞不能推进序列");
    assert_eq!(a.produced(), 4, "阻塞不能多算一个 token");
    assert_eq!(scheduler.cache().in_use(), 3, "阻塞不能动页池");
    assert_eq!(scheduler.pages_of(1).map(crate::kv::Table::len), Some(2));
    assert_eq!(
        scheduler.blocked(),
        vec![2, 1],
        "按准入顺序列出：空闲为零时两张页表都还要长，所以两条都在等"
    );
    // 别人的页回来之后，同一步原样重试就能过。
    assert!(scheduler.finish(2).expect("结束"));
    assert!(scheduler.blocked().is_empty(), "页回来了，谁都不再等");
    assert_eq!(
        scheduler.advance(&mut a).expect("页回来了"),
        Progress::Running
    );
    assert_eq!((a.len(), scheduler.cache().in_use()), (9, 3));
}

/// 判据本身也是公开的：服务层要按它区分"现在满"与"永远装不下"。
#[test]
fn fits_separates_a_full_pool_from_a_never_finishable_request() {
    let scheduler = Scheduler::new(pool(2, 16));
    assert!(scheduler.fits(&sequence(1, 4, 2)), "两页里跑得完");
    let endless = Sequence::new(2, &[0; 4], 1024, 16, crate::exec::Stream::new(0)).expect("序列");
    assert!(!scheduler.fits(&endless), "最坏情况远超两页");
    assert!(scheduler.cache().in_use() == 0, "判据不占页");
}

#[test]
fn advancing_a_sequence_that_is_not_in_the_batch_is_a_bug() {
    let mut scheduler = Scheduler::new(pool(2, 16));
    let mut sequence = sequence(4, 4, 4);
    assert_eq!(
        scheduler.advance(&mut sequence).unwrap_err().code(),
        Code::Invalid
    );
    assert!(admitted(&mut scheduler, &sequence, "准入"));
    assert!(scheduler.finish(4).expect("结束"));
    assert_eq!(
        scheduler.advance(&mut sequence).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(sequence.produced(), 0, "失败的推进不能留下痕迹");
}

/// 页宽是页数的唯一依据：序列与页池不一致时，准入必须拒绝而不是少占页。
#[test]
fn a_sequence_whose_page_width_disagrees_with_the_pool_is_rejected() {
    let mut scheduler = Scheduler::new(pool(4, 16));
    let narrow = Sequence::new(5, &[0; 16], 4, 8, crate::exec::Stream::new(0)).expect("序列");
    assert_eq!(scheduler.admit(&narrow).unwrap_err().code(), Code::Invalid);
    assert_eq!(scheduler.cache().in_use(), 0, "拒绝不能留下分配");

    let mut relaxed = sequence(6, 16, 4);
    assert!(admitted(&mut scheduler, &relaxed, "准入"));
    assert_eq!(
        scheduler.advance(&mut relaxed).expect("生成"),
        Progress::Running
    );
}

#[test]
fn a_done_sequence_is_not_admitted() {
    let mut scheduler = Scheduler::new(pool(2, 16));
    let mut sequence = sequence(7, 0, 1);
    assert!(admitted(&mut scheduler, &sequence, "准入"));
    assert_eq!(
        scheduler.advance(&mut sequence).expect("唯一一步"),
        Progress::Done
    );
    assert!(scheduler.finish(7).expect("结束"));
    assert_eq!(
        scheduler.admit(&sequence).unwrap_err().code(),
        Code::Invalid,
        "结束的序列再准入只会白占页"
    );
}

/// 随机准入 / 推进 / 结束，任何一步之后账都要对得上：
///
/// - 页池占用 = 各页表长度之和，空闲 + 占用恒等于页总数；
/// - 每张页表覆盖当前长度，最多短一页（下个 token 那一页）；
/// - 报错的推进不改变 `produced`，`Blocked` 也不改变页池。
#[test]
fn random_batches_keep_the_page_accounts_consistent() {
    let mut rng = Rng::new(0x5eed);
    let mut scheduler = Scheduler::new(pool(6, 4));
    let mut sequences: Vec<Sequence> = (0..4u64)
        .map(|id| {
            let prompt_len = rng.below(8) as usize;
            let max_new = 1 + rng.below(6) as usize;
            let prompt = vec![0u32; prompt_len];
            Sequence::new(id, &prompt, max_new, 4, crate::exec::Stream::new(0)).expect("序列")
        })
        .collect();

    for _ in 0..400 {
        let index = rng.below(sequences.len() as u32) as usize;
        let id = sequences[index].id();
        match rng.below(3) {
            0 => {
                let admitted = scheduler.pages_of(id).is_some();
                let done = sequences[index].is_done();
                match scheduler.admit(&sequences[index]) {
                    Err(error) => {
                        assert_eq!(error.code(), Code::Invalid);
                        assert!(admitted || done, "只有重复准入与已结束的序列才是错误");
                    }
                    Ok(grown) => {
                        assert!(!admitted && !done, "不该准入的序列被准入");
                        if grown.is_some() {
                            assert!(scheduler.pages_of(id).is_some(), "准入成功必须占页");
                        }
                    }
                }
            }
            1 => {
                let before = sequences[index].produced();
                let admitted = scheduler.pages_of(id).is_some();
                let outcome = scheduler.advance(&mut sequences[index]);
                // 批级信号必须与实际结局同源：`blocked()` 就是刚刚返回 Blocked 的那个集合。
                assert_eq!(
                    scheduler.blocked().contains(&id),
                    outcome == Ok(Progress::Blocked),
                    "序列 {id} 的批级阻塞信号与单步结局不一致"
                );
                match outcome {
                    Err(error) => {
                        assert_eq!(error.code(), Code::Invalid);
                        assert!(
                            !admitted || sequences[index].is_done(),
                            "已准入且未结束的推进不该报错"
                        );
                        assert_eq!(sequences[index].produced(), before, "报错不能推进");
                    }
                    Ok(Progress::Blocked) => {
                        assert!(admitted);
                        assert_eq!(sequences[index].produced(), before, "阻塞不能推进");
                    }
                    Ok(progress) => {
                        assert_eq!(sequences[index].produced(), before + 1);
                        let expected = if sequences[index].is_done() {
                            Progress::Done
                        } else {
                            Progress::Running
                        };
                        assert_eq!(progress, expected);
                    }
                }
            }
            _ => {
                let had = scheduler.pages_of(id).is_some();
                assert_eq!(scheduler.finish(id).expect("归还"), had);
            }
        }

        let mut sum = 0;
        for sequence in &sequences {
            let Some(table) = scheduler.pages_of(sequence.id()) else {
                continue;
            };
            sum += table.len();
            assert!(table.len() <= sequence.pages_needed());
            assert!(
                sequence.pages_needed() - table.len() <= 1,
                "页表落后超过一页：序列 {} 长度 {} 需要 {} 页，只有 {} 页",
                sequence.id(),
                sequence.len(),
                sequence.pages_needed(),
                table.len()
            );
        }
        assert_eq!(scheduler.cache().in_use(), sum, "占用与页表不一致");
        assert_eq!(
            scheduler.cache().free_pages() + sum,
            scheduler.cache().total()
        );
    }
}
