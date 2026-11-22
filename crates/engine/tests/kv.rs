//! 页缓存的账目夹具：分配、复用、引用计数与前缀复用。

use super::*;
use diagnostic::Code;

#[test]
fn pages_are_handed_out_in_order_and_reused() {
    let mut cache = Cache::new(2, 4).expect("页池");
    let first = cache.alloc().expect("第一页");
    let second = cache.alloc().expect("第二页");
    assert_eq!((first.id(), second.id()), (0, 1));
    assert!(cache.alloc().is_none(), "池空必须返回 None");
    cache.release(first).expect("归还");
    assert_eq!(cache.alloc().map(|page| page.id()), Some(0));
}

#[test]
fn double_release_and_out_of_range_are_rejected() {
    let mut cache = Cache::new(1, 4).expect("页池");
    let page = cache.alloc().expect("分配");
    cache.release(page).expect("归还");
    assert_eq!(cache.release(page).unwrap_err().code(), Code::Invalid);
    assert_eq!(
        cache.release(Page::new(9)).unwrap_err().code(),
        Code::Invalid
    );
}

#[test]
fn a_table_refuses_to_drop_pages_silently() {
    let mut table = Table::new(1).expect("页表");
    assert!(table.is_empty());
    table.push(Page::new(0)).expect("未满");
    assert!(table.is_full());
    assert!(table.push(Page::new(1)).is_err(), "满页表必须报错");
    assert_eq!(table.get(0), Some(0));
    assert_eq!(table.get(1), None);
    assert_eq!(Table::new(0).unwrap_err().code(), Code::Invalid);
}

#[test]
fn a_zero_sized_pool_is_rejected() {
    assert_eq!(Cache::new(0, 16).unwrap_err().code(), Code::Invalid);
    assert_eq!(Cache::new(4, 0).unwrap_err().code(), Code::Invalid);
}

#[test]
fn a_block_shape_matches_the_page_width() {
    let cache = Cache::new(4, 16).expect("页池");
    assert_eq!(cache.block_shape(2).dims(), [2, 16]);
    assert_eq!(cache.total(), 4);
    assert_eq!(cache.free_pages(), 4);
}

#[test]
fn a_table_can_grow_but_never_past_its_capacity() {
    let mut table = Table::new(1).expect("页表");
    table.push(Page::new(0)).expect("未满");
    assert!(table.push(Page::new(1)).is_err(), "满页表仍然拒绝追加");
    table.grow(1).expect("抬高容量");
    assert_eq!(table.capacity(), 2);
    table.push(Page::new(1)).expect("容量已抬高");
    assert!(table.push(Page::new(2)).is_err(), "抬容量不能顺手放开越界");
    table.grow(0).expect("零页增长是恒等");
    assert_eq!(table.capacity(), 2);
}

/// 位图判定要同时认得出"从没分配过"与"已经还过"：两者都不是合法的归还。
#[test]
fn releasing_a_page_that_was_never_allocated_is_rejected() {
    let mut cache = Cache::new(2, 4).expect("页池");
    assert_eq!(
        cache.release(Page::new(1)).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(cache.in_use(), 0, "失败不能改变占用");
    assert_eq!(cache.free_pages(), 2, "失败不能把页塞进空闲表");
}

/// 稳态：整池分配、整池归还重复多轮，账目与页编号都不漂。
#[test]
fn allocate_and_release_cycles_keep_the_accounts_balanced() {
    let mut cache = Cache::new(8, 4).expect("页池");
    let mut held = Vec::new();
    for round in 0..1000 {
        while let Some(page) = cache.alloc() {
            held.push(page);
        }
        assert_eq!(
            (cache.in_use(), cache.free_pages()),
            (8, 0),
            "第 {round} 轮"
        );
        let mut ids: Vec<u32> = held.iter().map(Page::id).collect();
        ids.sort_unstable();
        assert_eq!(
            ids,
            (0..8).collect::<Vec<u32>>(),
            "第 {round} 轮：同一个页不能发两次"
        );
        for page in held.drain(..) {
            cache.release(page).expect("归还");
        }
        assert_eq!(
            (cache.in_use(), cache.free_pages()),
            (0, 8),
            "第 {round} 轮"
        );
    }
    assert_eq!(cache.total(), 8, "页总数与分配历史无关");
}

/// 寻址是唯一的除法：第 `idx` 个 token 落在 `idx / page_size` 页的 `idx % page_size` 格。
#[test]
fn a_token_idx_maps_to_one_page_and_one_offset() {
    let mut table = Table::new(4).expect("页表");
    for id in [7, 3, 11] {
        table.push(Page::new(id)).expect("追加");
    }
    assert_eq!(
        table.slot(0, 4).expect("第 0 个"),
        Slot {
            page: Page::new(7),
            offset: 0
        }
    );
    assert_eq!(table.slot(5, 4).expect("第 5 个").page().id(), 3);
    assert_eq!(table.slot(5, 4).expect("第 5 个").offset(), 1);
    assert_eq!(table.slot(11, 4).expect("第 11 个").offset(), 3);
    // 页表只盖住已分配的 3 页：第 12 个 token 还没有落点。
    assert_eq!(table.slot(12, 4).unwrap_err().code(), Code::Invalid);
    assert_eq!(table.slot(0, 0).unwrap_err().code(), Code::Invalid);
}

/// 截断只摘页、不动容量：回滚一次生成之后页表还要长回去。
#[test]
fn truncating_returns_the_pages_it_dropped() {
    let mut table = Table::new(4).expect("页表");
    for id in [0, 1, 2] {
        table.push(Page::new(id)).expect("追加");
    }
    assert_eq!(
        table.truncate(1).expect("截断"),
        [Page::new(1), Page::new(2)]
    );
    assert_eq!(table.len(), 1);
    assert_eq!(table.capacity(), 4, "容量不变");
    assert!(table.truncate(2).is_err(), "不能截到比现有页数更多的位置");
    assert_eq!(table.truncate(1).expect("重复截断是恒等"), []);
}

/// 一页 `page_size` 行、每行 `width` 个元素：借出就有一块可写的行，归还就清零。
///
/// 这一条是页池存在的理由：`Cache` 只发页号时，"这页被复用"与"页里还留着上一条序列的
/// 字节"可以同时成立，而两种错都不会让任何断言失败——只有读回旧值时才暴露。
#[test]
fn a_pager_writes_rows_and_clears_them_on_release() {
    let mut pager = Pager::new(Cache::new(2, 2).expect("页池"), 3).expect("行宽");
    let page = pager.alloc().expect("第一页");
    let first = Slot { page, offset: 0 };
    let second = Slot { page, offset: 1 };
    pager.write(first, &[1.0, 2.0, 3.0]).expect("写第一行");
    pager.write(second, &[4.0, 5.0, 6.0]).expect("写第二行");
    assert_eq!(pager.row(first).expect("读第一行"), [1.0, 2.0, 3.0]);
    assert_eq!(pager.row(second).expect("读第二行"), [4.0, 5.0, 6.0]);

    pager.release(page).expect("归还");
    assert_eq!(pager.in_use(), 0);
    let again = pager.alloc().expect("同一页再借出来");
    assert_eq!(again, page, "空闲表按页号复用");
    assert_eq!(
        pager.row(first).expect("读第一行"),
        [0.0, 0.0, 0.0],
        "复用页不能带出上一条序列的字节"
    );
    assert_eq!(pager.row(second).expect("读第二行"), [0.0, 0.0, 0.0]);
}

/// 行在两个维度上都校验：页必须是活的，偏移必须落在页内，长度必须恰好是行宽。
#[test]
fn a_pager_refuses_stale_slots_and_partial_rows() {
    let mut pager = Pager::new(Cache::new(2, 2).expect("页池"), 3).expect("行宽");
    let page = pager.alloc().expect("页");
    let row = Slot { page, offset: 0 };
    assert_eq!(
        pager.write(row, &[1.0, 2.0]).unwrap_err().code(),
        Code::Invalid,
        "半行会让同一页里的后续行错位"
    );
    assert_eq!(
        pager.row(Slot { page, offset: 2 }).unwrap_err().code(),
        Code::Invalid,
        "偏移等于页宽已经越界"
    );
    pager.release(page).expect("归还");
    assert_eq!(
        pager.row(row).unwrap_err().code(),
        Code::Invalid,
        "已归还的页不是合法地址：这张页可能已经属于别人"
    );
    assert_eq!(
        pager.release(page).unwrap_err().code(),
        Code::Invalid,
        "重复归还由页池判定"
    );
}

/// 行宽为零是"不存 KV"，不该有一个静默的空池。
#[test]
fn a_zero_width_pager_is_rejected() {
    let cache = Cache::new(2, 4).expect("页池");
    let error = Pager::new(cache, 0).expect_err("行宽为零");
    assert_eq!(error.code(), Code::Invalid);
}

/// 整池一次分配：容量与页池相等，跨页的行不共享元素。
#[test]
fn pages_do_not_share_rows() {
    let mut pager = Pager::new(Cache::new(2, 2).expect("页池"), 1).expect("行宽");
    let first = pager.alloc().expect("第一页");
    let second = pager.alloc().expect("第二页");
    pager
        .write(
            Slot {
                page: first,
                offset: 1,
            },
            &[7.0],
        )
        .expect("写");
    assert_eq!(
        pager
            .row(Slot {
                page: second,
                offset: 1
            })
            .expect("读"),
        [0.0],
        "另一页的同一格不受影响"
    );
    assert_eq!((pager.total(), pager.page_size()), (2, 2));
}

/// 两表共引一页：第一张表归还后页仍存活、前缀行未被清零；最后一个持有者归还才回池并清零。
#[test]
fn a_shared_page_survives_the_first_table_releasing_it() {
    let mut pager = Pager::new(Cache::new(2, 2).expect("页池"), 3).expect("行宽");
    let page = pager.alloc().expect("第一页");
    let slot = Slot { page, offset: 0 };
    pager.write(slot, &[1.0, 2.0, 3.0]).expect("写前缀行");

    pager.share(page).expect("另一张表采纳前缀页");
    assert_eq!(pager.cache().reference_count(page), 2);

    pager.release(page).expect("第一张表归还");
    assert!(pager.cache().is_live(page), "还有共享者，页仍存活");
    assert_eq!(
        pager.row(slot).expect("读"),
        [1.0, 2.0, 3.0],
        "共享期内前缀行只读，不能清零"
    );
    assert_eq!(pager.in_use(), 1);

    pager.release(page).expect("最后一个持有者归还");
    assert!(!pager.cache().is_live(page));
    assert_eq!(pager.in_use(), 0);
    let again = pager.alloc().expect("同一页再借出来");
    assert_eq!(again, page, "空闲表按页号复用");
    assert_eq!(
        pager.row(slot).expect("读"),
        [0.0, 0.0, 0.0],
        "归零时才清零"
    );
}

/// 引用计数口径：共享两次、释放两次后仍剩一个持有者；归零时才回空闲池。
#[test]
fn reference_count_is_accounted_across_share_and_release() {
    let mut cache = Cache::new(2, 4).expect("页池");
    let page = cache.alloc().expect("分配");
    assert_eq!(cache.reference_count(page), 1);

    cache.share(page).expect("第一次共享");
    cache.share(page).expect("第二次共享");
    assert_eq!(cache.reference_count(page), 3);
    assert_eq!((cache.in_use(), cache.free_pages()), (1, 1));

    cache.release(page).expect("第一次归还");
    cache.release(page).expect("第二次归还");
    assert_eq!(cache.reference_count(page), 1);
    assert!(cache.is_live(page), "仍有一个持有者，页不能回池");

    cache.release(page).expect("最后一次归还");
    assert_eq!(cache.reference_count(page), 0);
    assert_eq!((cache.in_use(), cache.free_pages()), (0, 2));
}

/// 共享入口同样校验越界与非存活页；失败不能改账目。
#[test]
fn sharing_an_out_of_range_or_dead_page_is_rejected() {
    let mut cache = Cache::new(1, 4).expect("页池");
    assert_eq!(cache.share(Page::new(9)).unwrap_err().code(), Code::Invalid);
    let page = cache.alloc().expect("分配");
    cache.release(page).expect("先归还");
    assert_eq!(cache.share(page).unwrap_err().code(), Code::Invalid);
    assert_eq!(cache.in_use(), 0, "共享失败不动占用");
    assert_eq!(cache.free_pages(), 1, "共享失败不把页塞进空闲表");
}

/// 多层模型共用一行：每层切自己那段，层与层不重叠。
///
/// 页池不知道层，所以"第 `l` 层从哪个元素开始"只能由模型给偏移——这条测试锁住的是
/// "给整行、按偏移切"这套用法本身，而不是某个层宽度的具体数值。
#[test]
fn layers_slice_their_own_part_of_one_row() {
    let mut pager = Pager::new(Cache::new(1, 2).expect("页池"), 4).expect("行宽");
    let page = pager.alloc().expect("页");
    let slot = Slot { page, offset: 1 };

    pager.row_mut(slot).expect("整行")[..2].copy_from_slice(&[1.0, 2.0]);
    pager.row_mut(slot).expect("整行")[2..].copy_from_slice(&[3.0, 4.0]);

    assert_eq!(
        pager.row(slot).expect("读"),
        [1.0, 2.0, 3.0, 4.0],
        "两段各写各的，读回时拼成一行"
    );
}

/// E12 随机历史访问 + 块表 O(1)（`#[ignore]` 实测：
/// `cargo test -p llm-engine --lib -- e12_random_access --release --ignored --nocapture`）。
///
/// 冻结工作集 4096 页 × 16 token/页 = 65536 历史 token；确定性生成 10 万次随机访问，
/// `Table::slot` 只做 `idx / page_size` + `idx % page_size` + 一次 `Vec` 下标（源码即
/// O(1)，不随历史长度遍历），这里再用不同历史长度下的查表墙钟做经验佐证。
#[test]
#[ignore = "E12 块表基准：cargo test -p llm-engine --lib -- e12_random_access --release --ignored --nocapture"]
fn e12_random_access() {
    let page_size = 16usize;
    let pages = 4096usize;
    let history = pages * page_size;
    let mut table = Table::new(pages).expect("页表");
    for id in 0..pages as u32 {
        table.push(Page::new(id)).expect("追加");
    }

    // 确定性 xorshift64 随机访问集（冻结种子，可复现）。
    let mut seed = 0xdead_beef_cafe_1234u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    let samples = 100_000usize;
    let mut hits = 0usize;
    for _ in 0..samples {
        let idx = (next() % history as u64) as usize;
        let slot = table.slot(idx, page_size).expect("冻结工作集内必有页");
        let expected_page = (idx / page_size) as u32;
        let expected_offset = idx % page_size;
        if slot.page().id() == expected_page && slot.offset() == expected_offset {
            hits += 1;
        }
    }
    let recall = hits as f64 / samples as f64;
    println!(
        "E12 随机访问 recall={recall:.6}（{hits}/{samples}，history={history}，page_size={page_size}）"
    );

    // 操作计数佐证：不同历史长度下 slot 的墙钟（每点取 min），应不随页数增长。
    for page_count in [1024usize, 4096, 16384, 65536] {
        let mut big = Table::new(page_count).expect("页表");
        for id in 0..page_count as u32 {
            big.push(Page::new(id)).expect("追加");
        }
        let reps = 1_000_000usize;
        let idx = page_count * page_size - 1; // 最远端 token，也不该更慢
        for _ in 0..3 {
            big.slot(idx, page_size).expect("有页");
        }
        let start = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(big.slot(idx, page_size).expect("有页"));
        }
        let per = start.elapsed().as_secs_f64() / reps as f64;
        println!(
            "E12 slot 查表 pages={page_count:<6} history={:<8} 每次={:.3} ns",
            page_count * page_size,
            per * 1e9
        );
    }
}
