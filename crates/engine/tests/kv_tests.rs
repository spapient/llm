use super::*;

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

/// 寻址是唯一的除法：第 `index` 个 token 落在 `index / page_size` 页的 `index % page_size` 格。
#[test]
fn a_token_index_maps_to_one_page_and_one_offset() {
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
