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
