use super::*;
use crate::kv::Cache;

/// `pages` 页、每页 `page_size` 个 token 的页池（行宽 1：这几条测的是页的账，不是行里的数）。
fn pool(pages: usize, page_size: usize) -> Pager {
    Pager::new(Cache::new(pages, page_size).expect("页池"), 1).expect("行宽")
}

/// 登记一块并交出该页；返回那一页与登记是否新发生。
fn publish(pager: &mut Pager, radix: &mut Radix, tokens: &[u32], block: usize) -> (Page, bool) {
    let page = pager.alloc().expect("还有空闲页");
    let width = pager.page_size();
    let fresh = radix
        .publish(pager, tokens, width, block, page)
        .expect("登记");
    pager.release(page).expect("表里那份归还");
    (page, fresh)
}

/// 树自己持有页：请求结束归还自己那份之后，页仍活着并还能被命中。
#[test]
fn a_published_block_outlives_the_table_that_wrote_it() {
    let mut pager = pool(2, 2);
    let tokens = [7, 8, 9];
    let mut radix = Radix::new();
    let page = pager.alloc().expect("页");
    assert!(
        radix
            .publish(&mut pager, &tokens, 2, 0, page)
            .expect("登记")
    );
    assert_eq!(pager.cache().reference_count(page), 2, "树多持一份");
    pager.release(page).expect("请求结束");
    assert!(pager.cache().is_live(page), "树还持有，页不能回池");

    assert_eq!(radix.longest(&tokens, 2), vec![page]);
    assert_eq!(
        radix.longest(&[7, 8, 1], 2),
        vec![page],
        "第 0 块相同就命中"
    );
    assert!(radix.longest(&[7, 1, 9], 2).is_empty(), "块内不同不命中");
}

/// 只按整块命中：不满一页的尾巴、不同的第 2 块都在它的前一块上停住。
#[test]
fn the_match_stops_at_full_blocks() {
    let mut pager = pool(4, 2);
    let tokens = [1, 2, 3, 4, 5, 6];
    let mut radix = Radix::new();
    let mut pages = Vec::new();
    for block in 0..2 {
        let (page, fresh) = publish(&mut pager, &mut radix, &tokens, block);
        assert!(fresh);
        pages.push(page);
    }
    assert_eq!(radix.len(), 2);
    assert_eq!(radix.longest(&tokens, 2), pages);
    assert_eq!(
        radix.longest(&tokens[..5], 2),
        pages,
        "尾巴不满一页照命中前两块"
    );
    assert_eq!(radix.longest(&[1, 2, 3, 4, 7, 8], 2), pages, "第 2 块不同");
    assert!(radix.longest(&[1], 2).is_empty(), "不足一块不共享");
    assert!(radix.longest(&tokens, 0).is_empty(), "页宽为 0 无从切块");

    let page = pager.alloc().expect("页");
    assert_eq!(
        radix
            .publish(&mut pager, &tokens[..5], 2, 2, page)
            .unwrap_err()
            .code(),
        Code::Invalid,
        "不满一页的第 2 块不许登记"
    );
}

/// 重复登记同一块：不加第二份引用，页仍是那一张。
#[test]
fn reusing_a_block_does_not_take_a_second_reference() {
    let mut pager = pool(2, 2);
    let tokens = [1, 2, 3, 4];
    let mut radix = Radix::new();
    let (page, fresh) = publish(&mut pager, &mut radix, &tokens, 0);
    assert!(fresh);
    assert!(
        !publish(&mut pager, &mut radix, &tokens, 0).1,
        "同一块第二次不登记"
    );
    assert_eq!(pager.cache().reference_count(page), 1, "只留树那一份");
    assert_eq!(radix.len(), 1);
}

/// 父块不在树上就不登记：路径不许指向不存在的 KV。
#[test]
fn a_broken_parent_chain_is_not_registered() {
    let mut pager = pool(2, 2);
    let tokens = [1, 2, 3, 4];
    let mut radix = Radix::new();
    let page = pager.alloc().expect("页");
    assert!(
        !radix
            .publish(&mut pager, &tokens, 2, 1, page)
            .expect("不登记而不是报错"),
        "第 0 块不在树上，第 1 块不登记"
    );
    assert_eq!(pager.cache().reference_count(page), 1, "没加引用");
    assert!(radix.is_empty());
}

/// 驱逐先摘最久没被命中的叶，页真的回池。
#[test]
fn an_eviction_takes_the_least_recently_used_leaf() {
    let mut pager = pool(4, 2);
    let mut radix = Radix::new();
    let first = [1, 2, 3, 4];
    assert!(publish(&mut pager, &mut radix, &first, 0).1);
    assert!(publish(&mut pager, &mut radix, &first, 1).1);
    let other = [5, 6];
    let (stale, _) = publish(&mut pager, &mut radix, &other, 0);

    assert_eq!(radix.longest(&first, 2).len(), 2, "命中会刷新这条路径");
    assert_eq!(radix.evict(&mut pager, 1).expect("腾一页"), 1);
    assert_eq!(
        pager.cache().reference_count(stale),
        0,
        "被命中最少的叶先走"
    );
    assert_eq!(radix.longest(&first, 2).len(), 2, "常用路径没被摘");
}
