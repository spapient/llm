//! 满块前缀树：[`Radix`] 把「写满的一页」登记成一条边，跨请求复用同一段前缀的 KV。

use crate::invalid;
use diagnostic::{Error, Result};

use crate::kv::{Page, Pager};

/// 满块前缀树：token 块 → 那一页的 KV，跨请求复用。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Radix {
    /// 节点槽位：`None` 是已释放的空位，复用它避免按下标无限增长。
    nodes: Vec<Option<Block>>,
    /// 根的孩子（第 0 块）。
    root: Vec<usize>,
    /// 单调时序戳：命中刷新、驱逐取最小。
    clock: u64,
}

/// 树里的一个满块：块内 token + 它那一页 + 它与父子的连接。
#[derive(Debug, Clone, PartialEq, Eq)]
struct Block {
    tokens: Vec<u32>,
    page: Page,
    parent: Option<usize>,
    children: Vec<usize>,
    tick: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::Cache;
    use diagnostic::Code;

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

    testkit::test_cases! {
    /// 树自己持有页：请求结束归还自己那份之后，页仍活着并还能被命中。
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
    }
}

impl Radix {
    /// 空树。
    pub fn new() -> Self {
        Self::default()
    }

    /// 已登记的块数。
    pub fn len(&self) -> usize {
        self.nodes.iter().flatten().count()
    }

    /// 树是否为空。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 最长满块前缀命中的页，按块序；页可直接采纳进新页表（调用方按 `share` 加计数）。
    pub fn longest(&mut self, tokens: &[u32], page_size: usize) -> Vec<Page> {
        let path = self.walk(tokens, page_size);
        self.clock += 1;
        let tick = self.clock;
        let mut pages = Vec::with_capacity(path.len());
        for idx in path {
            if let Some(block) = self.nodes[idx].as_mut() {
                block.tick = tick;
                pages.push(block.page);
            }
        }
        pages
    }

    /// 沿 token 路径一直往下走，返回命中的节点下标（按块序）。
    fn walk(&self, tokens: &[u32], page_size: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut at: Option<usize> = None;
        while let Some(span) = block_tokens(tokens, page_size, path.len()) {
            let Some(idx) = self.child(at, span) else {
                break;
            };
            path.push(idx);
            at = Some(idx);
        }
        path
    }

    /// 把一个写满的块登记进树：对它的页加一份引用，返回是否新登记。
    ///
    /// 父块必须先已在树上：某块能在树里当父，说明它的页当时被登记过；若父块已被驱逐，就
    /// 不再登记这一块（返回 `false`）——补一个没有页的中间节点会让路径指向不存在的 KV。
    /// 同一块重复登记返回 `false` 且不再加引用：已有那一页就是这条前缀的 KV。
    ///
    /// # Errors
    ///
    /// 页宽为 0、或 `block` 超出现有 token 长度时返回 `invalid`；页不在借出状态由
    /// [`Pager::share`] 报错。
    pub fn publish(
        &mut self,
        pager: &mut Pager,
        tokens: &[u32],
        page_size: usize,
        block: usize,
        page: Page,
    ) -> Result<bool> {
        match self.path(tokens, page_size, block)? {
            Path::Linked(at) => self.link(pager, tokens, page_size, block, at, page),
            Path::Broken => Ok(false),
        }
    }

    /// 把第 `block` 块挂到 `at` 之下：同一块已经在树上就不重复登记、也不多加引用。
    fn link(
        &mut self,
        pager: &mut Pager,
        tokens: &[u32],
        page_size: usize,
        block: usize,
        at: Option<usize>,
        page: Page,
    ) -> Result<bool> {
        let span = span_of(tokens, page_size, block)?;
        if self.child(at, span).is_some() {
            return Ok(false);
        }
        pager.share(page)?;
        self.attach(at, span.to_vec(), page);
        Ok(true)
    }

    /// 按 LRU 驱逐叶，直到腾出 `wanted` 页或再没有叶可摘；返回实际腾出的块数。
    ///
    /// 只摘叶：摘中间节点会让它下面仍被引用的块失去路径，那些页就再也命中不到、也再没人
    /// 能归还。摘掉一个叶之后它的父可能变成叶，下一轮自然轮到它。
    ///
    /// # Errors
    ///
    /// 页不在借出状态由 [`Pager::release`] 报错。
    pub fn evict(&mut self, pager: &mut Pager, wanted: usize) -> Result<usize> {
        let mut freed = 0;
        while freed < wanted {
            let Some(idx) = self.least_leaf() else {
                break;
            };
            freed += self.drop_subtree(pager, idx)?;
        }
        Ok(freed)
    }

    /// 第 `block` 块的父节点：`Linked(Some(i))` 是树里的父块，`Linked(None)` 是根（第 0 块）；
    /// `Broken` 表示路径中间有块不在树上，这一块不登记。
    fn path(&self, tokens: &[u32], page_size: usize, block: usize) -> Result<Path> {
        let mut at: Option<usize> = None;
        for level in 0..block {
            let span = span_of(tokens, page_size, level)?;
            let Some(idx) = self.child(at, span) else {
                return Ok(Path::Broken);
            };
            at = Some(idx);
        }
        Ok(Path::Linked(at))
    }

    /// `at` 下面的孩子；`at` 为 `None` 时是根的孩子。
    fn children(&self, at: Option<usize>) -> &[usize] {
        match at {
            None => &self.root,
            Some(idx) => self
                .nodes
                .get(idx)
                .and_then(Option::as_ref)
                .map_or_default(|block| block.children.as_slice()),
        }
    }

    /// 在 `at` 下面找块内 token 相同的孩子。
    fn child(&self, at: Option<usize>, tokens: &[u32]) -> Option<usize> {
        self.children(at)
            .iter()
            .copied()
            .find(|&idx| self.tokens_of(idx) == Some(tokens))
    }

    /// 某个节点记的块内 token。
    fn tokens_of(&self, idx: usize) -> Option<&[u32]> {
        self.nodes
            .get(idx)
            .and_then(Option::as_ref)
            .map(|block| block.tokens.as_slice())
    }

    /// 在 `at` 下面挂一个新块；空槽位优先复用。
    fn attach(&mut self, at: Option<usize>, tokens: Vec<u32>, page: Page) {
        let block = Block {
            tokens,
            page,
            parent: at,
            children: Vec::new(),
            tick: self.clock,
        };
        let idx = if let Some(idx) = self.nodes.iter().position(Option::is_none) {
            self.nodes[idx] = Some(block);
            idx
        } else {
            self.nodes.push(Some(block));
            self.nodes.len() - 1
        };
        match at {
            Some(parent) => {
                if let Some(block) = self.nodes[parent].as_mut() {
                    block.children.push(idx);
                }
            }
            None => self.root.push(idx),
        }
    }

    /// 摘掉一棵子树（`idx` 起），返回摘掉的块数；每块归还自己那一份页引用。
    fn drop_subtree(&mut self, pager: &mut Pager, idx: usize) -> Result<usize> {
        let mut count = 0;
        let mut stack = vec![idx];
        while let Some(current) = stack.pop() {
            let Some(block) = self.nodes.get_mut(current).and_then(Option::take) else {
                continue;
            };
            self.unlink(block.parent, current);
            pager.release(block.page)?;
            stack.extend(block.children);
            count += 1;
        }
        Ok(count)
    }

    /// 把 `idx` 从父（或根）的孩子表里去掉。
    fn unlink(&mut self, parent: Option<usize>, idx: usize) {
        match parent {
            Some(parent) => {
                if let Some(block) = self.nodes[parent].as_mut() {
                    block.children.retain(|&child| child != idx);
                }
            }
            None => self.root.retain(|&child| child != idx),
        }
    }

    /// 时序戳最小的叶。
    fn least_leaf(&self) -> Option<usize> {
        self.nodes
            .iter()
            .flatten()
            .enumerate()
            .filter(|(_, block)| block.children.is_empty())
            .min_by_key(|(_, block)| block.tick)
            .map(|(idx, _)| idx)
    }
}

/// 登记路径的结局：父块在树上，还是路径中途断了。
#[derive(Debug, Clone, Copy)]
pub(super) enum Path {
    /// 找到了父节点（`None` 表示根）。
    Linked(Option<usize>),
    /// 途中某块不在树上：这一块不登记，也不补没有页的中间节点。
    Broken,
}

/// 第 `block` 块的 token 切片；整块不存在（不足一页）时 `None`。`page_size` 为 0 也得到 `None`。
pub(super) fn block_tokens(tokens: &[u32], page_size: usize, block: usize) -> Option<&[u32]> {
    let start = block.checked_mul(page_size)?;
    let end = start.checked_add(page_size)?;
    if page_size == 0 || end > tokens.len() {
        return None;
    }
    Some(&tokens[start..end])
}

/// 第 `level` 块的 token 段：页宽为零与块越界是两种调用方错误，各有各的话。
pub(super) fn span_of(tokens: &[u32], page_size: usize, level: usize) -> Result<&[u32]> {
    if page_size == 0 {
        return Err(zero_page());
    }
    block_tokens(tokens, page_size, level).ok_or_else(|| beyond(level, tokens.len()))
}

fn zero_page() -> Error {
    invalid("engine::radix::Radix", "页宽不能为 0：前缀按整页切块")
}

fn beyond(block: usize, len: usize) -> Error {
    invalid(
        "engine::radix::Radix",
        format!("第 {block} 块超出现有 {len} 个 token：不满一页的尾巴不登记"),
    )
}
