//! 满块前缀树：[`Radix`] 把「写满的一页」登记成一条边，跨请求复用同一段前缀的 KV。

use diagnostic::{Code, Error, Result};

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
#[path = "../tests/radix.rs"]
mod tests;

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
        let mut pages = Vec::new();
        let mut at: Option<usize> = None;
        let mut block = 0;
        let mut path = Vec::new();
        while let Some(span) = block_tokens(tokens, page_size, block) {
            let Some(index) = self.child(at, span) else {
                break;
            };
            path.push(index);
            at = Some(index);
            block += 1;
        }
        self.clock += 1;
        let tick = self.clock;
        for index in path {
            if let Some(block) = self.nodes[index].as_mut() {
                block.tick = tick;
                pages.push(block.page);
            }
        }
        pages
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
            let Some(index) = self.least_leaf() else {
                break;
            };
            freed += self.drop_subtree(pager, index)?;
        }
        Ok(freed)
    }

    /// 第 `block` 块的父节点：`Linked(Some(i))` 是树里的父块，`Linked(None)` 是根（第 0 块）；
    /// `Broken` 表示路径中间有块不在树上，这一块不登记。
    fn path(&self, tokens: &[u32], page_size: usize, block: usize) -> Result<Path> {
        let mut at: Option<usize> = None;
        for level in 0..block {
            let span = span_of(tokens, page_size, level)?;
            let Some(index) = self.child(at, span) else {
                return Ok(Path::Broken);
            };
            at = Some(index);
        }
        Ok(Path::Linked(at))
    }

    /// `at` 下面的孩子；`at` 为 `None` 时是根的孩子。
    fn children(&self, at: Option<usize>) -> &[usize] {
        match at {
            None => &self.root,
            Some(index) => self
                .nodes
                .get(index)
                .and_then(Option::as_ref)
                .map(|block| block.children.as_slice())
                .unwrap_or_default(),
        }
    }

    /// 在 `at` 下面找块内 token 相同的孩子。
    fn child(&self, at: Option<usize>, tokens: &[u32]) -> Option<usize> {
        self.children(at)
            .iter()
            .copied()
            .find(|&index| self.tokens_of(index) == Some(tokens))
    }

    /// 某个节点记的块内 token。
    fn tokens_of(&self, index: usize) -> Option<&[u32]> {
        self.nodes
            .get(index)
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
        let index = match self.nodes.iter().position(Option::is_none) {
            Some(index) => {
                self.nodes[index] = Some(block);
                index
            }
            None => {
                self.nodes.push(Some(block));
                self.nodes.len() - 1
            }
        };
        match at {
            Some(parent) => {
                if let Some(block) = self.nodes[parent].as_mut() {
                    block.children.push(index);
                }
            }
            None => self.root.push(index),
        }
    }

    /// 摘掉一棵子树（`index` 起），返回摘掉的块数；每块归还自己那一份页引用。
    fn drop_subtree(&mut self, pager: &mut Pager, index: usize) -> Result<usize> {
        let mut count = 0;
        let mut stack = vec![index];
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

    /// 把 `index` 从父（或根）的孩子表里去掉。
    fn unlink(&mut self, parent: Option<usize>, index: usize) {
        match parent {
            Some(parent) => {
                if let Some(block) = self.nodes[parent].as_mut() {
                    block.children.retain(|&child| child != index);
                }
            }
            None => self.root.retain(|&child| child != index),
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
            .map(|(index, _)| index)
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
    Error::new(Code::Invalid, "页宽不能为 0：前缀按整页切块").with("engine::radix::Radix")
}

fn beyond(block: usize, len: usize) -> Error {
    Error::new(
        Code::Invalid,
        format!("第 {block} 块超出现有 {len} 个 token：不满一页的尾巴不登记"),
    )
    .with("engine::radix::Radix")
}
