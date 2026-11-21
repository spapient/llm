//! 页池上的页内数据：给页表要页、写行、读行。

use super::*;

/// 页池 + 页里的数据：一页 = `page_size` 行、每行 `width` 个 `f32`。
///
/// 行宽是"一个 token 位置在模型里的 KV 宽度"（例如 `2 · kv_heads · head_dim`），由装载模型的
/// 一侧算好传进来：页池不知道层与头，也不该替模型猜。多层模型用 [`Pager::row_mut`] 拿整行，
/// 再按自己的层偏移切片——层的偏移表归模型，页池不复制第二份。
///
/// 布局是 `[页][页内偏移][行内元素]`，与 [`Slot`] 的寻址算术一致：一页的元素连续存放，
/// 所以"第几页第几个 token"只做一次乘加，不需要页内链表。整池在构造时一次分配完（页池就是
/// 这些元素），之后借出与归还都不再碰分配器——KV 不该在生成到一半时因为一次 `Vec` 扩容而抖动。
///
/// 数据是 `Vec<f32>`：本类型是 CPU 页池。设备后端的同类对象在边界 crate，接口形状相同
/// （[`Slot`] → 一段可写映射），本层不引入设备句柄。
#[derive(Debug, Clone, PartialEq)]
pub struct Pager {
    cache: Cache,
    width: usize,
    data: Vec<f32>,
}

impl Pager {
    /// 由页池与行宽构造；整池一次分配并清零。
    ///
    /// # Errors
    ///
    /// 行宽为 0（那是"不存 KV"，不该有一个静默的空池）、元素总数溢出、或内存不足。
    pub fn new(cache: Cache, width: usize) -> Result<Self> {
        if width == 0 {
            return Err(
                Error::new(Code::Invalid, "KV 行宽不能为 0：页池需要能放下至少一个元素")
                    .with("engine::kv::Pager::new"),
            );
        }
        let elements = cache
            .total()
            .checked_mul(cache.page_size())
            .and_then(|positions| positions.checked_mul(width))
            .ok_or_else(|| {
                Error::new(
                    Code::Invalid,
                    format!(
                        "KV 页池容量溢出：{} 页 × {} 行 × 行宽 {width}",
                        cache.total(),
                        cache.page_size()
                    ),
                )
                .with("engine::kv::Pager::new")
            })?;
        let mut data = Vec::new();
        data.try_reserve_exact(elements).map_err(|_| {
            Error::new(
                Code::Invalid,
                format!("KV 页池需要 {elements} 个 f32：超出可用内存"),
            )
            .with("engine::kv::Pager::new")
        })?;
        data.resize(elements, 0.0);
        Ok(Self { cache, width, data })
    }

    /// 行宽（一个 token 位置在模型里的 KV 宽度）。
    pub fn width(&self) -> usize {
        self.width
    }

    /// 借出一页；池空时返回 `None`。
    pub fn alloc(&mut self) -> Option<Page> {
        self.cache.alloc()
    }

    /// 采纳一页给另一张页表共享：只递增引用计数，不动页里的数据。
    ///
    /// 与 [`Pager::alloc`] 分开：新页从池里取（计数置 1），前缀共享页只经这里 +1。
    ///
    /// # Errors
    ///
    /// 页越界，或不在借出状态。
    pub fn share(&mut self, page: Page) -> Result<()> {
        self.cache.share(page)
    }

    /// 归还一页；只有这是最后一个持有者时才**清零它的数据**。
    ///
    /// 清零点只有这一处：借出一页时不做第二次清零，因为页只可能经这里回到池里。先让页池
    /// 判定越界与重复归还，再清数据——归还失败时那页仍然是活的，数据不能动；仍被共享时
    /// 前缀行必须保持可读，也不能清零。
    ///
    /// # Errors
    ///
    /// 页不在池里，或不在借出状态。
    pub fn release(&mut self, page: Page) -> Result<()> {
        let last = self.cache.release(page)?;
        if last {
            self.clear(page);
        }
        Ok(())
    }

    /// 第 `slot` 个位置的那一行（只读）。
    ///
    /// # Errors
    ///
    /// 页已归还/不存在，或页内偏移超出页宽。
    pub fn row(&self, slot: Slot) -> Result<&[f32]> {
        let span = self.span(slot)?;
        Ok(&self.data[span])
    }

    /// 第 `slot` 个位置的那一行（可写）：多层模型按自己的层偏移切它。
    ///
    /// # Errors
    ///
    /// 同 [`Pager::row`]。
    pub fn row_mut(&mut self, slot: Slot) -> Result<&mut [f32]> {
        let span = self.span(slot)?;
        Ok(&mut self.data[span])
    }

    /// 把一整行写成 `values`：长度必须恰好等于行宽。
    ///
    /// # Errors
    ///
    /// 长度不符（半行会让同一页里的行错位），或同 [`Pager::row`]。
    pub fn write(&mut self, slot: Slot, values: &[f32]) -> Result<()> {
        if values.len() != self.width {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "KV 行宽 {} 与实际写入的 {} 个元素不符",
                    self.width,
                    values.len()
                ),
            )
            .with("engine::kv::Pager::write"));
        }
        self.row_mut(slot)?.copy_from_slice(values);
        Ok(())
    }

    /// 页池（只读）：页宽与占用由它回答，数据不在这里。
    pub fn cache(&self) -> &Cache {
        &self.cache
    }

    /// 页宽（每页多少 token 位置）。
    pub fn page_size(&self) -> usize {
        self.cache.page_size()
    }

    /// 页总数。
    pub fn total(&self) -> usize {
        self.cache.total()
    }

    /// 已借出页数。
    pub fn in_use(&self) -> usize {
        self.cache.in_use()
    }

    /// 空闲页数。
    pub fn free_pages(&self) -> usize {
        self.cache.free_pages()
    }

    /// 一行的元素区间；页必须活着，偏移必须落在页内。
    fn span(&self, slot: Slot) -> Result<std::ops::Range<usize>> {
        if !self.cache.is_live(slot.page()) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "页 {} 不在借出状态：它已经归还（或从未分配），这一行不再属于任何序列",
                    slot.page().id()
                ),
            )
            .with("engine::kv::Pager"));
        }
        if slot.offset() >= self.cache.page_size() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "页内偏移 {} 超出页宽 {}",
                    slot.offset(),
                    self.cache.page_size()
                ),
            )
            .with("engine::kv::Pager"));
        }
        let start = self.start(slot.page(), slot.offset());
        Ok(start..start + self.width)
    }

    /// 线性起点：`(页号 × 页宽 + 页内偏移) × 行宽`。
    ///
    /// 只在这里写一次：`span` 与清零共用它，两处各写一遍就会在"清到哪"与"读到哪"之间漂。
    fn start(&self, page: Page, offset: usize) -> usize {
        (page.id() as usize * self.cache.page_size() + offset) * self.width
    }

    /// 把一页的全部行清零（归还时调用）。
    fn clear(&mut self, page: Page) {
        let start = self.start(page, 0);
        let end = start + self.cache.page_size() * self.width;
        self.data[start..end].fill(0.0);
    }
}
