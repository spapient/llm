use super::*;

impl Scheduler {
    /// 生成一个 token：先为它要落的那个位置预留页，再推进序列。
    ///
    /// 这是批里唯一的生成入口：`Sequence` 的计数器只在这里推进，所以"生成了却没有页账"
    /// 写不出来。池不够时返回 [`Progress::Blocked`] 且**不做任何改动**——序列没前进、页表
    /// 没长、页池没动，调用方可以等别的序列 [`Scheduler::finish`] 之后原样重试。
    ///
    /// # Errors
    ///
    /// 序列不在批里（没 `admit`、或已经 `finish`）、已经结束、或页宽与页池不一致时报错；
    /// 这些都属于调用方的状态错误，而不是"页不够"。
    pub fn advance(&mut self, sequence: &mut Sequence) -> Result<Progress> {
        let index = self.check_advance(sequence)?;
        if !self.extend(
            index,
            self.next_pages(sequence),
            "engine::Scheduler::advance",
        )? {
            return Ok(Progress::Blocked);
        }
        self.take_step(index, sequence)
    }

    /// 生成的前置判据：序列必须在批里、还没结束、页宽与页池一致。
    pub(super) fn check_advance(&self, sequence: &Sequence) -> Result<usize> {
        let index = self.position(sequence.id()).ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("序列 {} 不在批里：生成必须先 `admit`", sequence.id()),
            )
            .with("engine::Scheduler::advance")
        })?;
        if sequence.is_done() {
            return Err(
                Error::new(Code::Invalid, format!("序列 {} 已结束", sequence.id()))
                    .with("engine::Scheduler::advance"),
            );
        }
        self.check_width(sequence, "engine::Scheduler::advance")?;
        Ok(index)
    }

    /// 把第 `index` 条页表扩到 `needed` 页；池不够时返回 `false`，三者都不动。
    pub(super) fn extend(
        &mut self,
        index: usize,
        needed: usize,
        site: &'static str,
    ) -> Result<bool> {
        let extra = needed.saturating_sub(self.running[index].table.len());
        if extra == 0 {
            return Ok(true);
        }
        if self.pager.free_pages() < extra {
            return Ok(false);
        }
        self.running[index].table.grow(extra)?;
        Self::fill(&mut self.pager, &mut self.running[index].table, extra, site)?;
        Ok(true)
    }

    /// 走完一步并刷新批级派生值：推进计数器、重算「下一次要不要新页」、给出结局。
    pub(super) fn take_step(&mut self, index: usize, sequence: &mut Sequence) -> Result<Progress> {
        sequence.step()?;
        let needs_page = self.needs_page(sequence, &self.running[index].table);
        self.running[index].needs_page = needs_page;
        Ok(if sequence.is_done() {
            Progress::Done
        } else {
            Progress::Running
        })
    }

    /// 某个序列在批里的下标（插入顺序即批内顺序）。
    pub(super) fn position(&self, id: u64) -> Option<usize> {
        self.running.iter().position(|item| item.id == id)
    }

    /// 现在拿不到页、必须先等别人结束的序列（按插入顺序）。
    pub fn blocked(&self) -> Vec<u64> {
        if self.pager.free_pages() > 0 {
            return Vec::new();
        }
        self.running
            .iter()
            .filter(|item| item.needs_page)
            .map(|item| item.id)
            .collect()
    }

    /// 结束一个序列并归还它的页；返回它是否在批里。
    pub fn finish(&mut self, id: u64) -> Result<bool> {
        let Some(index) = self.position(id) else {
            return Ok(false);
        };
        let mut running = self.running.remove(index);
        for page in running.table.truncate(0)? {
            self.pager.release(page)?;
        }
        Ok(true)
    }

    /// 批里正在跑的序列数。
    pub fn running(&self) -> usize {
        self.running.len()
    }

    /// 某个序列占的页表。
    pub fn pages_of(&self, id: u64) -> Option<&crate::kv::Table> {
        self.position(id).map(|index| &self.running[index].table)
    }

    /// 页号分配器（只读）：服务层报"占了多少页"时用它，页里的数据不在这里。
    pub fn cache(&self) -> &crate::kv::Cache {
        self.pager.cache()
    }

    /// KV 页池（只读）：页号与页里的数据是同一个对象。
    pub fn pager(&self) -> &crate::kv::Pager {
        &self.pager
    }

    /// 页宽（每页多少 token）：调用方算"要几页"时不必再去拿页池。
    pub fn page_size(&self) -> usize {
        self.pager.page_size()
    }

    /// 批里某个序列的第 `token` 个位置落在哪一页的哪一格。
    ///
    /// 这是"KV 写到哪里"的唯一入口：页号归本层，页内偏移由页宽除出来。写 kernel 的一侧
    /// 因此不需要自己拿页表再算一遍偏移。
    ///
    /// # Errors
    ///
    /// 序列不在批里，或那个 token 还没有页（页表只盖住已经生成的位置）。
    pub fn slot(&self, id: u64, token: usize) -> Result<crate::kv::Slot> {
        let table = self.pages_of(id).ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("序列 {id} 不在批里：先 `admit` 再问页位置"),
            )
            .with("engine::Scheduler::slot")
        })?;
        table.slot(token, self.pager.page_size())
    }

    /// 把第 `token` 个位置的 KV 行写进页池（长度必须等于页池的行宽）。
    ///
    /// 位置必须已经有页：页表只盖住"已经生成到的地方"，所以这条路径写不出超前于生成的位置；
    /// 也不会写进一张已经归还的页（[`kv::Pager`] 会拒绝一个失效的 [`kv::Slot`]）。
    ///
    /// # Errors
    ///
    /// 序列不在批里、位置还没有页、行宽不符、或那页已归还。
    pub fn write_kv(&mut self, id: u64, token: usize, values: &[f32]) -> Result<()> {
        let slot = self.slot(id, token)?;
        self.pager.write(slot, values)
    }

    /// 把页表扩到盖住前 `rows` 行（1-based 行账：第 `n` 个 token 落在行 `n`）。
    ///
    /// 预填充一次写多行、不走 [`Scheduler::advance`] 的逐步预留，所以在写第一批行之前要先把页
    /// 一次要到。
    ///
    /// 返回「要到了没有」：`false` 是现在没页——调用方这一轮先不算，页紧是常态，不是别人的状态
    /// 错；只有序列不在批里、页宽为零这类调用方状态错误才是 `Err`。
    ///
    /// # Errors
    ///
    /// 序列不在批里或页宽为零（都算不出要几页）。
    pub fn reserve(&mut self, id: u64, rows: usize) -> Result<bool> {
        let Some(index) = self.position(id) else {
            return Err(Error::new(
                Code::Invalid,
                format!("序列 {id} 不在批里：先 `admit`（或它已经 `finish`）"),
            )
            .with("engine::Scheduler::reserve"));
        };
        let width = self.pager.page_size();
        if width == 0 {
            return Err(Error::new(Code::Invalid, "页宽为零：算不出要几页")
                .with("engine::Scheduler::reserve"));
        }
        let needed = pages_for(rows, width);
        self.extend(index, needed, "engine::Scheduler::reserve")
    }

    /// 读回第 `token` 个位置的 KV 行（只读）。
    ///
    /// # Errors
    ///
    /// 同 [`Scheduler::write_kv`] 的前两种情形。
    pub fn kv(&self, id: u64, token: usize) -> Result<&[f32]> {
        let slot = self.slot(id, token)?;
        self.pager.row(slot)
    }

    /// 下一次生成要不要新页：页表按 `len` 刚好够用，所以差数只可能是 0 或 1。
    pub(super) fn needs_page(&self, sequence: &Sequence, table: &crate::kv::Table) -> bool {
        if sequence.is_done() {
            return false;
        }
        self.next_pages(sequence) > table.len()
    }

    /// 写到下一个 token 需要的页数：这一步落在 `len()` 这个位置，所以页表要盖住 `len + 1` 个位置。
    pub(super) fn next_pages(&self, sequence: &Sequence) -> usize {
        pages_for(sequence.len() + 1, self.pager.page_size())
    }

    /// 页宽必须与页池一致：页数是按它算的，差一档就会少占一页。
    pub(super) fn check_width(&self, sequence: &Sequence, site: &'static str) -> Result<()> {
        if sequence.page_size() == self.pager.page_size() {
            return Ok(());
        }
        Err(Error::new(
            Code::Invalid,
            format!(
                "序列 {} 的页宽 {} 与页池的 {} 不一致：页数会算错",
                sequence.id(),
                sequence.page_size(),
                self.pager.page_size()
            ),
        )
        .with(site))
    }
}
