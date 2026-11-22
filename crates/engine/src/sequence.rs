//! 序列状态机：页数口径 [`pages_for`] 与单个请求的生命周期 [`Sequence`]。

use crate::invalid;
use diagnostic::Result;
use tensor::Shape;

/// 覆盖 `tokens` 个 token 需要多少页：向上取整，至少一页。
pub(crate) fn pages_for(tokens: usize, page_size: usize) -> usize {
    tokens.div_ceil(page_size).max(1)
}

/// [`Scheduler::advance`] 的结局："生成了一步"与"这一步没生成"必须能分开。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Progress {
    /// 生成了一个 token，序列还在跑。
    Running,
    /// 生成了一个 token，序列已结束（页仍然占着，直到 [`Scheduler::finish`]）。
    Done,
    /// 这一步需要的页不够：没有生成 token，序列与页表都没变，可以稍后原样重试。
    Blocked,
}

/// 序列状态：三个状态构成一条单向链路。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum State {
    /// 已提交、还没被准入。
    Waiting,
    /// 在批里，正在逐 token 生成。
    Running,
    /// 生成结束（达到上限或遇到停止符）。
    Done,
}

/// 一个推理请求的生命周期。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequence {
    id: u64,
    prompt_len: usize,
    /// 提示段已经算进页账的位置数（`0..=prompt_len`）；到了 `prompt_len` 才是「可以解码」。
    prefilled: usize,
    produced: usize,
    max_new: usize,
    page_size: usize,
    state: State,
    stream: crate::exec::Stream,
    /// 完整上下文：提示段在前、生成段在后；[`tokens`](Self::tokens) 切出后半段。
    all: Vec<u32>,
}

impl Sequence {
    /// 由请求参数构造：`prompt` 是提示段的 token id（整段留存，供嵌入查表与历史回看），
    /// `page_size` 来自页池，用于算需要多少页。
    pub fn new(
        id: u64,
        prompt: &[u32],
        max_new: usize,
        page_size: usize,
        stream: crate::exec::Stream,
    ) -> Result<Self> {
        if page_size == 0 {
            return Err(invalid("engine::Sequence::new", "页宽不能为 0"));
        }
        if max_new == 0 {
            return Err(invalid(
                "engine::Sequence::new",
                "max_new 不能为 0（否则请求一进来就结束）",
            ));
        }
        let prompt_len = prompt.len();
        // 上下文长度在整条生命周期里只增不减，`len()` 与 `len() + 1` 都不该溢出：一次校验
        // 比每处调用点各写一个 `saturating_add` 便宜，也避免“同一族量一处回绕、一处饱和”。
        if prompt_len.checked_add(max_new).is_none() {
            return Err(invalid(
                "engine::Sequence::new",
                format!("prompt_len {prompt_len} 加 max_new {max_new} 溢出：上下文长度装不下"),
            ));
        }
        Ok(Self {
            id,
            prompt_len,
            prefilled: 0,
            produced: 0,
            max_new,
            page_size,
            state: State::Waiting,
            stream,
            all: prompt.to_vec(),
        })
    }

    /// 请求编号。
    pub fn id(&self) -> u64 {
        self.id
    }

    /// 提示长度。
    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    /// 已生成 token 数。
    pub fn produced(&self) -> usize {
        self.produced
    }

    /// 提示段已经算进页账的位置数。
    pub fn prefilled(&self) -> usize {
        self.prefilled
    }

    /// 提示段是否已算完（可以直接解码）。
    pub fn prefilled_ready(&self) -> bool {
        self.prefilled >= self.prompt_len
    }

    /// 最多生成多少 token。
    pub fn max_new(&self) -> usize {
        self.max_new
    }

    /// 已经采样出的生成 token，按生成顺序（不含提示段）。
    pub fn tokens(&self) -> &[u32] {
        &self.all[self.prompt_len..]
    }

    /// 完整上下文：提示段在前、生成段在后。
    pub fn ctx(&self) -> &[u32] {
        &self.all
    }

    /// 记录一个已经采样出来的 token；**不**推进计数、也不碰页账。
    ///
    /// 只对同 crate 可见：[`Batch::emit`] 是"记账 + 推进"成对的唯一入口，绕过它单独记一笔
    /// 会让 `tokens` 与 `produced` 各说一套。上限也只有一处：`max_new`。
    ///
    /// # Errors
    ///
    /// `max_new` 已经记满时返回 `invalid`。
    pub(crate) fn push_token(&mut self, token: u32) -> Result<()> {
        if self.all.len() - self.prompt_len >= self.max_new {
            return Err(invalid(
                "engine::Sequence::push_token",
                format!(
                    "序列 {} 已经记满 {} 个生成 token：再多一个就超出上限",
                    self.id, self.max_new
                ),
            ));
        }
        self.all.push(token);
        Ok(())
    }

    /// 当前状态。
    pub fn state(&self) -> State {
        self.state
    }

    /// 提交所在的流。
    pub fn stream(&self) -> crate::exec::Stream {
        self.stream
    }

    /// 提前结束（采到停止符）：只改状态，不生成 token、也不碰页账。
    ///
    /// 与 `max_new` 到顶的区别只在计数：两种结束都不增不删页，页由 [`Scheduler::finish`] 归还。
    /// 已经结束的序列再 `stop` 是调用方的 bug——重复结束会让"为什么结束"变成两种说法。
    ///
    /// # Errors
    ///
    /// 序列已经结束时返回 `invalid`。
    pub fn stop(&mut self) -> Result<()> {
        if self.state == State::Done {
            return Err(invalid(
                "engine::Sequence::stop",
                format!("序列 {} 已结束", self.id),
            ));
        }
        self.state = State::Done;
        Ok(())
    }

    /// 推进一个 token；已经结束时再推进是调用方的 bug。
    pub(crate) fn step(&mut self) -> Result<()> {
        if self.state == State::Done {
            return Err(invalid("engine::Sequence::step", "序列已结束"));
        }
        self.state = State::Running;
        self.produced += 1;
        if self.produced >= self.max_new {
            self.state = State::Done;
        }
        Ok(())
    }

    /// 把提示段已算位置推进到 `to`；**不碰页**。
    ///
    /// 只对同 crate 可见：[`Batch::mark_prefilled`] 是"推进 + 登记块"成对的唯一入口，绕过它
    /// 单独推进会让前缀树少登记一个本可复用的块。
    ///
    /// # Errors
    ///
    /// `to` 超出 `0..=prompt_len` 或比当前已算位置更浅时返回 `invalid`。
    pub(crate) fn prefill_to(&mut self, to: usize) -> Result<()> {
        if to > self.prompt_len || to < self.prefilled {
            return Err(invalid(
                "engine::Sequence::prefill_to",
                format!(
                    "预填充位置 {to} 不在 {}..={} 之内（不许倒退、不许越过提示段）",
                    self.prefilled, self.prompt_len
                ),
            ));
        }
        self.prefilled = to;
        Ok(())
    }

    /// 页宽（与页池一致；准入与增长都按它算页数）。
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// 跑满 `max_new` 需要多少页（最坏情况的上界）。
    pub fn pages_required(&self) -> usize {
        pages_for(self.prompt_len.saturating_add(self.max_new), self.page_size)
    }

    /// 是否已结束。
    pub fn is_done(&self) -> bool {
        self.state == State::Done
    }

    /// 当前上下文长度（提示 + 已生成）。
    pub fn len(&self) -> usize {
        self.prompt_len + self.produced
    }

    /// 上下文是否为空（只有空提示且尚未生成时成立）。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 走到当前长度需要多少页：向上取整，至少一页（否则 [`crate::kv::Table`] 无处可放）。
    pub fn pages_needed(&self) -> usize {
        pages_for(self.len(), self.page_size)
    }

    /// KV 的逻辑形状：`[len, page_size]`。
    pub fn kv_shape(&self) -> Shape {
        Shape::new([self.len(), self.page_size])
    }
}
