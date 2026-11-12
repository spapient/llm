//! 服务侧状态机：[`Sequence`]、[`State`]、[`Scheduler`]。
//!
//! **拥有**：`Sequence`、`Scheduler`、`State`。**不拥有**：KV 页的簿记（`kv`）、提交（`exec`）、
//! 采样策略（P1：它只决定 `step()` 取哪个 token，不改变状态机）。
//! **依赖**：`kv`、`exec`、`tensor`、`diagnostic`。**被依赖**：`serve`、`llm`。
//!
//! 连续批的关键不是"批"，而是**准入**：每个序列占多少页是显式算出来的，页不够就拒绝，
//! 而不是让 KV 缓冲在跑到一半时溢出。
//!
//! # 例子
//!
//! ```
//! use llm_engine::{Scheduler, Sequence};
//! use exec::Stream;
//! use kv::Cache;
//!
//! let mut scheduler = Scheduler::new(Cache::new(4, 16).expect("页池"));
//! let mut sequence = Sequence::new(0, 8, 4, 16, Stream::new(0)).expect("序列");
//! assert!(scheduler.admit(&sequence).expect("准入"));
//! sequence.step().expect("生成一个 token");
//! assert_eq!(sequence.produced(), 1);
//! assert!(scheduler.finish(sequence.id()).expect("释放页"));
//! ```

#![deny(unsafe_code)]

use diagnostic::{Code, Error, Result};
use tensor::Shape;

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
    produced: usize,
    max_new: usize,
    page_size: usize,
    state: State,
    stream: exec::Stream,
}

impl Sequence {
    /// 由请求参数构造：`page_size` 来自页池，用于算需要多少页。
    pub fn new(
        id: u64,
        prompt_len: usize,
        max_new: usize,
        page_size: usize,
        stream: exec::Stream,
    ) -> Result<Self> {
        if page_size == 0 {
            return Err(Error::new(Code::Invalid, "页宽不能为 0").with("engine::Sequence::new"));
        }
        if max_new == 0 {
            return Err(
                Error::new(Code::Invalid, "max_new 不能为 0（否则请求一进来就结束）")
                    .with("engine::Sequence::new"),
            );
        }
        Ok(Self {
            id,
            prompt_len,
            produced: 0,
            max_new,
            page_size,
            state: State::Waiting,
            stream,
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

    /// 最多生成多少 token。
    pub fn max_new(&self) -> usize {
        self.max_new
    }

    /// 当前状态。
    pub fn state(&self) -> State {
        self.state
    }

    /// 提交所在的流。
    pub fn stream(&self) -> exec::Stream {
        self.stream
    }

    /// 推进一个 token；已经结束时再推进是调用方的 bug。
    pub fn step(&mut self) -> Result<()> {
        if self.state == State::Done {
            return Err(Error::new(Code::Invalid, "序列已结束").with("engine::Sequence::step"));
        }
        self.state = State::Running;
        self.produced += 1;
        if self.produced >= self.max_new {
            self.state = State::Done;
        }
        Ok(())
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

    /// 走到当前长度需要多少页：向上取整，至少一页（否则 `kv::Table` 无处可放）。
    pub fn pages_needed(&self) -> usize {
        self.len().div_ceil(self.page_size).max(1)
    }

    /// KV 的逻辑形状：`[len, page_size]`。
    pub fn kv_shape(&self) -> Shape {
        Shape::new([self.len(), self.page_size])
    }
}

/// 批里一个正在跑的序列：它的页表就是它占的页。
#[derive(Debug, Clone, PartialEq, Eq)]
struct Running {
    id: u64,
    table: kv::Table,
}

/// 连续批调度器：准入、占用页、结束后归还。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scheduler {
    cache: kv::Cache,
    running: Vec<Running>,
}

impl Scheduler {
    /// 由页池构造。
    pub fn new(cache: kv::Cache) -> Self {
        Self {
            cache,
            running: Vec::new(),
        }
    }

    /// 尝试准入：页够就占页并返回 `true`，不够返回 `false`（由调用方决定排队还是拒绝）。
    pub fn admit(&mut self, sequence: &Sequence) -> Result<bool> {
        if self.running.iter().any(|item| item.id == sequence.id()) {
            return Err(
                Error::new(Code::Invalid, format!("序列 {} 已经在批里", sequence.id()))
                    .with("engine::Scheduler::admit"),
            );
        }
        let needed = sequence.pages_needed();
        if self.cache.free_pages() < needed {
            return Ok(false);
        }
        let mut table = kv::Table::new(needed)?;
        for _ in 0..needed {
            let page = self.cache.alloc().ok_or_else(|| {
                Error::new(Code::Internal, "空闲页数与实际分配不符")
                    .with("engine::Scheduler::admit")
            })?;
            table.push(page)?;
        }
        self.running.push(Running {
            id: sequence.id(),
            table,
        });
        Ok(true)
    }

    /// 结束一个序列并归还它的页；返回它是否在批里。
    pub fn finish(&mut self, id: u64) -> Result<bool> {
        let Some(index) = self.running.iter().position(|item| item.id == id) else {
            return Ok(false);
        };
        let running = self.running.remove(index);
        for page in running.table.pages() {
            self.cache.release(*page)?;
        }
        Ok(true)
    }

    /// 批里正在跑的序列数。
    pub fn running(&self) -> usize {
        self.running.len()
    }

    /// 某个序列占的页表。
    pub fn pages_of(&self, id: u64) -> Option<&kv::Table> {
        self.running
            .iter()
            .find(|item| item.id == id)
            .map(|item| &item.table)
    }

    /// 页池（只读）。
    pub fn cache(&self) -> &kv::Cache {
        &self.cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kv::Cache;

    fn sequence(id: u64, prompt: usize, max_new: usize) -> Sequence {
        Sequence::new(id, prompt, max_new, 16, exec::Stream::new(0)).expect("序列")
    }

    #[test]
    fn a_sequence_generates_until_its_limit_then_stops() {
        let mut sequence = sequence(1, 4, 2);
        assert_eq!(sequence.state(), State::Waiting);
        sequence.step().expect("第一步");
        assert_eq!((sequence.produced(), sequence.state()), (1, State::Running));
        sequence.step().expect("第二步");
        assert!(sequence.is_done());
        assert_eq!(sequence.len(), 6);
        assert_eq!(sequence.step().unwrap_err().code(), Code::Invalid);
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

    #[test]
    fn admission_occupies_and_finish_returns_the_pages() {
        let mut scheduler = Scheduler::new(Cache::new(2, 16).expect("页池"));
        let sequence = sequence(1, 20, 4);
        assert!(scheduler.admit(&sequence).expect("准入"));
        assert_eq!(scheduler.running(), 1);
        assert_eq!(scheduler.cache().in_use(), 2);
        assert_eq!(scheduler.pages_of(1).map(kv::Table::len), Some(2));
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
        let mut scheduler = Scheduler::new(Cache::new(2, 16).expect("页池"));
        let big = sequence(7, 64, 4);
        assert!(
            !scheduler.admit(&big).expect("页不够"),
            "页不够时应返回 false"
        );
        assert_eq!(scheduler.cache().in_use(), 0, "拒绝不能留下半个分配");
        assert_eq!(scheduler.running(), 0);
    }

    #[test]
    fn admitting_the_same_sequence_twice_is_a_bug() {
        let mut scheduler = Scheduler::new(Cache::new(2, 16).expect("页池"));
        let sequence = sequence(3, 8, 4);
        assert!(scheduler.admit(&sequence).expect("准入"));
        assert_eq!(
            scheduler.admit(&sequence).unwrap_err().code(),
            Code::Invalid
        );
    }

    #[test]
    fn impossible_sequence_parameters_are_rejected() {
        assert_eq!(
            Sequence::new(0, 1, 1, 0, exec::Stream::new(0))
                .unwrap_err()
                .code(),
            Code::Invalid
        );
        assert_eq!(
            Sequence::new(0, 1, 0, 16, exec::Stream::new(0))
                .unwrap_err()
                .code(),
            Code::Invalid
        );
    }
}
