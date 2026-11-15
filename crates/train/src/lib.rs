//! 训练循环的状态：[`Schedule`]、[`Step`]、[`Checkpoint`]，以及激活缓冲 [`Buffers`]。
//!
//! **拥有**：`Schedule`、`Step`、`Checkpoint`、`Buffers`、`Kernel`、`Reference`、`Data`、
//! `View`、`ViewMut`，以及承载静态契约的 `Kind`、`Lane`、`Inputs`。
//! **不拥有**：反传公式（`autograd` 记录顺序、算子表提供公式）、激活的显存
//! 计划（`autograd::Plan`）、样本生成（`data`）、层定义（`nn`）、参数（`weights`）、拓扑
//! （`shard`）、算子语义（`op`）。**依赖**：`nn`、`autograd`、`data`、`weights`、`shard`、
//! `tensor`、`op`、`diagnostic`。**被依赖**：`llm`。
//!
//! 检查点的**字节格式**也在本 crate（[`snapshot`]）：它原是一个独立成员（`llm-checkpoint`），
//! 唯一消费者就是这里的 [`Checkpoint`]；合并后"恢复需要知道哪些事实"只有一处定义。
//!
//! 这一层回答两件事：**训练到哪了、能不能恢复**——学习率是完全确定的函数，检查点记录的是
//! "恢复所需的最小事实集"（步数、图规模、批形状、层数、权重条目数、并行度），少任何一项恢复出来
//! 的都不是原来那次训练；**激活放在哪、什么时候还**——[`Buffers`] 是 `autograd::Store` 在训练侧
//! 的实现，把重放要的值放进真实内存，算的部分交给 [`Kernel`]。
//!
//! # 例子
//!
//! ```
//! use llm_train::Schedule;
//!
//! let schedule = Schedule::new(10, 100).expect("合法");
//! assert_eq!(schedule.lr(1.0, 0), 0.1, "第 0 步只走到 warmup 的十分之一");
//! assert_eq!(schedule.lr(1.0, 9), 1.0, "warmup 结束达到峰值");
//! assert!(schedule.lr(1.0, 100) < 0.02, "衰减到接近 0");
//! ```

#![deny(unsafe_code)]

pub mod snapshot;
mod store;

use diagnostic::{Code, Error, Result};

pub use store::{Buffers, Data, Inputs, Kernel, Kind, Lane, Reference, View, ViewMut};

/// 一条训练步记录。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step {
    index: u64,
    loss: f32,
}

impl Step {
    /// 由步号与损失构造。
    pub fn new(index: u64, loss: f32) -> Self {
        Self { index, loss }
    }

    /// 步号（从 0 开始）。
    pub fn index(&self) -> u64 {
        self.index
    }

    /// 该步的平均损失。
    pub fn loss(&self) -> f32 {
        self.loss
    }
}

/// 学习率调度：线性 warmup 到峰值，之后线性衰减到 0。
///
/// 选择线性而不是余弦，是因为它只有两个参数、`lr()` 是纯函数——P0 的可解释性优先于
/// 最后一公里的收敛收益。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Schedule {
    warmup: u64,
    total: u64,
}

impl Schedule {
    /// `total` 必须非零，`warmup` 不得超过 `total`。
    pub fn new(warmup: u64, total: u64) -> Result<Self> {
        if total == 0 {
            return Err(Error::new(Code::Invalid, "总步数不能为 0").with("train::Schedule::new"));
        }
        if warmup > total {
            return Err(
                Error::new(Code::Invalid, format!("warmup {warmup} 超过总步数 {total}"))
                    .with("train::Schedule::new"),
            );
        }
        Ok(Self { warmup, total })
    }

    /// warmup 步数。
    pub fn warmup(&self) -> u64 {
        self.warmup
    }

    /// 总步数。
    pub fn total(&self) -> u64 {
        self.total
    }

    /// 第 `step` 步的学习率：`peak` 是 warmup 结束时的值，超过 `total` 后为 0。
    pub fn lr(&self, peak: f32, step: u64) -> f32 {
        if step >= self.total {
            return 0.0;
        }
        if self.warmup > 0 && step < self.warmup {
            // 第 0 步已经有非零学习率，否则前若干步等于白跑。
            return peak * ((step + 1) as f32) / (self.warmup as f32);
        }
        let span = self.total - self.warmup;
        if span == 0 {
            return peak;
        }
        let done = step - self.warmup;
        let remaining = 1.0 - (done as f32) / (span as f32);
        peak * remaining.max(0.0)
    }
}

/// 可恢复检查点：只记"要复现这次训练必须知道的东西"。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Checkpoint {
    step: u64,
    tape_nodes: usize,
    batch_rows: usize,
    layers: usize,
    weight_entries: usize,
    world: usize,
}

impl Checkpoint {
    /// 由当前状态构造；`layers` 为 0 说明没有可训练的层，直接报错。
    pub fn new(
        step: u64,
        tape: &autograd::Tape,
        batch: &data::Batch<'_>,
        layers: usize,
        weights: &weights::Header,
        topology: shard::Topology,
    ) -> Result<Self> {
        if layers == 0 {
            return Err(Error::new(Code::Invalid, "没有可训练的层").with("train::Checkpoint::new"));
        }
        Ok(Self {
            step,
            tape_nodes: tape.len(),
            batch_rows: batch.rows(),
            layers,
            weight_entries: weights.entries().len(),
            world: topology.world(),
        })
    }

    /// 步号。
    pub fn step(&self) -> u64 {
        self.step
    }

    /// 反向图的节点数（恢复后必须一致，否则图和参数对不上）。
    pub fn tape_nodes(&self) -> usize {
        self.tape_nodes
    }

    /// 批的行数。
    pub fn batch_rows(&self) -> usize {
        self.batch_rows
    }

    /// 训练层数。
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// 权重条目数。
    pub fn weight_entries(&self) -> usize {
        self.weight_entries
    }

    /// 恢复时要求的进程数。
    pub fn world(&self) -> usize {
        self.world
    }

    /// 把这些事实与当前权重表组合成可落盘的快照。
    ///
    /// 文件格式由 [`snapshot`] 拥有：本层只回答"恢复需要知道什么"，编码、版本与偏移校验
    /// 不在训练循环里再做一遍。
    pub fn snapshot(&self, weights: &weights::Header) -> Result<snapshot::Snapshot> {
        let facts = snapshot::Facts::new(self.step, self.world, self.layers, self.batch_rows)?;
        Ok(snapshot::Snapshot::new(facts, weights.clone()))
    }
}

/// 一个检查点能否在同一拓扑上恢复。
///
/// 并行度不同意味着参数切分不同，强行恢复只会读出形状正确、内容错位的权重——这是最贵的一类
/// 静默错误，所以在这里拦住。
pub fn resume(checkpoint: &Checkpoint, topology: shard::Topology) -> Result<()> {
    topology.check(checkpoint.world()).map_err(|error| {
        error.with(format!(
            "检查点在第 {} 步、{} 进程下保存",
            checkpoint.step(),
            checkpoint.world()
        ))
    })
}

/// 训练层的算子引用：`nn` 的层定义必须先通过 `op` 的闭集校验。
///
/// 这里只提供"取全部算子名"的便利函数，避免训练循环自己拼字符串。
pub fn operators(layer: &dyn nn::Contract) -> &[String] {
    layer.spca().ops()
}

#[cfg(test)]
mod tests;
