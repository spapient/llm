//! 训练编排：借用批次、前向评估、反向梯度、参数更新、分布式同步与恢复状态。
//!
//! 主流程是 `Loader → Learner → Session → autograd → Bufs → Optimizer → Checkpoint`。
//! `train` 拥有训练状态与更新算术，不复制 `autograd` 的反向计划、`nn` 的层注册、`shard` 的
//! 拓扑通信或 `storage` 的快照字节格式。配置解析属于 `cli`，本 crate 只接收并重新校验纯数据。
//!
//! # 模块导航
//!
//! | 模块 | 职责 |
//! |---|---|
//! | `session`（含 `checkpoint`） | [`Session`]、[`TrainConfig`]、[`Schedule`]、[`Distributed`] 与 [`Checkpoint`] |
//! | `learner`（含 `dataset`） | [`Learner`]、[`Batch`]、[`Loader`]、[`Dataset`] 与 [`MetricsBoard`] |
//! | `loss` | [`Loss`]、[`ObjectiveConfig`]、[`Distill`]、[`Grpo`] 与共享数值判据 |
//! | `optimizer` | [`Optimizer`]、[`Sgd`]、[`Adam`]、[`Muon`]、[`Hybrid`] 与参数矩状态 |
//! | `backprop` | 逆拓扑参考公式与参数端点梯度传播 |
//! | `geometry` | 实例几何、滚动统计与前向/反向参考数值 |
//! | `kernel` / `pool` | [`Kernel`]、[`Bufs`]、存储 lane 与等长缓冲复用 |
//! | `rng` | [`Rng`]、[`Uniform`]、[`Normal`]、[`Bernoulli`] 的确定性随机 |
//!
//! # 静态配置
//!
//! ```
//! use llm_train::TrainSpec;
//!
//! let spec = TrainSpec::default();
//! spec.validate().expect("默认训练配置合法");
//! let fingerprint = spec.config_fingerprint(&llm_train::Sgd::default());
//! assert_eq!(fingerprint, spec.config_fingerprint(&llm_train::Sgd::default()));
//! ```
//!
//! # 并行注解
//!
//! 注解只生成 [`ParallelConfig`](shard::ParallelConfig)，不读取环境或发起通信。
//!
//! ```
//! use annotation::parallel;
//!
//! #[parallel(tp = 2, pp = 1, dp = 2, reduce = "max", average = false, priority = "high")]
//! struct Sharded;
//!
//! let policy = Sharded::__parallel_config();
//! assert_eq!(policy.topology().expect("合法拓扑").world(), 4);
//! ```
//!
//! 完整的模块职责、恢复不变量、性能约束与后续改造基线见 `docs/train-architecture-2026-11-25.md`。

#![deny(unsafe_code)]

/// crate 内声明式字段投影：文档与访问器同名，避免为每个字段重复函数骨架。
macro_rules! getters {
    ($( $(#[$doc:meta])* $method:ident => $field:ident : $ty:ty ),* $(,)?) => {$(
        $(#[$doc])*
        pub fn $method(&self) -> $ty {
            self.$field
        }
    )*};
}

/// 借用字段单独声明，返回时只做一次引用与切片协变。
macro_rules! ref_getters {
    ($( $(#[$doc:meta])* $method:ident => $field:ident : $ty:ty ),* $(,)?) => {$(
        $(#[$doc])*
        pub fn $method(&self) -> $ty {
            &self.$field
        }
    )*};
}

/// 纯字段 builder 的声明式实现；带校验或条件转换的 builder 不使用此宏。
macro_rules! builders {
    ($( $(#[$doc:meta])* $method:ident => $field:ident : $ty:ty ),* $(,)?) => {$(
        $(#[$doc])*
        pub fn $method(mut self, $field: $ty) -> Self {
            self.$field = $field;
            self
        }
    )*};
}

/// 配置规则表：条件不成立时以统一错误类型和指定诊断站点提前返回。
macro_rules! validation_rules {
    ($site:expr; $($condition:expr => $message:expr),* $(,)?) => {$(
        if !$condition {
            return Err(diagnostic::Error::new(diagnostic::Code::Invalid, $message).with($site));
        }
    )*};
}

mod backprop;
pub mod geometry;
mod kernel;
mod learner;
mod loss;
mod optimizer;
mod pool;
mod rng;
mod session;

pub use geometry::{Geometry, Running};
pub use kernel::{Data, Inputs, Kernel, Kind, Lane, Reference, View, ViewMut};
pub(crate) use kernel::{fits, lane_of, no_storage, orthogonalize, same_str, zeroed};
pub use learner::{
    Batch, Chunks, Dataset, ForwardOut, Learner, Loader, Metric, MetricPoint, MetricsBoard,
};
pub use loss::{
    Distill, Grpo, Loss, ObjectiveConfig, Outcome, advantage, argmax_hits, bce_with_logits,
    clipped_surrogate, cross_entropy, distillation_loss, entropy, grpo_loss, hinge, huber, kl_div,
    kl_per_row, log_prob, log_softmax, mse, policy_loss,
};
pub use optimizer::{Adam, Group, Hybrid, Muon, Optimizer, OwnedSlot, Sgd, Slot};
pub use pool::Bufs;
pub use pool::OperatorSetOps;
pub use rng::{Bernoulli, LoaderConfig, Normal, Rng, Sampling, Uniform, block};
pub use session::{
    Anneal, Checkpoint, Distributed, GradientSnapshot, Local, ParamSlot, Remote, Schedule, Session,
    Step, TrainConfig, TrainOverrides, TrainSpec, operators, resume,
};

/// 训练侧测试共用夹具：同构的图与形状装配只在 crate 根保留一份。
#[cfg(test)]
mod support {
    #![allow(dead_code)]
    use autograd::{Sizes, Tape, Var};
    use tensor::{DType, Shape};

    /// 批量登记 `(变量, 元素数)` 的 f32 形状。
    pub(crate) fn sizes_of(tape: &Tape, vars: &[(Var, usize)]) -> Sizes {
        let mut sizes = Sizes::new(tape.vars());
        for &(var, len) in vars {
            sizes.set(var, Shape::new([len]), DType::F32);
        }
        sizes
    }

    /// `loss = add_f32(p, x)`：参数、常量输入与输出使用同一宽度。
    pub(crate) fn add_graph(width: usize) -> (Tape, Sizes, Var, Var, Var) {
        let mut tape = Tape::new();
        let p = tape.parameter();
        let x = tape.leaf();
        let loss = tape.record("add_f32", &[p, x]).expect("记录 loss");
        let sizes = sizes_of(&tape, &[(p, width), (x, width), (loss, width)]);
        (tape, sizes, p, x, loss)
    }
}
