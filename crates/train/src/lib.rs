//! 训练循环的状态：[`Schedule`]、[`Step`]、[`Checkpoint`]，以及激活缓冲 [`Buffers`]。
//!
//! **拥有**：`Schedule`、`Anneal`、`Step`、`Checkpoint`、`Buffers`、`Kernel`、`Reference`、`Data`、
//! `View`、`ViewMut`，以及承载静态契约的 `Kind`、`Lane`、`Inputs`；训练样本的三件套
//! [`Batch`]、[`Dataset`]、[`Chunks`] 也在本 crate（`samples`）。
//! **不拥有**：反传公式（`autograd` 记录顺序、算子表提供公式）、激活的显存
//! 计划（`autograd::Plan`）、分词与打乱（`tokenizer`；打乱只改变游标推进方式）、层定义
//! （`nn`）、参数（`weights`）、拓扑（`shard`）、算子语义（`op`）、检查点字节格式（`snapshot`）。
//! **依赖**：`nn`、`autograd`、`weights`、`shard`、`tensor`、`op`、`diagnostic`、`snapshot`。
//! **被依赖**：`llm`。
//!
//! 检查点的**字节格式**已独立为 `snapshot` 成员（L5）：本 crate 只持有内存里的 [`Checkpoint`]
//! 事实集，编码、版本与偏移校验不在训练循环里再做一遍，两者各自一事。训练样本的 [`Batch`]、
//! [`Dataset`]、[`Chunks`] 仍在本 crate（`samples`）：唯一消费者是 [`Learner`]，
//! "批几何与批总数在哪算"因此也只有一处。
//!
//! 这一层回答两件事：**训练到哪了、能不能恢复**——学习率是完全确定的函数，检查点记录的是
//! "恢复所需的最小事实集"（步数、图规模、批形状、层数、权重条目数、并行度），少任何一项恢复出来
//! 的都不是原来那次训练；**激活放在哪、什么时候还**——[`Buffers`] 是 `autograd::Store` 在训练侧
//! 的实现，把重放要的值放进真实内存，算的部分交给 [`Kernel`]。
//!
//! 入口只做模块声明与再导出；实现按语义轴落在 `schedule`、`checkpoint`、`samples` 与 `store`
//! 四个子模块。
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

mod backprop;
mod dist;
mod distill;
mod kernel;
mod learner;
mod loss;
mod metrics;
mod optimizer;
mod pool;
mod rl;
mod samples;
mod schedule;
mod session;
pub(crate) use kernel::shape;
pub mod geometry;
mod store;

/// 训练侧测试共用的夹具图（`tests/support.rs`）：在 crate 根挂一次，测试模块引用它，
/// 而不是在自己的内联模块里另写一份相对路径——内联模块的 `#[path]` 基准是虚拟目录，跨级上跳会失败。
#[cfg(test)]
#[path = "../tests/support.rs"]
mod support;

pub use dist::{Distributed, Local, Remote};
pub use distill::{Distill, distillation_loss, kl_div, kl_per_row, log_softmax};
pub use geometry::{Geometry, Running};
pub use learner::{ForwardOut, Learner, Metric};
pub use loss::{Loss, Outcome, argmax_hits, bce_with_logits, cross_entropy, hinge, huber, mse};
pub use metrics::{MetricPoint, MetricsBoard};
pub use optimizer::{Adam, Group, Muon, Optimizer, OwnedSlot, Sgd, Slot};
pub use rl::{Grpo, advantage, clipped_surrogate, entropy, grpo_loss, log_prob, policy_loss};
pub use samples::{Batch, Chunks, Dataset};
pub use schedule::{Anneal, Schedule};
pub use session::checkpoint::{Checkpoint, Step, operators, resume};
pub use session::{ParamSlot, Session, TrainConfig};
pub use store::OperatorSetOps;
pub use store::{Buffers, Data, Inputs, Kernel, Kind, Lane, Reference, View, ViewMut};
