//! 训练循环的状态：[`Schedule`]、[`Step`]、[`Checkpoint`]，以及激活缓冲 [`Buffers`]。
//!
//! **拥有**：`Schedule`、`Anneal`、`Step`、`Checkpoint`、`Buffers`、`Kernel`、`Reference`、`Data`、
//! `View`、`ViewMut`，以及承载静态契约的 `Kind`、`Lane`、`Inputs`；`offload` 只留
//! [`WeightOffload`] 暂缓接缝，真实换入换出通路未建。
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
//! 入口只做模块声明与再导出；实现按语义轴落在 `schedule`、`checkpoint` 与 `store` 三个子模块。
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
mod checkpoint;
mod dist;
mod distill;
mod kernel;
mod learner;
mod loss;
mod metrics;
mod objective;
mod offload;
mod optimizer;
mod pool;
mod rl;
mod schedule;
mod session;
mod shape;
pub mod snapshot;
mod store;

pub use checkpoint::{Checkpoint, Step, operators, resume};
pub use dist::{Distributed, Local, Remote};
pub use distill::{Distill, distillation_loss, kl_div, kl_per_row, log_softmax};
pub use learner::{ForwardOut, Learner, Metric};
pub use loss::{argmax_hits, cross_entropy};
pub use metrics::{MetricPoint, MetricsBoard};
pub use objective::{Loss, Outcome};
pub use offload::{DeferredOffload, WeightOffload};
pub use optimizer::{Adam, Group, Optimizer, OwnedSlot, Sgd, Slot};
pub use rl::{Grpo, advantage, clipped_surrogate, entropy, grpo_loss, log_prob, policy_loss};
pub use schedule::{Anneal, Schedule};
pub use session::{ParamSlot, Session, TrainConfig};
pub use store::{Buffers, Data, Inputs, Kernel, Kind, Lane, Reference, View, ViewMut};
