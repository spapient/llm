//! 训练栈：一次训练从"学什么"到"学到哪了、能不能恢复"的全部事实。
//!
//! 五条语义轴，各占一组子模块，互不越界：
//!
//! 1. **循环状态**——[`Schedule`]/[`Anneal`]（学习率是完全确定的函数）、[`Session`]/[`TrainConfig`]、
//!    [`Step`]/[`Checkpoint`]（"恢复所需的最小事实集"：步数、图规模、批形状、层数、权重条目数、
//!    并行度，少任何一项恢复出来的都不是原来那次训练）、[`Distributed`]、[`Learner`]/[`MetricsBoard`]、
//!    [`Batch`]/[`Dataset`]/[`Chunks`]（`learner`；唯一消费者是 [`Learner`]，故批几何与批总数只有一处算）。
//! 2. **目标函数**——[`Loss`]/[`Outcome`] 与分类/回归损失、蒸馏族（[`Distill`]、`kl_div`、
//!    `log_softmax`）、策略族（[`Grpo`]、`policy_loss`、`clipped_surrogate`、`entropy`）。三族共用
//!    `loss` 的几何校验与数值助手，因此只有一份长度/词表/行数契约。
//! 3. **参数更新**——[`Optimizer`] 契约与 [`Sgd`]/[`Adam`]/[`Muon`]；[`Group`] 按 `ParamRole` 分组，
//!    矩状态按参数下标对齐，与激活槽平行。
//! 4. **反向**——`backprop` 按 `Schedule` 的逆拓扑序把梯度走回参数端点，文件内的公式表是
//!    各算子的**参考公式数值体**（`nn` 提供算子名/输入个数，`prof` 提供声明档语义，本 crate 提供算得出的那条式子）。
//! 5. **激活缓冲与几何**——[`Bufs`] 是 [`autograd::Store`] 在训练侧的实现：等长空闲表，稳态零
//!    分配（装入新值即把旧块还表），`live`/`peak` 与 [`autograd::Plan`] 对账；[`Kernel`] 把它适配到
//!    标量参考实现，`geometry` 是各经典算子的实例几何与前向/反向参考数值（梯度对拍的基准）。
//!
//! **拥有**：上面点名的全部类型，加上承载静态契约的 `Kind`、`Lane`、`Inputs` 与 [`OperatorSetOps`]。
//! **不拥有**：图与记录顺序、激活的显存计划（`autograd`）、算子注册表（`nn`）与声明档语义（`prof`）、
//! 分词与打乱（`tokenizer`；打乱只改变游标推进方式）、层定义（`nn`）、参数（`weights`）、
//! 拓扑与集合通信（`shard`）、检查点字节格式（`storage`；编码、版本与偏移校验不在训练循环里再做一遍）。
//! **依赖**：`nn`、`autograd`、`weights`、`shard`、`tensor`、`prof`、`storage`、`diagnostic`。
//! **被依赖**：`cli`、`benchmarks`、`fuzz`、`serve`、`testkit`。
//!
//! 入口只做模块声明与再导出：实现按上面五条轴分布在 10 个子模块里（`backprop`、`geometry`、`kernel`、
//! `learner`（含 `Batch`/`Dataset`/`Chunks`/`Loader`/`LoaderConfig`）、`loss`（含策略族）、`optimizer`、`pool`（含 `store` 门面）、
//! `rng`、`schedule`（学习率调度曲线）、`session`（会话与分布式编排）；`rng` 是并入的计数器型确定性随机，独立于五条轴）。
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
pub mod geometry;
mod kernel;
mod learner;
mod loss;
mod optimizer;
mod pool;
mod rng;
mod schedule;
mod session;

/// 训练侧测试共用的夹具图（`tests/support.rs`）：在 crate 根挂一次，测试模块引用它，
/// 而不是在自己的内联模块里另写一份相对路径——内联模块的 `#[path]` 基准是虚拟目录，跨级上跳会失败。
#[cfg(test)]
#[path = "../tests/support.rs"]
mod support;

pub use geometry::{Geometry, Running};
pub use learner::{
    Batch, Chunks, Dataset, ForwardOut, Learner, Loader, LoaderConfig, Metric, MetricPoint,
    MetricsBoard, Sampling,
};
pub use loss::{
    Distill, Grpo, Loss, Outcome, advantage, argmax_hits, bce_with_logits, clipped_surrogate,
    cross_entropy, distillation_loss, entropy, grpo_loss, hinge, huber, kl_div, kl_per_row,
    log_prob, log_softmax, mse, policy_loss,
};
pub use optimizer::{Adam, Group, Muon, Optimizer, OwnedSlot, Sgd, Slot};
pub use pool::OperatorSetOps;
pub use pool::{Bufs, Data, Inputs, Kernel, Kind, Lane, Reference, View, ViewMut};
pub use rng::{Bernoulli, Normal, Rng, Uniform, block};
pub use schedule::{Anneal, Schedule};
pub use session::{
    Checkpoint, Distributed, GradientSnapshot, Local, ParamSlot, Remote, Session, Step, TrainConfig,
    TrainSpec, operators, resume,
};
