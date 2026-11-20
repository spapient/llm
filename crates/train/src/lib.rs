//! 训练循环的状态：[`Schedule`]、[`Step`]、[`Checkpoint`]，以及激活缓冲 [`Buffers`]。
//!
//! **拥有**：`Schedule`、`Anneal`、`Step`、`Checkpoint`、`Buffers`、`Kernel`、`Reference`、`Data`、
//! `View`、`ViewMut`，以及承载静态契约的 `Kind`、`Lane`、`Inputs`；训练样本的三件套
//! [`Batch`]、[`Dataset`]、[`Chunks`] 也在本 crate（`samples`）；`offload` 只留
//! [`WeightOffload`] 暂缓接缝，真实换入换出通路未建。
//! **不拥有**：反传公式（`autograd` 记录顺序、算子表提供公式）、激活的显存
//! 计划（`autograd::Plan`）、分词与打乱（`tokenizer`；打乱只改变游标推进方式）、层定义
//! （`nn`）、参数（`weights`）、拓扑（`shard`）、算子语义（`op`）。**依赖**：`nn`、`autograd`、
//! `weights`、`shard`、`tensor`、`op`、`diagnostic`。**被依赖**：`llm`。
//!
//! 检查点的**字节格式**也在本 crate（[`snapshot`]）：它原是一个独立成员（`llm-checkpoint`），
//! 唯一消费者就是这里的 [`Checkpoint`]；合并后"恢复需要知道哪些事实"只有一处定义。
//! 训练样本也一样（`llm-data` → `samples`）：唯一消费者是 [`Learner`]，
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
mod checkpoint;
mod dist;
mod distill;
mod kernel;
mod learner;
mod loss;
mod metrics;
pub(crate) use loss::objective;
mod operators;
mod optimizer;
mod pool;
mod rl;
mod samples;
mod schedule;
mod session;
pub(crate) use kernel::shape;
pub mod geometry;
pub mod snapshot;
mod store;

/// 训练侧测试共用的夹具图（`tests/support.rs`）：在 crate 根挂一次，测试模块引用它，
/// 而不是在自己的内联模块里另写一份相对路径——内联模块的 `#[path]` 基准是虚拟目录，跨级上跳会失败。
#[cfg(test)]
#[path = "../tests/support.rs"]
mod support;

pub use checkpoint::{Checkpoint, Step, operators, resume};
pub use dist::{Distributed, Local, Remote};
pub use distill::{Distill, distillation_loss, kl_div, kl_per_row, log_softmax};
pub use geometry::Geometry;
pub use learner::{ForwardOut, Learner, Metric};
pub use loss::{argmax_hits, bce_with_logits, cross_entropy, hinge, huber, mse};
pub use metrics::{MetricPoint, MetricsBoard};
pub use objective::{Loss, Outcome};
pub use offload::{DeferredOffload, WeightOffload};
pub use operators::OperatorSetOps;
pub use optimizer::{Adam, Group, Muon, Optimizer, OwnedSlot, Sgd, Slot};
pub use rl::{Grpo, advantage, clipped_surrogate, entropy, grpo_loss, log_prob, policy_loss};
pub use samples::{Batch, Chunks, Dataset};
pub use schedule::{Anneal, Schedule};
pub use session::{ParamSlot, Session, TrainConfig};
pub use store::{Buffers, Data, Inputs, Kernel, Kind, Lane, Reference, View, ViewMut};

pub(crate) mod offload {
    //! 参数 offload 接缝（**暂缓**）。
    //!
    //! 本模块只保留接口形状：把权重在驻留内存与后备存储之间换入/换出的通路**尚未建立**。前置
    //! 依赖——页池（块/页编号与驻留表）、权重驻留记账、换入换出搬运与驻留命中判定——都未落地。
    //! 本刀不造假实现、不接虚假换入：接口方法统一返回 [`Code::Unsupported`]，等页池与驻留表
    //! 落地后再填真实通路。
    //!
    //! 暂缓清单：ZeRO、分布式优化器、序列并行、真实权重落盘端到端、设备后端
    //! 均不在本轨；offload 真实通路随后续里程碑接入。

    use diagnostic::{Code, Error, Result};

    /// 本模块诊断站点。
    pub(crate) const SITE: &str = "train::offload";

    /// 参数换入/换出接缝：槽位编号标识一块权重；方法签名是后续真实通路的形状占位。
    ///
    /// # 暂缓前置
    ///
    /// - **页池**：槽位/块编号 → 驻留页或后备页的映射表尚未建立；
    /// - **权重驻留**：哪一块权重此刻在场、哪一块被换出的记账不存在；
    /// - **换入/换出**：后备存储搬运、驻留命中判定与换入后的缓冲镜像都未接。
    ///
    /// 因此当前实现一律返回 [`Code::Unsupported`]，不做任何搬运，也不登记驻留状态；训练闭环
    /// 不调用本 trait 的任何方法。
    pub trait WeightOffload {
        /// 把一块权重换出驻留内存到后备存储（接缝，暂缓）。
        ///
        /// # Errors
        ///
        /// 通路未建立：[`Code::Unsupported`]。
        fn swap_out(&mut self, slot: usize) -> Result<()>;

        /// 把一块权重换回驻留内存（接缝，暂缓）。
        ///
        /// # Errors
        ///
        /// 通路未建立：[`Code::Unsupported`]。
        fn swap_in(&mut self, slot: usize) -> Result<()>;
    }

    /// 未接通路的占位实现：所有方法返回 [`Code::Unsupported`]，不持页池、不搬内存。
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct DeferredOffload;

    impl WeightOffload for DeferredOffload {
        fn swap_out(&mut self, slot: usize) -> Result<()> {
            deferred(slot)
        }

        fn swap_in(&mut self, slot: usize) -> Result<()> {
            deferred(slot)
        }
    }

    /// 统一的暂缓错误：把前置缺口写进消息，避免调用方误以为"换出成功"。
    fn deferred(slot: usize) -> Result<()> {
        Err(Error::new(
        Code::Unsupported,
        format!(
            "参数槽 {slot} 的 offload 换入/换出通路暂缓：页池、权重驻留记账与换入换出搬运尚未建立"
        ),
    )
    .with(SITE))
    }
}
