//! 反向图：记录（[`Tape`]、[`Var`]、[`Node`]）、调度（[`Schedule`]）与账本（[`Account`]）。
//!
//! **拥有**：`Requirement`、`Role`、`Var`、`Node`、`Tape`、`Ordering`、`Schedule`、
//! `Account`、`MemoryPolicy`、`Plan`、`Budget`、`Sizes`、`Footprint`、`Choice`、`Config`、
//! `Summary`、`Replay`、`Store`。**不拥有**：梯度数值与前向激活（运行时缓冲）、具体算子的
//! 反向公式（P1：算子反向表会引用算子名，但记录结构不会变）、前向计算（`nn` 与 `cutile`
//! 生成）。**依赖**：`tensor`、`diagnostic`。**被依赖**：`train`。
//!
//! 这一层只回答四个问题，且答案都是**数据**，不含数值：
//!
//! 1. **谁在图上**：只有 [`Requirement`] 不是 [`Requirement::None`] 的变量参与反向。`record`
//!    从输入推出需求，`record_with(.., None)` 是显式断开（detach），`retain` 把已存在的叶子
//!    升级为梯度终点。[`Role`] 把出身分成参数 / 常量 / 激活，是需求与是否有生产者的派生事实。
//! 2. **按什么顺序执行**：[`Tape::backward`] 只排 loss 可达的节点。排序由 [`Ordering`] 配置：
//!    默认深度降序、同深度编号降序（同时是合法反向拓扑序与并行提示，[`Schedule::levels`]
//!    把同一深度切成互不依赖的批次）；[`Ordering::Insertion`] 按前向磁带逆序。
//! 3. **梯度什么时候齐**：每个变量的扇出（在图上被引用的次数）记在 [`Account`] 里，调用方不必
//!    自己数——`settle` 只允许收满贡献的步骤执行，顺序错了直接报错。
//! 4. **激活留哪些、何时还**：[`Plan`] 在预算内决定保留集、释放点与重放步数；策略由
//!    [`MemoryPolicy`] 配置（自动 / 强制全留 / 强制分段）。[`Config`] 把排序与显存策略收成一份
//!    纯数据约定。[`Summary`] 导出图规模、角色计数与可达性，供诊断、缓存键与容量预估。
//!    `Plan` 是图上的显存决策；名字与 `cutile::Plan`（单条 kernel 选哪个 tile）不冲突。
//!
//! 训练循环里三个对象各复用一份：`Tape::clear` 清图、[`Tape::plan_into`] 重写计划、
//! [`Account::rebuild`] 重置账本，稳定态不再分配。
//!
//! 计划本身是纯数据，把它变成动作的是 [`Replay`]：反向第 `k` 步之前重算哪几步、之后交还哪些
//! 激活，构造时编成两张扁平区间表，执行时零分配。数值不在本 crate，[`Store`] 是唯一的口子——
//! 真实缓冲归 `train`，于是"谁该被重算"与"怎么算"各归各家。
//!
//! # 例子
//!
//! ```
//! use llm_autograd::{Account, Config, Ordering, Schedule, Tape};
//!
//! let mut tape = Tape::new();
//! let x = tape.parameter(); // 要算梯度的叶子
//! let w = tape.leaf(); // 常量输入，不进图
//! let y = tape.record("mul", &[x, w]).expect("输入已存在");
//! let loss = tape.record("reduce_sum", &[y]).expect("输入已存在");
//!
//! let config = Config::default();
//! let mut schedule = Schedule::new(); // 空计划，可反复复用
//! tape.plan_config(&mut schedule, loss, &config).expect("loss 在图上");
//! assert_eq!(schedule.len(), 2, "只有 loss 与 y 要执行反向");
//! assert_eq!(schedule.endpoints(), [x], "梯度终点只有参数 x");
//! assert_eq!(schedule.ordering(), Ordering::Depth);
//!
//! let mut account = Account::new();
//! account.rebuild(&schedule);
//! let mut ready = Vec::new();
//! for step in schedule.steps() {
//!     account.settle_into(step, &mut ready).expect("按调度顺序执行");
//! }
//! assert!(account.finished());
//! ```

#![deny(unsafe_code)]

mod account;
mod config;
mod error;
mod graph;
mod plan;
mod replay;
mod schedule;
mod summary;
mod tape;

pub use account::Account;
pub use config::Config;
pub use graph::{Node, Requirement, Role, Var};
pub use plan::{Budget, Choice, Footprint, MemoryPolicy, Plan, Sizes};
pub use replay::{Replay, Store};
pub use schedule::{Ordering, Schedule};
pub use summary::Summary;
pub use tape::Tape;

#[cfg(test)]
pub(crate) mod testutil {
    use crate::{Requirement, Schedule, Tape, Var};

    /// 从变量栈反向找第一个要求梯度的叶子作 loss，并 backward 出执行序；无 loss 或 backward
    /// 失败返回 `None`（随机图测试据此跳过本轮）。
    pub(crate) fn schedule_of(tape: &Tape, vars: &[Var]) -> Option<Schedule> {
        let loss = vars
            .iter()
            .rev()
            .find(|var| tape.requirement(**var).is_some_and(Requirement::requires))
            .copied()?;
        tape.backward(loss).ok()
    }
}
