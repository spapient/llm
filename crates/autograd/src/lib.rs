//! 反向图：记录（[`Tape`]、[`Var`]、[`Node`]）、调度（[`Schedule`]）与账本（[`Account`]）。
//!
//! **拥有**：`Requirement`、`Var`、`Node`、`Tape`、`Schedule`、`Account`、`Plan`、`Budget`、
//! `Sizes`、`Footprint`、`Choice`、`Replay`、`Store`。**不拥有**：梯度数值与前向激活（运行时
//! 缓冲）、具体算子的反向公式（P1：算子反向表会引用算子名，但记录结构不会变）、前向计算
//! （`nn` 与 `cutile` 生成）。**依赖**：`tensor`、`diagnostic`。**被依赖**：`train`。
//!
//! 这一层只回答三个问题，且答案都是**数据**，不含数值：
//!
//! 1. **谁在图上**：只有 [`Requirement`] 不是 [`Requirement::None`] 的变量参与反向。`record` 从输入
//!    推出需求，`record_with(.., None)` 是显式断开（detach），`retain` 把已存在的叶子升级为梯度终点。
//! 2. **按什么顺序执行**：[`Tape::backward`] 只排 loss 可达的节点，顺序是"深度降序、同深度编号降序"。
//!    它同时是合法的反向拓扑序（输出总比输入深）与并行提示：[`Schedule::levels`] 把同一深度切成
//!    一层一层的批次，层内互不依赖。
//! 3. **梯度什么时候齐**：每个变量的扇出（在图上被引用的次数）记在 [`Account`] 里，调用方不必
//!    自己数——`settle` 只允许收满贡献的步骤执行，顺序错了直接报错。
//!
//! 训练循环里三个对象各复用一份：`Tape::clear` 清图、[`Tape::plan_into`] 重写计划、
//! [`Account::rebuild`] 重置账本，稳定态不再分配。
//!
//! 显存侧另有一个纯算术的产物：[`Plan`] 在预算内决定"哪些激活保留、何时释放、重放多少步"。
//! 它读的是 [`Schedule`]（谁在图上、按什么顺序反向），不读任何张量，也不执行重放。名字与
//! `cutile::Plan`（一个 kernel 选哪个 tile）不冲突：一个是图上的显存决策，一个是单条 kernel 的
//! 调度选择。
//!
//! 计划本身是纯数据，把它变成动作的是 [`Replay`]：反向第 `k` 步之前重算哪几步、之后交还哪些
//! 激活，构造时编成两张扁平区间表，执行时零分配。数值不在本 crate，[`Store`] 是唯一的口子——
//! 真实缓冲归 `train`，于是"谁该被重算"与"怎么算"各归各家。
//!
//! # 例子
//!
//! ```
//! use llm_autograd::{Account, Schedule, Tape};
//!
//! let mut tape = Tape::new();
//! let x = tape.parameter(); // 要算梯度的叶子
//! let w = tape.leaf(); // 常量输入，不进图
//! let y = tape.record("mul", &[x, w]).expect("输入已存在");
//! let loss = tape.record("reduce_sum", &[y]).expect("输入已存在");
//!
//! let mut schedule = Schedule::new(); // 空计划，可反复复用
//! tape.plan_into(&mut schedule, loss).expect("loss 在图上");
//! assert_eq!(schedule.len(), 2, "只有 loss 与 y 要执行反向");
//! assert_eq!(schedule.endpoints(), [x], "梯度终点只有参数 x");
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
mod error;
mod graph;
mod plan;
mod replay;
mod schedule;
mod tape;

pub use account::Account;
pub use graph::{Node, Requirement, Var};
pub use plan::{Budget, Choice, Footprint, Plan, Sizes};
pub use replay::{Replay, Store};
pub use schedule::Schedule;
pub use tape::Tape;
