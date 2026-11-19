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
mod config {
    //! 规划配置：反向排序与显存策略的纯数据约定。
    //!
    //! [`Config`] 是一次规划的**全部可调旋钮**：闭集枚举、可比较、可缓存，不持有缓冲、不认识
    //! 设备。排序策略决定步骤之间的次序，显存策略决定在预算内偏好全留还是分段重放；两者互不
    //! 耦合，分别进入 [`crate::Schedule`] 与 [`crate::Plan`]。

    use crate::plan::MemoryPolicy;
    use crate::schedule::Ordering;

    /// 一次规划的配置：排序策略 + 显存策略。
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub struct Config {
        ordering: Ordering,
        policy: MemoryPolicy,
    }

    impl Config {
        /// 按两个策略构造配置；默认 [`Ordering::Depth`] + [`MemoryPolicy::Auto`]。
        pub const fn new(ordering: Ordering, policy: MemoryPolicy) -> Self {
            Self { ordering, policy }
        }

        /// 反向步骤的排序策略。
        pub const fn ordering(self) -> Ordering {
            self.ordering
        }

        /// 显存规划策略。
        pub const fn policy(self) -> MemoryPolicy {
            self.policy
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn default_config_uses_depth_and_auto() {
            let config = Config::default();
            assert_eq!(config.ordering(), Ordering::Depth);
            assert_eq!(config.policy(), MemoryPolicy::Auto);
        }

        #[test]
        fn config_carries_both_strategies() {
            let config = Config::new(Ordering::Insertion, MemoryPolicy::Recompute);
            assert_eq!(config.ordering(), Ordering::Insertion);
            assert_eq!(config.policy(), MemoryPolicy::Recompute);
        }
    }
}

mod error {
    //! 本层的错误构造：类别只有两种，站点字符串由调用方给出。
    //!
    //! 反传调度里没有"缺能力"：`Unsupported` / `Missing` 都属于别的层（缺设备、缺声明），
    //! 这里只会出现"调用顺序或声明写错"（`Invalid`）与"账本自身不一致"（`Internal`）。

    use diagnostic::{Code, Error};

    /// 调用顺序或声明写错了：引用不存在的变量、乱序结算、给不进图的节点要梯度。
    pub(crate) fn invalid(site: &'static str, message: impl Into<String>) -> Error {
        Error::new(Code::Invalid, message).with(site)
    }

    /// 本层自己的不变量被破坏：扇出计数已经减到 0 却还在收贡献。
    pub(crate) fn internal(site: &'static str, message: impl Into<String>) -> Error {
        Error::new(Code::Internal, message).with(site)
    }
}

mod graph;
mod plan;
mod replay;
mod schedule;
mod summary {
    //! 图的只读统计：变量、节点、角色、可达性与深度计数。
    //!
    //! [`Summary`] 是**前向记录与反向规划的快照**，不持有梯度、不认识数值与设备：它只回答
    //! "这张图有多大、哪些分支进了反向、各角色各有多少"，供诊断、缓存键与容量预估使用。
    //! 同一张磁带、同一份调度导出的统计逐字节相同；统计从磁带与调度各自读出，不改写它们。

    use crate::graph::{Role, Var};
    use crate::schedule::Schedule;
    use crate::tape::Tape;

    /// 一张图与一次反向规划的计数快照。
    ///
    /// 字段全部是计数：不存变量列表、不存边表，避免随图规模线性复制。需要逐个变量时仍走
    /// [`Tape`] 的访问器。
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct Summary {
        vars: u32,
        nodes: usize,
        edges: usize,
        max_depth: u32,
        parameters: usize,
        constants: usize,
        activations: usize,
        steps: usize,
        endpoints: usize,
        reachable: usize,
        dead: usize,
    }

    impl Summary {
        /// 只统计磁带本身：反向相关计数（步骤、终点、可达、死分支）记为 0。
        pub fn of(tape: &Tape) -> Self {
            let mut summary = Self::default();
            summary.collect_tape(tape);
            summary
        }

        /// 连同反向调度一起统计：步骤、终点、可达变量与不可达分支从调度读出。
        pub fn for_schedule(tape: &Tape, schedule: &Schedule) -> Self {
            let mut summary = Self::of(tape);
            summary.steps = schedule.steps().len();
            summary.endpoints = schedule.endpoints().len();
            summary.reachable = schedule.graph().iter().filter(|seen| **seen).count();
            summary.dead = (summary.vars as usize).saturating_sub(summary.reachable);
            summary
        }

        /// 变量总数（叶子加算子输出）。
        pub fn vars(&self) -> u32 {
            self.vars
        }

        /// 节点总数（算子输出的个数）。
        pub fn nodes(&self) -> usize {
            self.nodes
        }

        /// 输入边总数：所有节点的输入个数之和。
        pub fn edges(&self) -> usize {
            self.edges
        }

        /// 最深节点的深度；空图为 0。
        pub fn max_depth(&self) -> u32 {
            self.max_depth
        }

        /// 参数终点叶子个数。
        pub fn parameters(&self) -> usize {
            self.parameters
        }

        /// 不进反向的常量叶子个数。
        pub fn constants(&self) -> usize {
            self.constants
        }

        /// 算子输出（激活）个数。
        pub fn activations(&self) -> usize {
            self.activations
        }

        /// 反向步骤数（可达节点个数）；未连调度时为 0。
        pub fn steps(&self) -> usize {
            self.steps
        }

        /// 梯度终点个数；未连调度时为 0。
        pub fn endpoints(&self) -> usize {
            self.endpoints
        }

        /// 可达变量个数（终点加每个步骤输出）；未连调度时为 0。
        pub fn reachable(&self) -> usize {
            self.reachable
        }

        /// 不在 loss 可达集里的变量个数；未连调度时为 0。
        pub fn dead(&self) -> usize {
            self.dead
        }

        /// 从磁带收集变量、节点、边、深度与角色计数。
        fn collect_tape(&mut self, tape: &Tape) {
            self.vars = tape.vars();
            self.nodes = tape.nodes().len();
            for node in tape.nodes() {
                self.edges += node.inputs().len();
                self.max_depth = self.max_depth.max(node.depth());
            }
            for id in 0..self.vars {
                let Some(role) = tape.role(Var::at(id)) else {
                    continue;
                };
                self.count_role(role);
            }
        }

        /// 把一个变量的出身计入对应角色桶。
        fn count_role(&mut self, role: Role) {
            match role {
                Role::Parameter => self.parameters += 1,
                Role::Constant => self.constants += 1,
                Role::Activation => self.activations += 1,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn counts_separate_roles_and_graph_edges() {
            let mut tape = Tape::new();
            let w = tape.parameter();
            let c = tape.leaf();
            let hidden = tape.record("relu", &[w]).expect("记录");
            let _unused = tape.record("relu", &[c]).expect("记录");
            let loss = tape.record("add", &[hidden]).expect("记录");

            let summary = Summary::of(&tape);
            assert_eq!(summary.vars(), 5);
            assert_eq!(summary.nodes(), 3);
            assert_eq!(summary.edges(), 3);
            assert_eq!(summary.parameters(), 1);
            assert_eq!(summary.constants(), 1);
            assert_eq!(summary.activations(), 3);
            assert_eq!(summary.max_depth(), 2);
            assert_eq!(summary.steps(), 0);

            let schedule = tape.backward(loss).expect("在图上");
            let planned = Summary::for_schedule(&tape, &schedule);
            assert_eq!(planned.steps(), 2, "loss 可达 hidden 与 loss 两步");
            assert_eq!(planned.endpoints(), 1);
            assert!(planned.reachable() >= planned.endpoints());
            assert_eq!(
                planned.dead(),
                summary.vars() as usize - planned.reachable()
            );
        }
    }
}

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
