//! 显存计划：[`Plan`] 把"反向图 + 每变量字节数 + 预算"变成一组显存决策。
//!
//! 三个问题：[`Plan::saved`] 留谁；[`Plan::release`] 何时释放——"最后一次被需要"，含重算；
//! [`Plan::replayed`] 与 [`Plan::peak`] 代价多大，`peak <= budget` 由构造保证。预算不够时按字节
//! 等分切段：段界留检查点、段内丢弃后就地重放，段长由预算解出来，不是常数。
//!
//! 一般 DAG 的最优是 PSPACE 难的，所以这里不声称最优，只给三个可核对的数：保守峰值、
//! [`Plan::floor`]（任何计划都绕不过的下界）、重放步数。

use diagnostic::Result;

use crate::Var;
use crate::budget::{Budget, Choice, MemoryPolicy};
use crate::config::Config;
use crate::error::invalid;
use crate::facts::{Facts, NONE};
use crate::schedule::Schedule;
use crate::sizes::Footprint;
use crate::tape::Tape;

/// 诊断来源：本模块只有一个入口。
pub(crate) const SITE: &str = "autograd::Plan::of";

/// 一次反向外加前向重放时的显存决策：纯数据（可比较、可缓存），变成动作的是 [`crate::Replay`]。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub(crate) choice: Choice,
    pub(crate) policy: MemoryPolicy,
    pub(crate) budget: Budget,
    pub(crate) saved: Vec<Var>,
    pub(crate) keep: Vec<bool>,
    pub(crate) release: Vec<u32>,
    pub(crate) retained: u64,
    pub(crate) peak: u64,
    pub(crate) floor: u64,
    pub(crate) replayed: u64,
}

impl Plan {
    /// 由反向调度、字节数与预算算出计划。
    ///
    /// # 例子
    ///
    /// 一张 `x → relu → loss` 的图：`x` 是权重（常驻，不进激活预算），`relu` 的输出是唯一的
    /// 激活，反向第 0 步用完就可以还回去。
    ///
    /// ```
    /// use llm_autograd::{Budget, Choice, Plan, Tape};
    ///
    /// let mut tape = Tape::new();
    /// let x = tape.parameter();
    /// let hidden = tape.record("relu", &[x]).expect("记录");
    /// let loss = tape.record("reduce_sum", &[hidden]).expect("记录");
    /// let schedule = tape.backward(loss).expect("在图上");
    ///
    /// let bytes = [0u64, 4]; // 只有 hidden 占 4 字节
    /// let plan = Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(1 << 20)).expect("放得下");
    /// assert_eq!(plan.choice(), Choice::KeepAll, "装得下就不重放");
    /// assert_eq!(plan.saved().len(), 1);
    /// assert_eq!(plan.saved()[0], hidden);
    /// assert_eq!(plan.release(hidden), Some(0), "第 0 步消费完即可释放");
    /// ```
    ///
    /// # Errors
    ///
    /// 预算低于必然下界，或低于任何分段方案能取到的最小峰值时返回 [`diagnostic::Code::Invalid`]：
    /// 宁可报"放不下"，也不给出实际会超预算的计划。
    pub fn of(
        tape: &Tape,
        schedule: &Schedule,
        footprint: impl Footprint,
        budget: Budget,
    ) -> Result<Self> {
        Self::of_policy(tape, schedule, footprint, budget, MemoryPolicy::default())
    }

    /// 按指定策略由反向调度、字节数与预算算出计划。
    ///
    /// [`MemoryPolicy::KeepAll`] 强制全留；[`MemoryPolicy::Recompute`] 强制分段；
    /// [`MemoryPolicy::Auto`] 按预算实测松紧选择，与 [`Plan::of`] 一致。
    ///
    /// # Errors
    ///
    /// 预算低于必然下界，或低于任何分段方案能取到的最小峰值时返回 [`diagnostic::Code::Invalid`]：
    /// 宁可报"放不下"，也不给出实际会超预算的计划。
    pub fn of_policy(
        tape: &Tape,
        schedule: &Schedule,
        footprint: impl Footprint,
        budget: Budget,
        policy: MemoryPolicy,
    ) -> Result<Self> {
        let facts = Facts::new(tape, schedule, &footprint);
        match policy {
            MemoryPolicy::KeepAll => Ok(facts.keep_all(tape, schedule, budget, policy)),
            MemoryPolicy::Auto if facts.total <= budget.bytes() => {
                Ok(facts.keep_all(tape, schedule, budget, policy))
            }
            MemoryPolicy::Auto | MemoryPolicy::Recompute => {
                Self::check_floor(&facts, budget)?;
                facts.plan(tape, schedule, budget, policy)
            }
        }
    }

    /// 按配置里的显存策略算出计划；排序策略属于调度，不在这里使用。
    pub fn of_config(
        tape: &Tape,
        schedule: &Schedule,
        footprint: impl Footprint,
        budget: Budget,
        config: &Config,
    ) -> Result<Self> {
        Self::of_policy(tape, schedule, footprint, budget, config.policy())
    }

    /// 预算低于必然下界时报错；分段路径共用这一道闸。
    fn check_floor(facts: &Facts, budget: Budget) -> Result<()> {
        if budget.bytes() < facts.floor {
            return Err(invalid(
                SITE,
                format!(
                    "预算 {} 字节低于下界 {} 字节：一次算子的输入必须同时在场，没有更省的计划",
                    budget.bytes(),
                    facts.floor
                ),
            ));
        }
        Ok(())
    }

    /// 走的是哪条路。
    pub fn choice(&self) -> Choice {
        self.choice
    }

    /// 生成本计划时采用的策略。
    pub fn policy(&self) -> MemoryPolicy {
        self.policy
    }

    /// 计划针对的预算。
    pub fn budget(&self) -> Budget {
        self.budget
    }

    /// 保留的激活，按前向顺序（编号升序）；这就是要留在显存里的全部中间结果。
    pub fn saved(&self) -> &[Var] {
        &self.saved
    }

    /// 该变量的值是否被保留（没被消费的节点输出与叶子都不是保留对象）。
    pub fn keep(&self, var: Var) -> bool {
        self.keep.get(var.id() as usize).copied().unwrap_or(false)
    }

    /// 该变量在反向走到第几步之后可以释放；不保留时为 `None`。
    pub fn release(&self, var: Var) -> Option<usize> {
        let step = *self.release.get(var.id() as usize)?;
        if step == NONE || !self.keep(var) {
            return None;
        }
        Some(step as usize)
    }

    /// 保留的字节数：反向一开始就占着的量。
    pub fn retained(&self) -> u64 {
        self.retained
    }

    /// 按本计划执行的峰值字节数（保守上界）。
    pub fn peak(&self) -> u64 {
        self.peak
    }

    /// 必然下界：任何计划都绕不过的峰值。
    pub fn floor(&self) -> u64 {
        self.floor
    }

    /// 一次反向里要重放的步数合计：预算换来的计算代价。
    pub fn replayed(&self) -> u64 {
        self.replayed
    }
}

#[cfg(test)]
#[path = "../tests/plan_tests.rs"]
mod tests;
