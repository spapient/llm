//! 显存计划：[`Plan`] 把"反向图 + 每变量字节数 + 预算"变成一组显存决策。

use diagnostic::Result;
use tensor::{DType, Shape};

use crate::Var;
use crate::facts::{Facts, NONE};
use crate::invalid;
use crate::schedule::{Ordering, Schedule};
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
            MemoryPolicy::KeepAll => Ok(facts.keep_all(budget, policy)),
            MemoryPolicy::Auto if facts.total <= budget.bytes() => {
                Ok(facts.keep_all(budget, policy))
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

/// 显存预算（字节）：只是一个数，不读环境变量也不读 TOML（配置在 `cli::settings`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Budget {
    bytes: u64,
}

impl Budget {
    /// 由字节数构造；`0` 也是合法值——它会以 [`diagnostic::Code::Invalid`] 报"放不下"，而不是被静默改写。
    pub const fn new(bytes: u64) -> Self {
        Self { bytes }
    }

    /// 预算字节数。
    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

/// 计划走的是哪条路。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Choice {
    /// 预算装得下全部激活：不留检查点，也不重放。
    KeepAll,
    /// 预算不够：按段保留检查点，段内重放。
    Recompute,
}

/// 显存规划策略：预算是硬约束，策略决定在约束内偏好全留还是主动分段重放。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum MemoryPolicy {
    /// 默认：预算装得下就全留，装不下才分段重放；策略让位于实测字节数。
    #[default]
    Auto,
    /// 强制全留：忽略预算松紧，保留全部候选激活，不安排重放（峰值可能超过预算，由调用方兜底）。
    KeepAll,
    /// 强制分段：即使预算宽裕也放检查点、段内重放，用显存换可预测的峰值与释放节奏。
    Recompute,
}

impl MemoryPolicy {
    /// 策略短名：诊断与报告里的稳定标签。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::KeepAll => "keep-all",
            Self::Recompute => "recompute",
        }
    }
}

/// 一个变量的值占多少字节。
pub trait Footprint {
    /// 字节数；`0` 表示不需要为它留显存（标量、或者调用方还没登记形状）。
    fn bytes(&self, var: Var) -> u64;
}

impl Footprint for [u64] {
    fn bytes(&self, var: Var) -> u64 {
        self.get(var.id() as usize).copied().unwrap_or(0)
    }
}

impl Footprint for Vec<u64> {
    fn bytes(&self, var: Var) -> u64 {
        self.as_slice().bytes(var)
    }
}

impl<F: Footprint + ?Sized> Footprint for &F {
    fn bytes(&self, var: Var) -> u64 {
        (**self).bytes(var)
    }
}

/// 变量形状表：记录前向时顺手登记，`Plan` 直接换算成字节；只存形状与元素类型，不复制张量。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Sizes {
    entries: Vec<Option<(Shape, DType)>>,
}

impl Sizes {
    /// 空表，容量按变量数预分配。
    pub fn new(vars: u32) -> Self {
        Self {
            entries: vec![None; vars as usize],
        }
    }

    /// 登记一个变量的形状；编号可以跳跃（后创建的变量先登记也不会错位）。
    pub fn set(&mut self, var: Var, shape: Shape, dtype: DType) {
        let idx = var.id() as usize;
        if idx >= self.entries.len() {
            self.entries.resize(idx + 1, None);
        }
        self.entries[idx] = Some((shape, dtype));
    }

    /// 一个变量登记的形状与元素类型；没登记时为 `None`。
    pub fn get(&self, var: Var) -> Option<(&Shape, DType)> {
        self.entries
            .get(var.id() as usize)
            .and_then(Option::as_ref)
            .map(|(shape, dtype)| (shape, *dtype))
    }

    /// 表的长度：覆盖到的变量数（含还没登记的槽位），与 `Tape::vars()` 同口径。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否还没有任何变量被登记过（表的长度为 0）。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Footprint for Sizes {
    fn bytes(&self, var: Var) -> u64 {
        self.entries
            .get(var.id() as usize)
            .and_then(Option::as_ref)
            .map_or(0, |(shape, dtype)| shape.element_bytes(*dtype) as u64)
    }
}

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
#[path = "../tests/plan_tests.rs"]
mod tests;
