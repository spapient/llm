//! 显存计划：[`Plan`] 把"反向图 + 每变量字节数 + 预算"变成一组显存决策。
//!
//! 三个问题：[`Plan::saved`] 留谁；[`Plan::release`] 何时释放——"最后一次被需要"，含重算；
//! [`Plan::replayed`] 与 [`Plan::peak`] 代价多大，`peak <= budget` 由构造保证。预算不够时按字节
//! 等分切段：段界留检查点、段内丢弃后就地重放，段长由预算解出来，不是常数。
//!
//! 一般 DAG 的最优是 PSPACE 难的，所以这里不声称最优，只给三个可核对的数：保守峰值、
//! [`Plan::floor`]（任何计划都绕不过的下界）、重放步数。

use crate::config::Config;
use crate::error::invalid;
use crate::{Node, Schedule, Tape, Var};
use diagnostic::Result;
use tensor::{DType, Shape};

/// 诊断来源：本模块只有一个入口。
const SITE: &str = "autograd::Plan::of";

/// "没有"：既表示变量没有生产者，也表示它没有被消费过。
const NONE: u32 = u32::MAX;

/// 一个变量的值占多少字节。
///
/// 形状不在反向图里（[`crate::Var`] 只是编号），所以字节数由调用方给：`nn` 用层的形状表，
/// `train` 用真实缓冲的容量。泛型而不是 `dyn`——`Plan::of` 每个变量只查一次，没有动态分派。
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

/// 变量形状表：记录前向时顺手登记，[`Plan`] 直接换算成字节；只存形状与元素类型，不复制张量。
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
        let index = var.id() as usize;
        if index >= self.entries.len() {
            self.entries.resize(index + 1, None);
        }
        self.entries[index] = Some((shape, dtype));
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

/// 一次反向外加前向重放时的显存决策：纯数据（可比较、可缓存），变成动作的是 [`crate::Replay`]。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    choice: Choice,
    policy: MemoryPolicy,
    budget: Budget,
    saved: Vec<Var>,
    keep: Vec<bool>,
    release: Vec<u32>,
    retained: u64,
    peak: u64,
    floor: u64,
    replayed: u64,
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

/// 从调度读出的事实：只读，构造一次，之后每次评估计划都复用它。
struct Facts {
    /// 每个变量的字节数；不是候选（叶子、没有被消费的节点输出）为 0。
    bytes: Vec<u64>,
    /// 每个变量最后一次被消费的反向步号；没被消费为 [`NONE`]。
    last_use: Vec<u32>,
    /// 每个变量的生产者（`schedule.steps()` 的下标）。
    producer: Vec<u32>,
    /// 全部候选激活，按前向顺序（编号升序）。
    candidates: Vec<u32>,
    /// 候选里字节数大于 0 的那部分：分段只按它们切，零字节的没有可切的东西。
    order: Vec<u32>,
    /// `order` 的字节前缀和，含自身。
    prefix: Vec<u64>,
    /// 全留时的峰值：所有候选的字节和。
    total: u64,
    /// 必然下界：某一步同时要放下的输入字节和的最大值。
    floor: u64,
}

impl Facts {
    /// 一遍扫描定"生产者与最后一次消费"，再按候选算字节、前缀和与下界。
    fn new(tape: &Tape, schedule: &Schedule, footprint: &impl Footprint) -> Self {
        let steps = schedule.steps();
        let vars = tape.vars();
        // 按**整张图**的变量数开表：释放点的传播会沿被丢弃的链往上走，链上可能有调度外的祖先。
        let (last_use, producer) = uses(steps, vars as usize);
        // 候选 = 被调度消费过的**算子输出**：叶子常驻、loss 的输出没人消费，都不进预算。
        let output = |id: u32| last_use[id as usize] != NONE && tape.node(Var::at(id)).is_some();
        let bytes: Vec<u64> = (0..vars)
            .map(|id| match output(id) {
                true => footprint.bytes(Var::at(id)),
                false => 0,
            })
            .collect();
        let candidates: Vec<u32> = (0..vars).filter(|id| output(*id)).collect();
        let order: Vec<u32> = candidates
            .iter()
            .copied()
            .filter(|id| bytes[*id as usize] > 0)
            .collect();
        let mut prefix = Vec::with_capacity(order.len());
        let mut running = 0u64;
        for id in &order {
            running = running.saturating_add(bytes[*id as usize]);
            prefix.push(running);
        }
        let total = bytes.iter().copied().fold(0u64, u64::saturating_add);
        let floor = floor_of(steps, &bytes);
        Self {
            bytes,
            last_use,
            producer,
            candidates,
            order,
            prefix,
            total,
            floor,
        }
    }

    /// 全留：预算装得下时这是唯一不引入重放的计划。
    fn keep_all(
        &self,
        tape: &Tape,
        schedule: &Schedule,
        budget: Budget,
        policy: MemoryPolicy,
    ) -> Plan {
        let mut keep = vec![false; self.bytes.len()];
        self.candidates
            .iter()
            .for_each(|id| keep[*id as usize] = true);
        let ledger = self.account(tape, schedule, &keep);
        self.assemble(keep, ledger, budget, policy)
    }

    /// 分段：段数由预算解出来，段界上的激活保留，段内的丢弃后重放。
    fn plan(
        &self,
        tape: &Tape,
        schedule: &Schedule,
        budget: Budget,
        policy: MemoryPolicy,
    ) -> Result<Plan> {
        if self.order.is_empty() {
            // total > budget 且 order 为空只可能是 total 与 budget 都为 0，已被全留路径拦下。
            return Err(invalid(SITE, "没有需要保留的激活，但预算为 0"));
        }
        // 每个候选段数都记一份账：能装下的取段数最多的一份，保留多一些、重放少一些。
        let mut scored: Vec<(usize, Vec<bool>, Ledger)> = self
            .trials(budget)
            .into_iter()
            .map(|segments| {
                let keep = self.segmented(segments);
                let ledger = self.account(tape, schedule, &keep);
                (segments, keep, ledger)
            })
            .collect();
        let fitted = scored
            .iter()
            .enumerate()
            .filter(|(_, (_, _, ledger))| ledger.peak <= budget.bytes())
            .max_by_key(|(_, (segments, _, _))| *segments)
            .map(|(index, _)| index);
        match fitted {
            Some(index) => {
                let (_, keep, ledger) = scored.swap_remove(index);
                Ok(self.assemble(keep, ledger, budget, policy))
            }
            None => {
                let minimum = scored.iter().map(|(_, _, ledger)| ledger.peak).min();
                Err(invalid(
                    SITE,
                    format!(
                        "预算 {} 字节放不下：{} 个激活共 {} 字节，最小峰值约 {} 字节（下界 {} 字节）",
                        budget.bytes(),
                        self.order.len(),
                        self.total,
                        minimum.unwrap_or(self.total),
                        self.floor
                    ),
                ))
            }
        }
    }

    /// 候选段数：在 `√n` 与"保留量不超过预算一半"两个中心各扫 ±2 档，然后去重。
    fn trials(&self, budget: Budget) -> Vec<usize> {
        // 等字节切分下峰值 ≈ k·(total/n) + total/k，在 k = √n 处取到最小。
        let count = self.order.len();
        let root = (count as f64).sqrt() as usize;
        let quota = self.total.div_ceil((budget.bytes() / 2).max(1)) as usize;
        let mut trials: Vec<usize> = Vec::new();
        for centre in [root, quota] {
            for offset in [-2i64, -1, 0, 1, 2] {
                let trial = centre as i64 + offset;
                if trial >= 1 && trial as usize <= count {
                    trials.push(trial as usize);
                }
            }
        }
        trials.sort_unstable();
        trials.dedup();
        trials
    }

    /// 分段后的保留集：段界上是检查点；零字节的与 detach 出去的值无处重放，无条件留。
    fn segmented(&self, segments: usize) -> Vec<bool> {
        let mut keep = self.boundaries(segments);
        self.candidates
            .iter()
            .filter(|id| self.bytes[**id as usize] == 0 || self.producer[**id as usize] == NONE)
            .for_each(|id| keep[*id as usize] = true);
        keep
    }

    /// 按字节数均匀地放 `segments` 个检查点：段界落在前缀和最接近等分点的那个激活上。
    fn boundaries(&self, segments: usize) -> Vec<bool> {
        let mut keep = vec![false; self.bytes.len()];
        let last = self.order.len() - 1;
        let total = self.total.max(1);
        for index in 1..=segments {
            let target = (u128::from(total) * index as u128 / segments as u128) as u64;
            let position = self.prefix.partition_point(|sum| *sum < target).min(last);
            keep[self.order[position] as usize] = true;
        }
        // 第一个激活没有更早的检查点可依，必须留；否则第一段无从重放。
        keep[self.order[0] as usize] = true;
        keep
    }

    /// 用同一份账核对一个保留集：保留量、峰值（保守）、重放步数、释放点。
    fn account(&self, tape: &Tape, schedule: &Schedule, keep: &[bool]) -> Ledger {
        let (wide, work) = self.widths(schedule, keep);
        let (transient, replayed) = self.transient(schedule, keep, &wide, &work);
        let retained = self
            .order
            .iter()
            .filter(|id| keep[**id as usize])
            .fold(0u64, |sum, id| sum.saturating_add(self.bytes[*id as usize]));
        Ledger {
            retained,
            peak: retained.saturating_add(transient),
            replayed,
            release: self.release(tape, keep),
        }
    }

    /// 重算一个激活要先重算它被丢弃的输入：`wide` 是瞬时占用，`work` 是步数。
    fn widths(&self, schedule: &Schedule, keep: &[bool]) -> (Vec<u64>, Vec<u64>) {
        // 输入编号必然小于输出，所以前向一遍即可。
        let mut wide = vec![0u64; self.bytes.len()];
        let mut work = vec![0u64; self.bytes.len()];
        for id in &self.order {
            let slot = *id as usize;
            if keep[slot] {
                continue;
            }
            let step = &schedule.steps()[self.producer[slot] as usize];
            let mut width = 0u64;
            let mut chain = 1u64;
            for input in step.inputs() {
                let source = input.id() as usize;
                if keep[source] {
                    continue;
                }
                width = width.max(wide[source]);
                chain = chain.saturating_add(work[source]);
            }
            wide[slot] = self.bytes[slot].saturating_add(width);
            work[slot] = chain;
        }
        (wide, work)
    }

    /// 每一步的瞬时工作集与重放步数：它要的、被丢弃的输入要同时在场。
    fn transient(
        &self,
        schedule: &Schedule,
        keep: &[bool],
        wide: &[u64],
        work: &[u64],
    ) -> (u64, u64) {
        let mut stamp = vec![0u32; self.bytes.len()];
        let mut transient = 0u64;
        let mut replayed = 0u64;
        for (index, step) in schedule.steps().iter().enumerate() {
            let mut sum = 0u64;
            each_input(step, &mut stamp, index as u32 + 1, |slot| {
                if keep[slot] {
                    return;
                }
                sum = sum.saturating_add(wide[slot]);
                replayed = replayed.saturating_add(work[slot]);
            });
            transient = transient.max(sum);
        }
        (transient, replayed)
    }

    /// 释放点从"直接的最后一次消费"出发，再沿被丢弃的链往上传播。
    fn release(&self, tape: &Tape, keep: &[bool]) -> Vec<u32> {
        let mut release = self.last_use.clone();
        // 节点编号沿边严格递增，所以按编号降序走一遍即可：重算需求只会来自更高的编号。
        for node in tape.nodes().iter().rev() {
            carry(&mut release, node, keep, &self.bytes);
        }
        release
    }

    /// 把保留集装配成计划：释放点来自账本对"最后一次需要"的传播。
    fn assemble(
        &self,
        keep: Vec<bool>,
        ledger: Ledger,
        budget: Budget,
        policy: MemoryPolicy,
    ) -> Plan {
        let saved = self
            .candidates
            .iter()
            .filter(|id| keep[**id as usize])
            .map(|id| Var::at(*id))
            .collect();
        // 只看候选：叶子与 loss 的输出从来不是保留对象，不该影响"是否走了重放"。
        let kept = self.order.iter().filter(|id| keep[**id as usize]).count();
        let choice = match kept == self.order.len() {
            true => Choice::KeepAll,
            false => Choice::Recompute,
        };
        Plan {
            choice,
            policy,
            budget,
            saved,
            keep,
            release: ledger.release,
            retained: ledger.retained,
            peak: ledger.peak,
            floor: self.floor,
            replayed: ledger.replayed,
        }
    }
}

/// 每个变量的生产者与最后一次被消费的反向步号；没被消费为 [`NONE`]。
fn uses(steps: &[Node], len: usize) -> (Vec<u32>, Vec<u32>) {
    let mut last_use = vec![NONE; len];
    let mut producer = vec![NONE; len];
    for (index, step) in steps.iter().enumerate() {
        producer[step.output().id() as usize] = index as u32;
        for input in step.inputs() {
            // 反向顺序遍历，最后一次覆盖就是"最后一次被消费"。
            last_use[input.id() as usize] = index as u32;
        }
    }
    (last_use, producer)
}

/// 必然下界：某一步同时要放下的输入字节和的最大值。
fn floor_of(steps: &[Node], bytes: &[u64]) -> u64 {
    let mut stamp = vec![0u32; bytes.len()];
    let mut floor = 0u64;
    for (index, step) in steps.iter().enumerate() {
        let mut sum = 0u64;
        each_input(step, &mut stamp, index as u32 + 1, |slot| {
            sum = sum.saturating_add(bytes[slot]);
        });
        floor = floor.max(sum);
    }
    floor
}

/// 一步的输入去重后逐个交给 `visit`：同一个输入只算一次（只有一块缓冲）。
fn each_input(step: &Node, stamp: &mut [u32], mark: u32, mut visit: impl FnMut(usize)) {
    for input in step.inputs() {
        let slot = input.id() as usize;
        if stamp[slot] == mark {
            continue;
        }
        stamp[slot] = mark;
        visit(slot);
    }
}

/// 把一步的重算需求推给它的输入：输出被丢弃且仍被需要，链上的祖先就得活到那一步。
fn carry(release: &mut [u32], node: &Node, keep: &[bool], bytes: &[u64]) {
    let output = node.output().id() as usize;
    if keep[output] || release[output] == NONE {
        return;
    }
    let needed = release[output];
    for input in node.inputs() {
        let slot = input.id() as usize;
        if later(release[slot], needed, bytes[slot]) {
            release[slot] = needed;
        }
    }
}

/// 释放点是否要往后推到 `needed`：零字节的不占预算，没记过（`NONE`）与记得更早的都算。
fn later(seen: u32, needed: u32, bytes: u64) -> bool {
    bytes > 0 && (seen == NONE || seen < needed)
}

/// 一个保留集的账：所有数字必须来自同一次统计，否则峰值、重放步数与释放点会对不上。
#[derive(Debug, Clone)]
struct Ledger {
    retained: u64,
    peak: u64,
    replayed: u64,
    /// 每个变量最后一次被需要（直接消费或作为重算起点）的反向步号。
    release: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use crate::{
        Budget, Choice, Config, Footprint, MemoryPolicy, Ordering, Plan, Requirement, Schedule,
        Sizes, Tape, Var,
    };
    use diagnostic::Code;
    use tensor::{DType, Shape};
    use testkit::Rng;

    /// 一条链：`length` 个 `relu` 加一个 `reduce_sum`，反向顺序与深度都已知。
    fn chain(length: usize) -> (Tape, Schedule) {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let mut current = x;
        for _ in 0..length {
            current = tape.record("relu", &[current]).expect("记录");
        }
        let loss = tape.record("reduce_sum", &[current]).expect("记录");
        let schedule = tape.backward(loss).expect("在图上");
        (tape, schedule)
    }

    fn uniform(tape: &Tape, bytes: u64) -> Vec<u64> {
        vec![bytes; tape.vars() as usize]
    }

    #[test]
    fn keep_all_when_the_budget_fits_everything() {
        let (tape, schedule) = chain(4);
        let plan =
            Plan::of(&tape, &schedule, uniform(&tape, 8), Budget::new(1 << 20)).expect("放得下");
        assert_eq!(plan.choice(), Choice::KeepAll);
        assert_eq!(plan.replayed(), 0);
        // 四个 relu 的输出被消费，loss 的输出没人消费：峰值就是这四块。
        assert_eq!(plan.saved().len(), 4);
        assert_eq!(plan.retained(), 32);
        assert_eq!(plan.peak(), 32);
        assert!(plan.peak() >= plan.floor(), "峰值不能低于下界");
    }

    #[test]
    fn forced_keep_all_keeps_everything_even_under_pressure() {
        let (tape, schedule) = chain(16);
        let plan = Plan::of_policy(
            &tape,
            &schedule,
            uniform(&tape, 16),
            Budget::new(200),
            MemoryPolicy::KeepAll,
        )
        .expect("全留");
        assert_eq!(plan.policy(), MemoryPolicy::KeepAll);
        assert_eq!(plan.choice(), Choice::KeepAll);
        assert_eq!(plan.replayed(), 0);
    }

    #[test]
    fn forced_recompute_segments_even_with_room_to_spare() {
        let (tape, schedule) = chain(8);
        let plan = Plan::of_policy(
            &tape,
            &schedule,
            uniform(&tape, 16),
            Budget::new(1 << 20),
            MemoryPolicy::Recompute,
        )
        .expect("分段");
        assert_eq!(plan.policy(), MemoryPolicy::Recompute);
        assert_eq!(plan.choice(), Choice::Recompute);
        assert!(plan.replayed() > 0, "强制重放必须留下段内重算");
    }

    #[test]
    fn of_config_forwards_the_configured_memory_policy() {
        let (tape, schedule) = chain(8);
        let config = Config::new(Ordering::Depth, MemoryPolicy::Recompute);
        let plan = Plan::of_config(
            &tape,
            &schedule,
            uniform(&tape, 16),
            Budget::new(1 << 20),
            &config,
        )
        .expect("分段");
        assert_eq!(plan.policy(), MemoryPolicy::Recompute);
        assert_eq!(plan.choice(), Choice::Recompute);

        let auto = Config::default();
        let plan = Plan::of_config(
            &tape,
            &schedule,
            uniform(&tape, 16),
            Budget::new(1 << 20),
            &auto,
        )
        .expect("放得下");
        assert_eq!(plan.policy(), MemoryPolicy::Auto);
        assert_eq!(plan.choice(), Choice::KeepAll);
    }

    #[test]
    fn a_budget_below_the_floor_is_unsupported() {
        let mut tape = Tape::new();
        let p = tape.parameter();
        let q = tape.parameter();
        let r = tape.parameter();
        let a = tape.record("relu", &[p]).expect("记录");
        let b = tape.record("relu", &[q]).expect("记录");
        let c = tape.record("relu", &[r]).expect("记录");
        let joined = tape.record("add", &[a, b, c]).expect("记录");
        let loss = tape.record("reduce_sum", &[joined]).expect("记录");
        let schedule = tape.backward(loss).expect("在图上");

        let mut bytes = vec![0u64; tape.vars() as usize];
        for var in [a, b, c] {
            bytes[var.id() as usize] = 100;
        }
        let error =
            Plan::of(&tape, &schedule, bytes, Budget::new(200)).expect_err("三个输入要同时在场");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("300"), "{error}");
    }

    #[test]
    fn a_tight_budget_buys_replay_instead_of_memory() {
        let (tape, schedule) = chain(16);
        let plan =
            Plan::of(&tape, &schedule, uniform(&tape, 16), Budget::new(200)).expect("可分段");
        assert_eq!(plan.choice(), Choice::Recompute);
        assert!(plan.peak() <= 200, "峰值 {} 超预算", plan.peak());
        assert!(plan.replayed() > 0, "省下的显存必须用重放换");
        assert!(
            plan.saved().windows(2).all(|pair| pair[0] < pair[1]),
            "检查点按前向顺序且不重复"
        );
        assert!(plan.saved().len() >= 2, "至少要留第一个激活和段界");
        assert!(plan.peak() >= plan.floor());
    }

    #[test]
    fn release_points_follow_the_last_consumer_not_the_scope() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let shared = tape.record("relu", &[x]).expect("记录");
        let left = tape.record("mul", &[shared]).expect("记录");
        let right = tape.record("mul", &[shared]).expect("记录");
        let joined = tape.record("add", &[left, right]).expect("记录");
        let loss = tape.record("reduce_sum", &[joined]).expect("记录");
        let schedule = tape.backward(loss).expect("在图上");
        let plan =
            Plan::of(&tape, &schedule, uniform(&tape, 4), Budget::new(1 << 20)).expect("放得下");

        // 反向顺序是 loss(0)、joined(1)、right(2)、left(3)、shared(4)：共享的激活活到第 3 步。
        assert_eq!(
            schedule
                .steps()
                .iter()
                .map(|step| step.output())
                .collect::<Vec<_>>(),
            [loss, joined, right, left, shared]
        );
        assert_eq!(plan.release(shared), Some(3), "活到较晚的那一支");
        assert_eq!(plan.release(loss), None, "没人消费就不必保留");
        assert!(!plan.keep(loss));
        assert!(!plan.keep(x), "叶子是常驻的权重，不进激活预算");
    }

    #[test]
    fn a_zero_budget_cannot_hold_anything() {
        let (tape, schedule) = chain(3);
        let error =
            Plan::of(&tape, &schedule, uniform(&tape, 8), Budget::new(0)).expect_err("放不下");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("预算 0"), "{error}");
    }

    #[test]
    fn the_plan_is_deterministic() {
        let (tape, schedule) = chain(12);
        let first =
            Plan::of(&tape, &schedule, uniform(&tape, 16), Budget::new(160)).expect("可分段");
        let second =
            Plan::of(&tape, &schedule, uniform(&tape, 16), Budget::new(160)).expect("可分段");
        assert_eq!(first, second);
    }

    #[test]
    fn the_size_table_turns_shapes_into_bytes() {
        let mut sizes = Sizes::new(1);
        assert!(!sizes.is_empty());
        sizes.set(Var::at(0), Shape::new([2, 3]), DType::F32);
        assert_eq!(sizes.bytes(Var::at(0)), 24);
        sizes.set(Var::at(0), Shape::new([4]), DType::U4);
        assert_eq!(sizes.bytes(Var::at(0)), 2, "4 位元素按位打包");
        sizes.set(Var::at(9), Shape::new([5]), DType::F16);
        assert_eq!(sizes.len(), 10, "跳跃登记会补齐中间的空位");
        assert_eq!(sizes.bytes(Var::at(8)), 0, "没登记的变量按 0 计");
    }

    #[test]
    fn a_size_table_drives_the_plan() {
        let (tape, schedule) = chain(4);
        let mut sizes = Sizes::new(tape.vars());
        for step in schedule.steps() {
            sizes.set(step.output(), Shape::new([2, 2]), DType::F32);
        }
        let plan = Plan::of(&tape, &schedule, &sizes, Budget::new(1 << 20)).expect("放得下");
        assert_eq!(plan.retained(), 4 * 16, "四个 2×2 的 f32 激活");
    }

    #[test]
    fn a_dead_branch_stays_out_of_the_plan() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let dead = tape
            .record_with("stop_gradient", &[x], Requirement::None)
            .expect("记录");
        let _hidden = tape.record("relu", &[dead]).expect("记录");
        let live = tape.record("relu", &[x]).expect("记录");
        let loss = tape.record("reduce_sum", &[live]).expect("记录");
        let schedule = tape.backward(loss).expect("在图上");
        let plan =
            Plan::of(&tape, &schedule, uniform(&tape, 8), Budget::new(1 << 20)).expect("放得下");
        assert!(
            !plan.keep(dead),
            "不进图的分支没有反向步骤，也就不进显存计划"
        );
        assert!(plan.keep(live));
        assert_eq!(plan.saved().len(), 1);
    }

    #[test]
    fn a_detached_value_consumed_by_a_live_step_is_kept_and_never_dropped() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let frozen = tape
            .record_with("stop_gradient", &[x], Requirement::None)
            .expect("记录");
        let mut current = tape.record("relu", &[frozen, x]).expect("记录");
        for _ in 0..8 {
            current = tape.record("relu", &[current]).expect("记录");
        }
        let loss = tape.record("reduce_sum", &[current]).expect("记录");
        let schedule = tape.backward(loss).expect("在图上");

        // 10 个 8 字节的激活共 80 字节；预算 64 时必然要丢一些，但 frozen 无处可重放。
        let plan = Plan::of(&tape, &schedule, uniform(&tape, 8), Budget::new(64)).expect("可分段");
        assert_eq!(plan.choice(), Choice::Recompute);
        assert!(
            plan.keep(frozen),
            "detach 出去的值仍被活着的步骤消费，只能留"
        );
        assert!(plan.saved().contains(&frozen));
        assert!(plan.peak() <= 64, "峰值 {} 超预算", plan.peak());
    }

    #[test]
    fn a_deep_chain_does_not_recurse_and_fits_a_tight_budget() {
        let (tape, schedule) = chain(20_000);
        let bytes = uniform(&tape, 1);
        let large =
            Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(1 << 20)).expect("全留");
        assert_eq!(large.choice(), Choice::KeepAll);

        let tight =
            Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(2_000)).expect("可分段");
        assert!(tight.peak() <= 2_000, "峰值 {} 超预算", tight.peak());
        assert!(tight.replayed() > 0);
    }

    #[test]
    fn random_graphs_never_exceed_their_budget() {
        let mut rng = Rng::new(0x9E37_79B9);
        for round in 0..200 {
            let mut tape = Tape::new();
            let mut vars: Vec<Var> = (0..1 + rng.below(4)).map(|_| tape.parameter()).collect();
            let rounds = 1 + rng.below(24);
            crate::testutil::random_edges(&mut tape, &mut vars, &mut rng, rounds, false);
            let Some(schedule) = crate::testutil::schedule_of(&tape, &vars) else {
                continue;
            };
            let bytes: Vec<u64> = (0..tape.vars()).map(|_| u64::from(rng.below(64))).collect();
            let budget = Budget::new(u64::from(rng.below(512)));
            match Plan::of(&tape, &schedule, bytes.as_slice(), budget) {
                Ok(plan) => {
                    assert!(
                        plan.peak() <= budget.bytes(),
                        "第 {round} 轮：峰值 {} 超预算 {}",
                        plan.peak(),
                        budget.bytes()
                    );
                    assert!(plan.peak() >= plan.floor(), "第 {round} 轮：峰值低于下界");
                    for var in plan.saved() {
                        assert!(plan.keep(*var), "第 {round} 轮：保留集与位图不一致");
                        assert!(plan.release(*var).is_some(), "第 {round} 轮：保留却不释放");
                        assert!(
                            tape.node(*var).is_some(),
                            "第 {round} 轮：叶子（参数/输入）不该进激活预算"
                        );
                    }
                }
                Err(error) => {
                    assert_eq!(error.code(), Code::Invalid, "第 {round} 轮：{error}");
                }
            }
        }
    }

    /// 固定图形与预算下的计划摘要：`saved`/`release`/`replayed`/`peak` 逐值冻结。
    fn summarize(
        case: &str,
        tape: &Tape,
        schedule: &Schedule,
        bytes: &[u64],
        budget: u64,
    ) -> String {
        let Ok(plan) = Plan::of(tape, schedule, bytes, Budget::new(budget)) else {
            return format!("{case}/{budget}: 放不下");
        };
        let saved: Vec<u32> = plan.saved().iter().map(|var| var.id()).collect();
        let release: Vec<Option<usize>> =
            plan.saved().iter().map(|var| plan.release(*var)).collect();
        format!(
            "{case}/{budget}: {:?} saved={saved:?} release={release:?} replayed={} peak={}",
            plan.choice(),
            plan.replayed(),
            plan.peak()
        )
    }

    /// 改造重构前的实测值：同一组图形与预算必须逐字相同。
    const FROZEN: [&str; 3] = [
        "chain16/1048576: KeepAll saved=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] release=[Some(15), Some(14), Some(13), Some(12), Some(11), Some(10), Some(9), Some(8), Some(7), Some(6), Some(5), Some(4), Some(3), Some(2), Some(1), Some(0)] replayed=0 peak=256",
        "chain16/200: Recompute saved=[1, 3, 6, 8, 11, 14, 16] release=[Some(15), Some(13), Some(10), Some(8), Some(5), Some(2), Some(0)] replayed=12 peak=144",
        "chain16/0: 放不下",
    ];

    #[test]
    fn the_plan_numbers_are_frozen() {
        let mut lines = Vec::new();
        let (tape, schedule) = chain(16);
        for budget in [1 << 20, 200, 0] {
            let bytes = uniform(&tape, 16);
            lines.push(summarize("chain16", &tape, &schedule, &bytes, budget));
        }
        println!("{}", lines.join("\n"));
        assert_eq!(lines, FROZEN);
    }
}
