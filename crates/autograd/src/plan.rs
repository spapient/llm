//! 显存计划：[`Plan`] 把"反向图 + 每变量字节数 + 预算"变成一组显存决策。
//!
//! 这一层只做算术：不分配、不提交 kernel、也不假装自己会重放。它回答三个问题——
//!
//! 1. **留谁**：[`Plan::saved`] 是反向开始前就必须存在的激活（检查点）；
//! 2. **何时释放**：[`Plan::release`] 是每个保留的激活在反向走到第几步之后可以还给分配器。
//!    释放点由"最后一次被需要"决定——既包括直接消费它的步骤，也包括以它为起点重算的更晚步骤；
//!    不按作用域，这是显存收益的唯一来源；
//! 3. **代价多大**：[`Plan::replayed`] 是一次反向里要重算的步数，[`Plan::peak`] 是按本计划
//!    执行的峰值字节数，`peak <= budget` 由构造保证。
//!
//! # 模型
//!
//! 一个节点的输出值只在"把它当输入的、loss 可达的反向步骤"里被需要。于是每个激活的寿命都是
//! 从前向结束到它**最后一次**被消费；全部保留时的峰值就是这些激活的字节和。预算不够时按段切：
//! 每隔若干前向步留一个检查点，其余丢弃、需要时就地重放。段长由预算解出来，不是常数——
//! 旧实现里写死的 256 只是这条公式的一个特例。
//!
//! 峰值是**保守上界**：保留的激活按整段反向都活着计，因此 `peak <= budget` 不会低估真实占用。
//! 反过来，每个候选计划都用同一份账核对过，[`Plan::of`] 不会给出实际会超预算的计划。
//!
//! # 为什么不声称最优
//!
//! 这是图上的 pebbling 问题：一般 DAG 的最优解是 PSPACE 难的，任何"我最省"的说法都无法验证。
//! 所以这里给的是三个可核对的数：保守峰值、[`Plan::floor`]（任何计划都绕不过的下界）、以及
//! 重放步数。等字节时峰值约为 `total/√n` 的两倍，取的就是这个量级。

use crate::error::invalid;
use crate::{Schedule, Tape, Var};
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

/// 变量形状表：记录前向时顺手登记，[`Plan`] 直接换算成字节。
///
/// 只存形状与元素类型，不存数据——激活本身仍然归运行时缓冲，这里不复制任何张量。
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
    ///
    /// 计划只关心字节数（[`Footprint`]），而真正要把值写进一块内存的一方还需要知道"形状多大、
    /// 按哪种元素读写"——分配缓冲需要的就是这两个事实。
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

/// 显存预算（字节）。
///
/// 只是一个数：预算从哪来（配置、设备容量、上一次的峰值）不归这一层管，所以不读环境变量、
/// 也不读 TOML——配置在 `llm-config`，设备容量在 `prof`。
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

/// 一次反向外加前向重放时的显存决策。
///
/// 三个数一起看才有意义：`saved` 少了会推高 `replayed`，`replayed` 少了会推高 `peak`。
/// 计划是纯数据（可比较、可缓存）；把计划变成动作的是 [`crate::Replay`]，它按这些释放点
/// 编排重算与交还，值由调用方的 [`crate::Store`] 提供。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    choice: Choice,
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
    /// `tape` 用来区分"参数/输入"（叶子，常驻，不进预算）与算子输出；`schedule` 决定"谁在图上、
    /// 按什么顺序反向、每一步要哪些输入"；`footprint` 决定每个激活多大；`budget` 是可用的显存
    /// 字节数。
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
    /// 预算低于必然下界（一次算子的输入必须同时在场），或者低于任何分段方案能取到的最小
    /// 峰值时返回 [`diagnostic::Code::Invalid`]：这是"这次请求与这张图不相容"，不是缺能力。
    /// 宁可报"放不下"，也不给出实际会超预算的计划。
    pub fn of(
        tape: &Tape,
        schedule: &Schedule,
        footprint: impl Footprint,
        budget: Budget,
    ) -> Result<Self> {
        let facts = Facts::new(tape, schedule, &footprint);
        if facts.total <= budget.bytes() {
            return Ok(facts.keep_all(tape, schedule, budget));
        }
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
        facts.plan(tape, schedule, budget)
    }

    /// 走的是哪条路。
    pub fn choice(&self) -> Choice {
        self.choice
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
    ///
    /// 释放点是它的**最后一次被需要**：既包括直接消费它的反向步骤，也包括以它为起点重算的更晚
    /// 步骤。后一种不是冗余——一个变量可能一边被深层节点直接消费、一边是某个更浅的被丢弃值的
    /// 祖先，后者排得更晚；漏掉它就会在还需要的时候把检查点还回去。
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

/// 从调度读出的事实：全部只读，构造一次，之后每次评估计划都复用它。
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
    /// 该激活能否丢弃后重放：字节数大于 0，并且本次调度里有它的生产者。
    ///
    /// 零字节的激活不值得丢——留着不占预算，丢了只烧算力；detach 出去的值则是无处重放。
    /// 两种都由 [`Facts::candidates`] 的保留路径兜住，不必让调用方在形状表里补 0。
    droppable: Vec<bool>,
    /// 全留时的峰值：所有候选的字节和。
    total: u64,
    /// 必然下界：某一步同时要放下的输入字节和的最大值。
    floor: u64,
}

impl Facts {
    fn new<F: Footprint + ?Sized>(tape: &Tape, schedule: &Schedule, footprint: &F) -> Self {
        let steps = schedule.steps();
        // 按**整张图**的变量数开表，而不是按调度里出现过的最大编号：释放点的传播要沿被丢弃的
        // 链往上走，链上可能有本次调度没排进去的祖先——它们的编号可能超出调度的上界。
        let vars = tape.vars();
        let len = vars as usize;

        let mut last_use = vec![NONE; len];
        let mut producer = vec![NONE; len];
        for (index, step) in steps.iter().enumerate() {
            let output = step.output().id() as usize;
            producer[output] = index as u32;
            for input in step.inputs() {
                // 反向顺序遍历，最后一次覆盖就是"最后一次被消费"。
                last_use[input.id() as usize] = index as u32;
            }
        }

        // 候选 = 被调度消费过的**算子输出**：叶子是权重与输入（常驻），loss 的输出没人消费。
        // 用 `tape` 判断"是不是算子输出"，而不是让调用方在形状表里补 0——少一个出错的口子。
        let candidate: Vec<bool> = (0..vars)
            .map(|id| last_use[id as usize] != NONE && tape.node(Var::at(id)).is_some())
            .collect();
        let bytes: Vec<u64> = (0..vars)
            .map(|id| match candidate[id as usize] {
                true => footprint.bytes(Var::at(id)),
                false => 0,
            })
            .collect();
        // detach 出去的值仍是真实缓冲，但本次调度里没有"重放它"的一步：只能留，不能丢。
        // 零字节的激活同理反过来：丢它不省任何预算，只多算一遍。
        let droppable: Vec<bool> = (0..vars)
            .map(|id| {
                candidate[id as usize] && bytes[id as usize] > 0 && producer[id as usize] != NONE
            })
            .collect();
        let total = bytes.iter().copied().fold(0u64, u64::saturating_add);

        let candidates: Vec<u32> = (0..vars).filter(|id| candidate[*id as usize]).collect();
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

        // 下界：同一步的输入必须同时在场。重复出现的同一个输入只算一次（只有一块缓冲）。
        let mut stamp = vec![0u32; len];
        let mut floor = 0u64;
        for (index, step) in steps.iter().enumerate() {
            let mark = index as u32 + 1;
            let mut sum = 0u64;
            for input in step.inputs() {
                let slot = input.id() as usize;
                if stamp[slot] == mark {
                    continue;
                }
                stamp[slot] = mark;
                sum = sum.saturating_add(bytes[slot]);
            }
            floor = floor.max(sum);
        }

        Self {
            bytes,
            last_use,
            producer,
            candidates,
            order,
            prefix,
            droppable,
            total,
            floor,
        }
    }

    /// 全留：预算装得下时这是唯一不引入重放的计划。
    fn keep_all(&self, tape: &Tape, schedule: &Schedule, budget: Budget) -> Plan {
        let mut keep = vec![false; self.bytes.len()];
        for id in &self.candidates {
            keep[*id as usize] = true;
        }
        let ledger = self.account(tape, schedule, &keep);
        self.assemble(keep, ledger, budget)
    }

    /// 分段：段数由预算解出来，段界上的激活保留，段内的丢弃后重放。
    fn plan(&self, tape: &Tape, schedule: &Schedule, budget: Budget) -> Result<Plan> {
        if self.order.is_empty() {
            // total > budget 且 order 为空只可能是 total 与 budget 都为 0，已被全留路径拦下。
            return Err(invalid(SITE, "没有需要保留的激活，但预算为 0"));
        }
        // 均匀等字节切分下峰值 ≈ k·(total/n) + total/k，在 k = √n 处取到最小；预算另外给出
        // 一个"保留量不超过预算一半"的估计。两者各扫 ±2 档，足够覆盖字节分布不均的情形。
        let count = self.order.len();
        let root = (count as f64).sqrt() as usize;
        let half = (budget.bytes() / 2).max(1);
        let quota = self.total.div_ceil(half) as usize;

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

        let mut best: Option<(u64, Vec<bool>, Ledger)> = None;
        let mut feasible: Option<(usize, Vec<bool>, Ledger)> = None;
        for segments in &trials {
            let keep = self.segmented(*segments);
            let ledger = self.account(tape, schedule, &keep);
            if ledger.peak <= budget.bytes() {
                // 都能装下时取段数最多的一份：保留多一些，重放少一些。
                if feasible
                    .as_ref()
                    .is_none_or(|(seen, _, _)| *segments > *seen)
                {
                    feasible = Some((*segments, keep, ledger));
                }
            } else if best.as_ref().is_none_or(|(peak, _, _)| ledger.peak < *peak) {
                best = Some((ledger.peak, keep, ledger));
            }
        }

        match feasible {
            Some((_, keep, ledger)) => Ok(self.assemble(keep, ledger, budget)),
            None => {
                let minimum = best.map_or(self.total, |(peak, _, _)| peak);
                Err(invalid(
                    SITE,
                    format!(
                        "预算 {} 字节放不下：{} 个激活共 {} 字节，最小峰值约 {} 字节（下界 {} 字节）",
                        budget.bytes(),
                        count,
                        self.total,
                        minimum,
                        self.floor
                    ),
                ))
            }
        }
    }

    /// 分段后的保留集：段界上是检查点，不可丢弃的激活（detach 出去的值、零字节的）无条件保留。
    fn segmented(&self, segments: usize) -> Vec<bool> {
        let mut keep = self.boundaries(segments);
        for id in &self.candidates {
            let slot = *id as usize;
            if !self.droppable[slot] {
                keep[slot] = true;
            }
        }
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
        let mut retained = 0u64;
        for id in &self.order {
            if keep[*id as usize] {
                retained = retained.saturating_add(self.bytes[*id as usize]);
            }
        }

        // 重放一个激活要先重放它的、被丢弃的输入；输入编号必然小于输出，所以前向一遍即可。
        let mut wide = vec![0u64; self.bytes.len()];
        let mut work = vec![0u64; self.bytes.len()];
        for id in &self.order {
            let slot = *id as usize;
            if keep[slot] {
                continue;
            }
            let step = &schedule.steps()[self.producer[slot] as usize];
            let mut width = 0u64;
            let mut steps = 1u64;
            for input in step.inputs() {
                let source = input.id() as usize;
                if keep[source] {
                    continue;
                }
                width = width.max(wide[source]);
                steps = steps.saturating_add(work[source]);
            }
            wide[slot] = self.bytes[slot].saturating_add(width);
            work[slot] = steps;
        }

        // 每一步的瞬时工作集：它要的、被丢弃的输入要同时在场。同一个输入重复出现只算一次。
        let mut stamp = vec![0u32; self.bytes.len()];
        let mut transient = 0u64;
        let mut replayed = 0u64;
        for (index, step) in schedule.steps().iter().enumerate() {
            let mark = index as u32 + 1;
            let mut sum = 0u64;
            for input in step.inputs() {
                let slot = input.id() as usize;
                if keep[slot] || stamp[slot] == mark {
                    continue;
                }
                stamp[slot] = mark;
                sum = sum.saturating_add(wide[slot]);
                replayed = replayed.saturating_add(work[slot]);
            }
            transient = transient.max(sum);
        }

        // 释放点从"直接的最后一次消费"出发，再沿被丢弃的链往上传播：某个值要重算，链上所有
        // 仍然保留的祖先都得活到那一步。节点编号沿边严格递增，所以按编号降序走一遍即可，
        // 每个消费者都已经处理完（重算需求只会来自更高的编号）。
        let mut release = self.last_use.clone();
        for node in tape.nodes().iter().rev() {
            let output = node.output().id() as usize;
            if keep[output] || release[output] == NONE {
                continue;
            }
            let needed = release[output];
            for input in node.inputs() {
                let slot = input.id() as usize;
                if self.bytes[slot] > 0 && (release[slot] == NONE || release[slot] < needed) {
                    release[slot] = needed;
                }
            }
        }

        Ledger {
            retained,
            peak: retained.saturating_add(transient),
            replayed,
            release,
        }
    }

    /// 把保留集装配成计划：释放点来自账本对"最后一次需要"的传播。
    fn assemble(&self, keep: Vec<bool>, ledger: Ledger, budget: Budget) -> Plan {
        let mut saved = Vec::new();
        for id in &self.candidates {
            if keep[*id as usize] {
                saved.push(Var::at(*id));
            }
        }
        // 只看候选：叶子与 loss 的输出从来不是保留对象，不该影响"是否走了重放"。
        let kept = self.order.iter().filter(|id| keep[**id as usize]).count();
        let choice = match kept == self.order.len() {
            true => Choice::KeepAll,
            false => Choice::Recompute,
        };
        Plan {
            choice,
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
    use super::*;
    use crate::{Requirement, Tape};
    use diagnostic::Code;
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
            for _ in 0..1 + rng.below(24) {
                let width = 1 + rng.below(3) as usize;
                let inputs: Vec<Var> = (0..width)
                    .map(|_| vars[rng.below(vars.len() as u32) as usize])
                    .collect();
                let recorded = match rng.below(10) {
                    0 => tape.record_with("detach", &inputs, Requirement::None),
                    _ => tape.record("edge", &inputs),
                };
                if let Ok(var) = recorded {
                    vars.push(var);
                }
            }
            let Some(loss) = vars
                .iter()
                .rev()
                .find(|var| tape.requirement(**var).is_some_and(Requirement::requires))
                .copied()
            else {
                continue;
            };
            let Ok(schedule) = tape.backward(loss) else {
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
}
