//! 计划的事实与账本：`Facts` 只读调度与字节表，`Ledger` 汇总一次保留集的代价；
//! 段数由预算解出，段界保留、段内重放的分段求解也在本模块。

use diagnostic::Result;

use crate::graph::{Node, Var};
use crate::invalid;
use crate::plan::Plan;
use crate::plan::{Budget, Choice, Footprint, MemoryPolicy, SITE};
use crate::schedule::Schedule;
use crate::tape::Tape;

/// "没有"：既表示变量没有生产者，也表示它没有被消费过。
pub(crate) const NONE: u32 = u32::MAX;

/// 从调度读出的事实：只读，构造一次，之后每次评估计划都复用它。
pub(crate) struct Facts {
    /// 每个变量的字节数；不是候选（叶子、没有被消费的节点输出）为 0。
    pub(crate) bytes: Vec<u64>,
    /// 每个变量最后一次被消费的反向步号；没被消费为 [`NONE`]。
    last_use: Vec<u32>,
    /// 每个变量的生产者（`schedule.steps()` 的下标）。
    pub(crate) producer: Vec<u32>,
    /// 全部候选激活，按前向顺序（编号升序）。
    pub(crate) candidates: Vec<u32>,
    /// 候选里字节数大于 0 的那部分：分段只按它们切，零字节的没有可切的东西。
    pub(crate) order: Vec<u32>,
    /// `order` 的字节前缀和，含自身。
    pub(crate) prefix: Vec<u64>,
    /// 全留时的峰值：所有候选的字节和。
    pub(crate) total: u64,
    /// 必然下界：某一步同时要放下的输入字节和的最大值。
    pub(crate) floor: u64,
}

impl Facts {
    /// 一遍扫描定"生产者与最后一次消费"，再按候选算字节、前缀和与下界。
    pub(crate) fn new(tape: &Tape, schedule: &Schedule, footprint: &impl Footprint) -> Self {
        let steps = schedule.steps();
        let vars = tape.vars();
        // 按**整张图**的变量数开表：释放点的传播会沿被丢弃的链往上走，链上可能有调度外的祖先。
        let (last_use, producer) = uses(steps, vars as usize);
        // 候选 = 被调度消费过的**算子输出**：叶子常驻、loss 的输出没人消费，都不进预算。
        // 一趟扫描同时落字节表与候选表：`node` 查找只做一次；双输出向量无法在单趟迭代里
        // 同时收集，故保留循环（各变量判定彼此无串行依赖）。
        let mut bytes = vec![0u64; vars as usize];
        let mut candidates: Vec<u32> = Vec::with_capacity(vars as usize);
        for id in 0..vars {
            let var = Var::at(id);
            if last_use[id as usize] != NONE && tape.node(var).is_some() {
                let slot = id as usize;
                bytes[slot] = footprint.bytes(var);
                candidates.push(id);
            }
        }
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
    pub(crate) fn keep_all(mut self, budget: Budget, policy: MemoryPolicy) -> Plan {
        let mut keep = vec![false; self.bytes.len()];
        self.candidates
            .iter()
            .for_each(|id| keep[*id as usize] = true);
        // 没有丢弃的输入：瞬时工作集与重放步数都是 0，释放点就是直接的最后一次消费，
        // 沿被丢弃的链传播在这里必定是空转，可以整段省掉。
        let ledger = Ledger {
            retained: self.total,
            peak: self.total,
            replayed: 0,
            release: std::mem::take(&mut self.last_use),
        };
        self.assemble(keep, ledger, budget, policy)
    }

    /// 只算代价，不算释放点：分段试算只关心峰值，释放点留给最终选中的方案。
    pub(crate) fn measure(
        &self,
        schedule: &Schedule,
        keep: &[bool],
        scratch: &mut Scratch,
    ) -> Ledger {
        self.widths(schedule, keep, scratch);
        let (transient, replayed) = self.transient(
            schedule,
            keep,
            &scratch.wide,
            &scratch.work,
            &mut scratch.stamp,
        );
        let retained = self
            .order
            .iter()
            .filter(|id| keep[**id as usize])
            .fold(0u64, |sum, id| sum.saturating_add(self.bytes[*id as usize]));
        Ledger {
            retained,
            peak: retained.saturating_add(transient),
            replayed,
            release: Vec::new(),
        }
    }

    /// 重算一个激活要先重算它被丢弃的输入：`wide` 是瞬时占用，`work` 是步数。
    fn widths(&self, schedule: &Schedule, keep: &[bool], scratch: &mut Scratch) {
        // 输入编号必然小于输出，所以前向一遍即可。
        scratch.wide.fill(0);
        scratch.work.fill(0);
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
                width = width.max(scratch.wide[source]);
                chain = chain.saturating_add(scratch.work[source]);
            }
            scratch.wide[slot] = self.bytes[slot].saturating_add(width);
            scratch.work[slot] = chain;
        }
    }

    /// 每一步的瞬时工作集与重放步数：它要的、被丢弃的输入要同时在场。
    fn transient(
        &self,
        schedule: &Schedule,
        keep: &[bool],
        wide: &[u64],
        work: &[u64],
        stamp: &mut [u32],
    ) -> (u64, u64) {
        stamp.fill(0);
        let mut transient = 0u64;
        let mut replayed = 0u64;
        for (idx, step) in schedule.steps().iter().enumerate() {
            let mut sum = 0u64;
            each_input(step, stamp, idx as u32 + 1, |slot| {
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
    pub(crate) fn release(&self, tape: &Tape, keep: &[bool]) -> Vec<u32> {
        let mut release = self.last_use.clone();
        // 节点编号沿边严格递增，所以按编号降序走一遍即可：重算需求只会来自更高的编号。
        for node in tape.nodes().iter().rev() {
            carry(&mut release, node, keep, &self.bytes);
        }
        release
    }

    /// 把保留集装配成计划：释放点来自账本对"最后一次需要"的传播。
    pub(crate) fn assemble(
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

    /// 分段：段数由预算解出来，段界上的激活保留，段内的丢弃后重放。
    pub(crate) fn plan(
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
        let mut scratch = Scratch::new(self.bytes.len());
        let mut keep = vec![false; self.bytes.len()];
        let mut fitted: Option<(usize, Ledger)> = None;
        let mut minimum: Option<u64> = None;
        for segments in self.trials(budget) {
            self.segmented(segments, &mut keep);
            let ledger = self.measure(schedule, &keep, &mut scratch);
            minimum = Some(minimum.map_or(ledger.peak, |value| value.min(ledger.peak)));
            if ledger.peak <= budget.bytes() {
                // 试算按段数升序，后覆盖前：留下能装下的段数最多的一份。
                fitted = Some((segments, ledger));
            }
        }
        match fitted {
            Some((segments, mut ledger)) => {
                self.segmented(segments, &mut keep);
                ledger.release = self.release(tape, &keep);
                Ok(self.assemble(keep, ledger, budget, policy))
            }
            None => Err(invalid(
                SITE,
                format!(
                    "预算 {} 字节放不下：{} 个激活共 {} 字节，最小峰值约 {} 字节（下界 {} 字节）",
                    budget.bytes(),
                    self.order.len(),
                    self.total,
                    minimum.unwrap_or(self.total),
                    self.floor
                ),
            )),
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
    fn segmented(&self, segments: usize, keep: &mut [bool]) {
        keep.fill(false);
        self.boundaries(segments, keep);
        self.candidates
            .iter()
            .filter(|id| self.bytes[**id as usize] == 0 || self.producer[**id as usize] == NONE)
            .for_each(|id| keep[*id as usize] = true);
    }

    /// 按字节数均匀地放 `segments` 个检查点：段界落在前缀和最接近等分点的那个激活上。
    fn boundaries(&self, segments: usize, keep: &mut [bool]) {
        let last = self.order.len() - 1;
        let total = self.total.max(1);
        for idx in 1..=segments {
            let target = (u128::from(total) * idx as u128 / segments as u128) as u64;
            let position = self.prefix.partition_point(|sum| *sum < target).min(last);
            keep[self.order[position] as usize] = true;
        }
        // 第一个激活没有更早的检查点可依，必须留；否则第一段无从重放。
        keep[self.order[0] as usize] = true;
    }
}

/// 分段试算复用的缓冲：并行数组只按变量数分配一次。
pub(crate) struct Scratch {
    wide: Vec<u64>,
    work: Vec<u64>,
    stamp: Vec<u32>,
}

impl Scratch {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            wide: vec![0; len],
            work: vec![0; len],
            stamp: vec![0; len],
        }
    }
}

/// 一个保留集的账：所有数字必须来自同一次统计，否则峰值、重放步数与释放点会对不上。
#[derive(Debug, Clone)]
pub(crate) struct Ledger {
    pub(crate) retained: u64,
    pub(crate) peak: u64,
    pub(crate) replayed: u64,
    /// 每个变量最后一次被需要（直接消费或作为重算起点）的反向步号。
    pub(crate) release: Vec<u32>,
}

/// 每个变量的生产者与最后一次被消费的反向步号；没被消费为 [`NONE`]。
fn uses(steps: &[Node], len: usize) -> (Vec<u32>, Vec<u32>) {
    let mut last_use = vec![NONE; len];
    let mut producer = vec![NONE; len];
    for (idx, step) in steps.iter().enumerate() {
        producer[step.output().id() as usize] = idx as u32;
        for input in step.inputs() {
            // 反向顺序遍历，最后一次覆盖就是"最后一次被消费"。
            last_use[input.id() as usize] = idx as u32;
        }
    }
    (last_use, producer)
}

/// 必然下界：某一步同时要放下的输入字节和的最大值。
fn floor_of(steps: &[Node], bytes: &[u64]) -> u64 {
    let mut stamp = vec![0u32; bytes.len()];
    let mut floor = 0u64;
    for (idx, step) in steps.iter().enumerate() {
        let mut sum = 0u64;
        each_input(step, &mut stamp, idx as u32 + 1, |slot| {
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
