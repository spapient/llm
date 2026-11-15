//! 重放执行：[`Replay`] 把 [`Plan`] 编译成一条可执行的指令序列，并驱动整次反向。
//!
//! 反向第 `k` 步之前重算哪几步（拓扑序）、之后交还哪些激活，构造时编成两张扁平区间表，执行时
//! 每步只是两次切片遍历——稳定态不分配、不递归、不查哈希。数值不在本 crate：[`Store`] 是唯一的
//! 口子，真实缓冲在 `train`，测试里是一张标量表，于是"重算出来的值对不对"可以逐位对拍。
//!
//! 释放时机照 [`Plan::release`] 走：计划已经把重算需求沿被丢弃的链传播过了（见 `plan.rs` 的
//! 账本），`Replay` 不再做第二份判断。承诺的步数是 [`Plan::replayed`]，实际值只会更小（共享祖先
//! 在同一步里只展开一次）；一旦更大就报错，而不是继续跑。

use diagnostic::Result;

use crate::error::{internal, invalid};
use crate::{Node, Plan, Schedule, Tape, Var};

/// 构造与执行各一个诊断站点。
const BUILD: &str = "autograd::Replay::of";
const RUN: &str = "autograd::Replay::run";

/// 激活值的存放：重放只决定"算哪一步、什么时候丢"，真的算与存由调用方给。
pub trait Store {
    /// 重算一步前向 `op(inputs) -> output`；输入此刻必须都在，失败由 [`Replay::run`] 原样上抛。
    fn compute(&mut self, op: &str, inputs: &[Var], output: Var) -> Result<()>;

    /// 交还一块激活：之后它的值不再被读，直到被重新算出来；实现不必区分检查点与临时值。
    fn free(&mut self, var: Var);
}

/// 一条可执行的重放序列：与反向调度逐步对齐的两张扁平区间表（稳定态不留下碎片）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Replay {
    compute: Vec<Var>,
    compute_at: Vec<u32>,
    free: Vec<Var>,
    free_at: Vec<u32>,
    expected: u64,
}

impl Replay {
    /// 把计划编译成指令序列：`tape` 给算子与输入、`schedule` 给反向顺序、`plan` 给保留与丢弃。
    ///
    /// # Errors
    ///
    /// 只有一个来源：计划丢掉了一个本次调度里无处重放的值——内部不变量被破坏，不是调用方参数错。
    pub fn of(tape: &Tape, schedule: &Schedule, plan: &Plan) -> Result<Self> {
        let mut builder = Builder::new(tape, schedule.steps().len(), plan);
        for (index, step) in schedule.steps().iter().enumerate() {
            builder.begin(index);
            for input in step.inputs() {
                builder.chain(*input)?;
            }
            builder.close(index);
        }
        Ok(builder.finish())
    }

    /// 计划承诺的重算步数：与 [`Plan::replayed`] 同值，执行前后都可以对账。
    pub fn expected(&self) -> u64 {
        self.expected
    }

    /// 编入的重算指令条数：同一激活在不同步重复展开会各记一条，供审计与容量预估。
    pub fn compute_count(&self) -> usize {
        self.compute.len()
    }

    /// 编入的释放指令条数：到寿激活逐块交还，一条指令对应一次 [`Store::free`]。
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// 按计划执行整次反向：先重算、再让调用方算这一步的梯度、最后交还到寿的激活。返回实际重算步数。
    ///
    /// # Errors
    ///
    /// 序列与调度不是同一张图（步骤数不一致）、[`Store`] 报错、或者实际重算超过了计划承诺。
    pub fn run<S, F>(
        &self,
        tape: &Tape,
        schedule: &Schedule,
        store: &mut S,
        mut backward: F,
    ) -> Result<u64>
    where
        S: Store,
        F: FnMut(&mut S, &Node) -> Result<()>,
    {
        self.matches(schedule)?;
        let computed = self.drive(tape, schedule, store, &mut backward)?;
        self.affirm(computed)
    }

    /// 逐步执行：先按需重算，再跑调用方的反向公式，最后交还到寿的激活。
    fn drive<S, F>(
        &self,
        tape: &Tape,
        schedule: &Schedule,
        store: &mut S,
        backward: &mut F,
    ) -> Result<u64>
    where
        S: Store,
        F: FnMut(&mut S, &Node) -> Result<()>,
    {
        let mut computed = 0u64;
        for (index, step) in schedule.steps().iter().enumerate() {
            computed += self.recompute(tape, store, index)?;
            backward(store, step)?;
            // 交还放在反向公式之后：一个激活的寿命就是"直到最后一次被需要"。
            slice(&self.free, &self.free_at, index)
                .iter()
                .for_each(|var| store.free(*var));
        }
        Ok(computed)
    }

    /// 序列与调度必须是同一张图：步骤数对不上就拒绝执行。
    fn matches(&self, schedule: &Schedule) -> Result<()> {
        if schedule.steps().len() + 1 == self.compute_at.len() {
            return Ok(());
        }
        Err(invalid(
            RUN,
            format!(
                "重放序列按 {} 步编排，这次调度有 {} 步：序列与调度不是同一张图",
                self.compute_at.len() - 1,
                schedule.steps().len()
            ),
        ))
    }

    /// 第 `index` 步之前要重算的全部前向步，返回本步重算的个数。
    fn recompute<S: Store>(&self, tape: &Tape, store: &mut S, index: usize) -> Result<u64> {
        let mut computed = 0u64;
        for &var in slice(&self.compute, &self.compute_at, index) {
            let node = tape
                .node(var)
                .ok_or_else(|| internal(RUN, format!("变量 {} 没有生产者，无法重算", var.id())))?;
            store.compute(node.op(), node.inputs(), var)?;
            computed += 1;
        }
        Ok(computed)
    }

    /// 实际重算不能超过承诺：一旦超过，说明计划低估了代价。
    fn affirm(&self, computed: u64) -> Result<u64> {
        if computed <= self.expected {
            return Ok(computed);
        }
        Err(internal(
            RUN,
            format!(
                "实际重算 {computed} 步，超过计划承诺的 {} 步",
                self.expected
            ),
        ))
    }
}

/// 第 `index` 步的区间：两张表共用同一套下标。
fn slice<'a>(table: &'a [Var], at: &[u32], index: usize) -> &'a [Var] {
    let start = at[index] as usize;
    let end = at[index + 1] as usize;
    &table[start..end]
}

/// 构造期状态：一次用完就丢，所以这里可以随便分配，执行期才是零分配的那部分。
struct Builder<'a> {
    tape: &'a Tape,
    plan: &'a Plan,
    /// `stamp` 换代做去重：同一步里共享的祖先只展开一次，否则扇形图会退化成平方复杂度。
    stamp: Vec<u32>,
    mark: u32,
    stack: Vec<(Var, bool)>,
    /// `(反向步, 变量)`：保留的检查点到寿、以及重算出来的临时值，最后一起排序成区间表。
    frees: Vec<(u32, Var)>,
    current: u32,
    steps: usize,
    replay: Replay,
}

impl<'a> Builder<'a> {
    fn new(tape: &'a Tape, steps: usize, plan: &'a Plan) -> Self {
        Self {
            tape,
            plan,
            stamp: vec![0; tape.vars() as usize],
            mark: 0,
            stack: Vec::new(),
            frees: Vec::new(),
            current: 0,
            steps,
            replay: Replay {
                compute: Vec::new(),
                compute_at: vec![0; steps + 1],
                free: Vec::new(),
                free_at: vec![0; steps + 1],
                expected: plan.replayed(),
            },
        }
    }

    /// 进入反向第 `index` 步：换代 `stamp`，于是去重只在同一步内生效。
    fn begin(&mut self, index: usize) {
        self.current = index as u32;
        self.mark = self.mark.wrapping_add(1);
        if self.mark == 0 {
            self.stamp.fill(0);
            self.mark = 1;
        }
    }

    /// 本步的重算序列到此为止，记下区间右端。
    fn close(&mut self, index: usize) {
        self.replay.compute_at[index + 1] = self.replay.compute.len() as u32;
    }

    /// 保证 `root` 的值在这一步可用：被丢弃的输入按前向拓扑序重算，保留的检查点与叶子不动。
    fn chain(&mut self, root: Var) -> Result<()> {
        self.stack.clear();
        self.stack.push((root, false));
        while let Some((var, expanded)) = self.stack.pop() {
            if expanded {
                // 出栈的是"输入都已就绪"的节点：轮到它自己了。
                self.replay.compute.push(var);
                self.frees.push((self.current, var));
                continue;
            }
            self.visit(var)?;
        }
        Ok(())
    }

    /// 访问一个输入：叶子与检查点不重算，被丢弃的节点展开成"输入先、自身后"的拓扑序。
    fn visit(&mut self, var: Var) -> Result<()> {
        // 叶子是参数与输入：常驻，不进激活预算，也不必重算。
        let Some(node) = self.tape.node(var) else {
            return Ok(());
        };
        // 保留的检查点就是重放的起点：它的值从反向一开始就在。
        if self.plan.keep(var) {
            return Ok(());
        }
        if !node.requirement().requires() {
            return Err(internal(
                BUILD,
                format!(
                    "变量 {} 被丢弃了，但它不在反向图上、这次调度里没有一步能重放它",
                    var.id()
                ),
            ));
        }
        let slot = var.id() as usize;
        if self.stamp[slot] == self.mark {
            return Ok(());
        }
        self.stamp[slot] = self.mark;
        // 先登记自身、再压输入：出栈顺序是"输入先、自身后"，也就是合法的前向拓扑序。
        self.stack.push((var, true));
        self.stack
            .extend(node.inputs().iter().rev().map(|input| (*input, false)));
        Ok(())
    }

    /// 收尾：保留的检查点按计划给的释放点入表，两张表一次排序成型。
    fn finish(mut self) -> Replay {
        for var in self.plan.saved() {
            if let Some(step) = self.plan.release(*var) {
                self.frees.push((step as u32, *var));
            }
        }
        // 按 (步骤, 编号) 排序：每一步内部顺序稳定，同一张图每次编出来的序列都相同。
        self.frees
            .sort_unstable_by_key(|(step, var)| (*step, var.id()));
        let mut start = 0usize;
        for step in 0..=self.steps {
            let count = self.frees[start..]
                .iter()
                .take_while(|(at, _)| *at as usize == step)
                .count();
            self.replay.free_at[step] = start as u32;
            self.replay
                .free
                .extend(self.frees[start..start + count].iter().map(|(_, var)| *var));
            start += count;
        }
        debug_assert_eq!(start, self.frees.len(), "释放点越过了最后一步");
        self.replay
    }
}

#[cfg(test)]
mod tests;
