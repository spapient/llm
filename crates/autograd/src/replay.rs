//! 重放执行：[`Replay`] 把 [`Plan`] 编译成一条可执行的指令序列，并驱动整次反向。
//!
//! [`Plan`] 只回答"留谁、何时释放、重放多少步"；这一层把它变成**可执行**的东西：反向第 `k` 步
//! 之前先重算哪几步（拓扑序），执行完之后交还哪些激活。序列在构造时一次算好，执行时每一步只是
//! 两次切片遍历——稳定态不分配、不递归、不查哈希。
//!
//! # 谁来算
//!
//! 数值不在本 crate：[`Store`] 是唯一的口子。`compute` 拿算子名与输入变量、写出输出变量，
//! `free` 交还一块激活；真实缓冲在 `train`，测试里是一张标量表，于是"重算出来的值对不对"
//! 可以逐位对拍。
//!
//! # 为什么释放时机是安全的
//!
//! [`Plan::release`] 给的是"最后一次被需要"的步骤。这里有个容易漏掉的情形：一个检查点可能
//! 一边被深层节点直接消费、一边是某个更浅的被丢弃值的祖先——后者排得更晚。计划已经把重算需求
//! 沿被丢弃的链传播过了（见 `plan.rs` 的账本），所以只要照它的释放点走，就不会在还需要的时候
//! 把值还回去；`Replay` 本身不再做第二份判断。
//!
//! # 代价是可核对的
//!
//! [`Plan::replayed`] 是计划承诺的重算步数，[`Replay::run`] 返回实际执行的步数。共享的祖先在同
//! 一步里只展开一次，所以实际值只会更小；一旦更大，说明计划低估了代价，直接报错而不是继续跑。

use diagnostic::Result;

use crate::error::{internal, invalid};
use crate::{Node, Plan, Schedule, Tape, Var};

/// 构造与执行各一个诊断站点。
const BUILD: &str = "autograd::Replay::of";
const RUN: &str = "autograd::Replay::run";

/// 激活值的存放：重放只决定"算哪一步、什么时候丢"，真的算与存由调用方给。
///
/// 方法只有两个，语义都很窄：`compute` 用**当前**存着的输入值算出一步前向，`free` 交还一块
/// 激活。读值的口子不在 trait 里——反向公式本来就由调用方写，它掌握自己的缓冲布局。
pub trait Store {
    /// 重算一步前向：`op(inputs) -> output`，输入此刻必须都在。
    ///
    /// # Errors
    ///
    /// 输入缺失、设备失败、算子未知——由实现决定，[`Replay::run`] 原样向上抛。
    fn compute(&mut self, op: &str, inputs: &[Var], output: Var) -> Result<()>;

    /// 交还一块激活：之后它的值不再被读，直到被重新算出来。
    ///
    /// 交还的可能是保留的检查点，也可能是刚重算出来的临时值，实现不必区分。
    fn free(&mut self, var: Var);
}

/// 一条可执行的重放序列：与反向调度逐步对齐的两张区间表。
///
/// 反向第 `k` 步之前执行 `compute[k]`（前向拓扑序），之后交还 `free[k]`。两张表都是扁平的，
/// 第 `k` 步是两次切片遍历，没有逐步骤的 `Vec`——`Plan` 每步新建，重放序列也一样，稳定态
/// 不留下碎片。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Replay {
    compute: Vec<Var>,
    compute_at: Vec<u32>,
    free: Vec<Var>,
    free_at: Vec<u32>,
    expected: u64,
}

impl Replay {
    /// 把计划编译成指令序列。
    ///
    /// 只要三样东西：`tape`（重算时要知道每一步的算子与输入）、`schedule`（反向顺序与每一步
    /// 的输入）、`plan`（谁保留、谁丢弃、何时释放）。三者必须描述同一张图，否则 [`Replay::run`]
    /// 在第一步之前就会拒绝。
    ///
    /// # Errors
    ///
    /// 只有一个来源：计划丢掉了一个本次调度里无处重放的值。那说明计划与图不一致，属于内部
    /// 不变量被破坏，不是调用方写错了参数。
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

    /// 按计划执行整次反向：该重算的先重算，再让调用方算这一步的梯度，最后交还到寿的激活。
    ///
    /// `backward` 是调用方的反向公式：它拿到的变量值一定都在，因为重算已经在此之前做完。
    /// 交还放在它之后——一个激活的寿命就是"直到最后一次被需要"，不按作用域。
    ///
    /// 返回**实际**重算的前向步数：共享的祖先在同一步里只展开一次，所以它可以小于
    /// [`Replay::expected`]，但绝不会大于。
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
        if schedule.steps().len() + 1 != self.compute_at.len() {
            return Err(invalid(
                RUN,
                format!(
                    "重放序列按 {} 步编排，这次调度有 {} 步：序列与调度不是同一张图",
                    self.compute_at.len() - 1,
                    schedule.steps().len()
                ),
            ));
        }
        let mut computed = 0u64;
        for (index, step) in schedule.steps().iter().enumerate() {
            for &var in slice(&self.compute, &self.compute_at, index) {
                let node = tape.node(var).ok_or_else(|| {
                    internal(RUN, format!("变量 {} 没有生产者，无法重算", var.id()))
                })?;
                store.compute(node.op(), node.inputs(), var)?;
                computed += 1;
            }
            backward(store, step)?;
            for &var in slice(&self.free, &self.free_at, index) {
                store.free(var);
            }
        }
        if computed > self.expected {
            return Err(internal(
                RUN,
                format!(
                    "实际重算 {computed} 步，超过计划承诺的 {} 步",
                    self.expected
                ),
            ));
        }
        Ok(computed)
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
                self.replay.compute.push(var);
                self.frees.push((self.current, var));
                continue;
            }
            // 叶子是参数与输入：常驻，不进激活预算，也不必重算。
            let Some(node) = self.tape.node(var) else {
                continue;
            };
            // 保留的检查点就是重放的起点：它的值从反向一开始就在。
            if self.plan.keep(var) {
                continue;
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
                continue;
            }
            self.stamp[slot] = self.mark;
            // 先登记自身、再压输入：出栈顺序是"输入先、自身后"，也就是合法的前向拓扑序。
            self.stack.push((var, true));
            for input in node.inputs().iter().rev() {
                self.stack.push((*input, false));
            }
        }
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
        let mut cursor = 0usize;
        for step in 0..=self.steps {
            self.replay.free_at[step] = cursor as u32;
            while cursor < self.frees.len() && self.frees[cursor].0 as usize == step {
                self.replay.free.push(self.frees[cursor].1);
                cursor += 1;
            }
        }
        debug_assert_eq!(cursor, self.frees.len(), "释放点越过了最后一步");
        self.replay
    }
}

#[cfg(test)]
mod tests {
    use diagnostic::{Code, Error};

    use super::*;
    use crate::{Budget, Choice, Requirement};
    use testkit::Rng;

    /// 标量表：值的真值 + 当前活着的值 + 峰值。
    ///
    /// 三件事一起做：重算出来的值与真值逐位对拍、读已释放的值立刻报错、活着的字节数记峰值。
    /// 于是"重放执行"不是"跑完没崩"，而是"每一步都对得上"。
    struct Table {
        truth: Vec<f32>,
        bytes: Vec<u64>,
        values: Vec<Option<f32>>,
        live: u64,
        peak: u64,
        reads_of_dead: usize,
    }

    impl Table {
        /// 前向刚结束时的状态：叶子常驻，计划保留的激活都还在。
        ///
        /// 叶子（参数与输入）不算进 `live`：它们常驻，不属于激活预算，计划也不为它们记账，
        /// 于是实测峰值与 [`Plan::peak`] 可以直接比。
        fn new(tape: &Tape, plan: &Plan, truth: &[f32], bytes: &[u64]) -> Self {
            let mut values = vec![None; truth.len()];
            let mut live = 0u64;
            for id in 0..tape.vars() {
                let var = tape.var(id).expect("编号有效");
                let leaf = tape.node(var).is_none();
                values[id as usize] = (leaf || plan.keep(var)).then_some(truth[id as usize]);
                if plan.keep(var) {
                    live += bytes[id as usize];
                }
            }
            Self {
                truth: truth.to_vec(),
                bytes: bytes.to_vec(),
                values,
                live,
                peak: live,
                reads_of_dead: 0,
            }
        }

        fn value(&self, var: Var) -> Option<f32> {
            self.values.get(var.id() as usize).copied().flatten()
        }
    }

    impl Store for Table {
        fn compute(&mut self, op: &str, inputs: &[Var], output: Var) -> Result<()> {
            let mut args = [0.0f32; 4];
            if inputs.len() > args.len() {
                return Err(Error::new(
                    Code::Unsupported,
                    format!("测试引擎最多支持四个输入，实际 {}", inputs.len()),
                ));
            }
            for (slot, input) in inputs.iter().enumerate() {
                let Some(value) = self.value(*input) else {
                    self.reads_of_dead += 1;
                    return Err(Error::new(
                        Code::Internal,
                        format!("重算 {} 时读到已释放的变量 {}", output.id(), input.id()),
                    ));
                };
                args[slot] = value;
            }
            let value = eval(op, &args[..inputs.len()])?;
            let expected = self.truth[output.id() as usize];
            if value != expected {
                return Err(Error::new(
                    Code::Internal,
                    format!("重算 {} 得到 {value}，前向是 {expected}", output.id()),
                ));
            }
            self.values[output.id() as usize] = Some(value);
            self.live += self.bytes[output.id() as usize];
            self.peak = self.peak.max(self.live);
            Ok(())
        }

        fn free(&mut self, var: Var) {
            if self.values[var.id() as usize].take().is_some() {
                self.live -= self.bytes[var.id() as usize];
            }
        }
    }

    /// 反向那一步：读全部输入，读不到就当成"读已释放"。
    fn consume(table: &mut Table, step: &Node) -> Result<()> {
        for input in step.inputs() {
            if table.value(*input).is_none() {
                table.reads_of_dead += 1;
                return Err(Error::new(
                    Code::Internal,
                    format!(
                        "第 {} 步读到已释放的变量 {}",
                        step.output().id(),
                        input.id()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// 测试用的标量语义；与 `op` 里的真算子无关，只负责让数值随位置变化。
    fn eval(op: &str, args: &[f32]) -> Result<f32> {
        match (op, args) {
            ("add", [a, b]) => Ok(a + b),
            ("relu", [a]) => Ok(a.max(0.0)),
            ("reduce_sum", [a]) => Ok(*a),
            other => Err(Error::new(
                Code::Unsupported,
                format!("测试引擎没有算子 {other:?}"),
            )),
        }
    }

    /// 前向真值：叶子取种子，节点按前向顺序算一遍。
    fn forward(tape: &Tape, seeds: &[f32]) -> Vec<f32> {
        let mut values = vec![0.0f32; tape.vars() as usize];
        values[..seeds.len()].copy_from_slice(seeds);
        for node in tape.nodes() {
            let mut args = [0.0f32; 4];
            for (slot, input) in node.inputs().iter().enumerate() {
                args[slot] = values[input.id() as usize];
            }
            values[node.output().id() as usize] =
                eval(node.op(), &args[..node.inputs().len()]).expect("测试图只用已知算子");
        }
        values
    }

    /// 一条 `add(current, one)` 的链：每一步的值都不同，重算错了会被逐位对拍抓住。
    fn chain(length: usize) -> (Tape, Var) {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let one = tape.leaf();
        let mut current = x;
        for _ in 0..length {
            current = tape.record("add", &[current, one]).expect("记录");
        }
        let loss = tape.record("reduce_sum", &[current]).expect("记录");
        (tape, loss)
    }

    fn run_once(tape: &Tape, schedule: &Schedule, plan: &Plan, bytes: &[u64]) -> (u64, Table) {
        let seeds = [1.0, 1.0];
        let truth = forward(tape, &seeds);
        let mut table = Table::new(tape, plan, &truth, bytes);
        let replay = Replay::of(tape, schedule, plan).expect("可执行");
        assert_eq!(replay.expected(), plan.replayed(), "承诺值与计划同源");
        let computed = replay
            .run(tape, schedule, &mut table, consume)
            .expect("重放");
        (computed, table)
    }

    #[test]
    fn a_tight_budget_replays_exactly_what_the_plan_promised() {
        let (tape, loss) = chain(16);
        let schedule = tape.backward(loss).expect("在图上");
        let bytes = vec![8u64; tape.vars() as usize];
        let plan = Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(96)).expect("可分段");
        assert_eq!(plan.choice(), Choice::Recompute);

        let (computed, table) = run_once(&tape, &schedule, &plan, &bytes);
        assert_eq!(computed, plan.replayed(), "链上不共享前缀，实际就是承诺值");
        assert!(
            table.peak <= plan.peak(),
            "实测峰值 {} 超过计划 {}",
            table.peak,
            plan.peak()
        );
        assert!(table.peak <= plan.budget().bytes(), "峰值超预算");
        assert_eq!(table.live, 0, "执行完不该留下任何激活");
        assert_eq!(table.reads_of_dead, 0);
    }

    #[test]
    fn a_plan_that_fits_replays_nothing_but_still_frees_on_time() {
        let (tape, loss) = chain(4);
        let schedule = tape.backward(loss).expect("在图上");
        let bytes = vec![8u64; tape.vars() as usize];
        let plan =
            Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(1 << 20)).expect("放得下");
        assert_eq!(plan.choice(), Choice::KeepAll);

        let (computed, table) = run_once(&tape, &schedule, &plan, &bytes);
        assert_eq!(computed, 0, "装得下就不重放");
        assert_eq!(
            table.live, 0,
            "不重放也要按释放点还缓冲，否则显存收益就没了"
        );
        assert_eq!(table.reads_of_dead, 0);
    }

    #[test]
    fn a_checkpoint_lives_until_its_last_need() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let one = tape.leaf();
        let shared = tape.record("relu", &[x]).expect("记录");
        let left = tape.record("add", &[shared, one]).expect("记录");
        let right = tape.record("add", &[shared, one]).expect("记录");
        let joined = tape.record("add", &[left, right]).expect("记录");
        let loss = tape.record("reduce_sum", &[joined]).expect("记录");
        let schedule = tape.backward(loss).expect("在图上");
        let bytes = vec![8u64; tape.vars() as usize];
        let plan =
            Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(1 << 20)).expect("放得下");

        let truth = forward(&tape, &[1.0, 1.0]);
        let mut table = Table::new(&tape, &plan, &truth, &bytes);
        let replay = Replay::of(&tape, &schedule, &plan).expect("可执行");
        let mut live = Vec::new();
        replay
            .run(&tape, &schedule, &mut table, |table, step| {
                live.push(table.value(shared).is_some());
                consume(table, step)
            })
            .expect("重放");

        assert_eq!(
            live,
            [true, true, true, true, false],
            "共享的激活活到较晚的那一支，之后立刻还回去"
        );
        assert_eq!(plan.release(shared), Some(3));
    }

    /// 一个检查点可能既是深层节点的直接输入、又是更浅的被丢弃值的祖先：后者排得更晚。
    ///
    /// 计划必须把重算需求沿被丢弃的链传播上去，否则这个用例会在第 4 步读到已经释放的 `v`。
    #[test]
    fn a_checkpoint_that_anchors_a_later_replay_is_not_freed_early() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let one = tape.leaf();
        let v = tape.record("relu", &[x]).expect("记录");
        let w = tape.record("add", &[v, one]).expect("记录");
        let c1 = tape.record("add", &[w, one]).expect("记录");
        let c2 = tape.record("add", &[c1, one]).expect("记录");
        let deep = tape.record("add", &[v, c2]).expect("记录");
        let loss = tape.record("reduce_sum", &[deep]).expect("记录");
        let schedule = tape.backward(loss).expect("在图上");
        let bytes = vec![8u64; tape.vars() as usize];

        let mut saw_drop = false;
        for budget in [8u64, 16, 24, 32, 40, 56] {
            let Ok(plan) = Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(budget)) else {
                continue;
            };
            if !plan.keep(w) {
                saw_drop = true;
            }
            let (computed, table) = run_once(&tape, &schedule, &plan, &bytes);
            assert_eq!(table.reads_of_dead, 0, "预算 {budget}：释放早了");
            assert!(table.peak <= plan.peak(), "预算 {budget}：峰值超计划");
            assert_eq!(
                computed,
                plan.replayed(),
                "预算 {budget}：重算步数与承诺不符"
            );
        }
        assert!(saw_drop, "这组预算里必须有一个真丢掉了 w，否则用例没有意义");
    }

    #[test]
    fn a_deep_chain_neither_recurses_nor_blows_the_budget() {
        let (tape, loss) = chain(20_000);
        let schedule = tape.backward(loss).expect("在图上");
        let bytes = vec![8u64; tape.vars() as usize];
        let plan =
            Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(16_000)).expect("可分段");
        let (computed, table) = run_once(&tape, &schedule, &plan, &bytes);
        assert!(computed > 0);
        assert!(computed <= plan.replayed());
        assert!(
            table.peak <= plan.peak(),
            "峰值 {} 超过计划 {}",
            table.peak,
            plan.peak()
        );
        assert_eq!(table.live, 0);
    }

    #[test]
    fn a_sequence_from_another_schedule_is_rejected() {
        let (tape, loss) = chain(3);
        let schedule = tape.backward(loss).expect("在图上");
        let plan = Plan::of(
            &tape,
            &schedule,
            vec![8u64; tape.vars() as usize],
            Budget::new(1 << 20),
        )
        .expect("放得下");
        let replay = Replay::of(&tape, &schedule, &plan).expect("可执行");

        let mut other = Tape::new();
        let y = other.parameter();
        let long = other.record("relu", &[y]).expect("记录");
        let other_loss = other.record("reduce_sum", &[long]).expect("记录");
        let other_schedule = other.backward(other_loss).expect("在图上");
        let mut table = Table::new(&other, &plan, &[1.0, 1.0, 1.0], &[8, 8, 8]);
        let error = replay
            .run(&other, &other_schedule, &mut table, consume)
            .expect_err("步骤数不同");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("不是同一张图"), "{error}");
    }

    /// `Store` 的失败原样上抛：重放不能把"算不出来"悄悄当成"算完了"。
    #[test]
    fn a_store_failure_stops_the_replay() {
        struct Refuse;
        impl Store for Refuse {
            fn compute(&mut self, _: &str, _: &[Var], _: Var) -> Result<()> {
                Err(Error::new(Code::Unsupported, "设备拒绝"))
            }
            fn free(&mut self, _: Var) {}
        }

        let (tape, loss) = chain(16);
        let schedule = tape.backward(loss).expect("在图上");
        let bytes = vec![8u64; tape.vars() as usize];
        let plan = Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(96)).expect("可分段");
        let replay = Replay::of(&tape, &schedule, &plan).expect("可执行");
        let error = replay
            .run(&tape, &schedule, &mut Refuse, |_, _| Ok(()))
            .expect_err("重算失败");
        assert_eq!(error.code(), Code::Unsupported);
    }

    /// 随机图：释放时机、重算正确性、峰值与承诺值一起对拍。
    #[test]
    fn random_graphs_replay_without_reading_a_released_value() {
        let mut rng = Rng::new(0x1234_5678);
        let mut replayed_rounds = 0;
        for round in 0..200 {
            let mut tape = Tape::new();
            let mut vars: Vec<Var> = (0..1 + rng.below(3)).map(|_| tape.parameter()).collect();
            vars.extend((0..rng.below(2)).map(|_| tape.leaf()));
            let leaves = vars.len();
            for _ in 0..1 + rng.below(18) {
                let width = 1 + rng.below(2) as usize;
                let inputs: Vec<Var> = (0..width)
                    .map(|_| vars[rng.below(vars.len() as u32) as usize])
                    .collect();
                let op = match width {
                    1 => "relu",
                    _ => "add",
                };
                let recorded = match rng.below(10) {
                    0 => tape.record_with(op, &inputs, Requirement::None),
                    _ => tape.record(op, &inputs),
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
            let seeds: Vec<f32> = (0..leaves).map(|_| 1.0 + rng.below(4) as f32).collect();
            let truth = forward(&tape, &seeds);
            let bytes: Vec<u64> = (0..tape.vars())
                .map(|_| 8 * u64::from(rng.below(8)))
                .collect();
            let budget = Budget::new(8 * u64::from(rng.below(64)));
            let Ok(plan) = Plan::of(&tape, &schedule, bytes.as_slice(), budget) else {
                continue;
            };
            let mut table = Table::new(&tape, &plan, &truth, &bytes);
            let replay = Replay::of(&tape, &schedule, &plan).expect("可执行");
            let computed = replay
                .run(&tape, &schedule, &mut table, consume)
                .unwrap_or_else(|error| panic!("第 {round} 轮：{error}"));
            assert_eq!(table.reads_of_dead, 0, "第 {round} 轮：读到已释放的值");
            assert!(computed <= plan.replayed(), "第 {round} 轮：重算超过承诺");
            assert!(
                table.peak <= plan.peak(),
                "第 {round} 轮：峰值 {} 超过计划 {}",
                table.peak,
                plan.peak()
            );
            assert_eq!(table.live, 0, "第 {round} 轮：执行完还留着激活");
            replayed_rounds += u32::from(computed > 0);
        }
        assert!(
            replayed_rounds > 0,
            "随机图里至少要有一轮真的重算了，否则用例是空的"
        );
    }
}
