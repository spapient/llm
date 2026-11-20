//! 反向执行的**状态**：[`Account`] 数每个变量还差几份梯度贡献，并决定它的反向步骤何时可执行。
//!
//! 与 [`crate::Schedule`] 的分工是刻意的：调度是事实（只读、可反复用），账本是运行期状态
//! （每个 micro-batch 都要重建一次）。同一个计划可以配上任意多个账本，二者不共享可变状态。

use diagnostic::Result;

use crate::error::{internal, invalid};
use crate::graph::{Node, Var};
use crate::schedule::Schedule;

/// 反向执行的账本：扇出计数与"谁的梯度已经齐全"。
///
/// 一个变量被多少个**图上**步骤当作输入引用，就要接收多少份贡献；收满之前的步骤不允许执行。
/// 计数一次建好（O(输入数)），之后每一步只做减法，没有哈希、没有扫描。
///
/// 训练循环里复用一个账本：每步 `rebuild` 一次，`pending` 与 `state` 的容量留在原处。
#[derive(Debug, Clone, Default)]
pub struct Account {
    pending: Vec<u32>,
    state: Vec<State>,
    steps: usize,
    settled: usize,
}

/// 账本里每个变量的三态：不进图、等贡献、已执行。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Outside,
    Waiting,
    Done,
}

impl Account {
    /// 空账本：还没有对应任何调度，`finished` 为真，任何 `settle` 都会被拒。
    pub fn new() -> Self {
        Self::default()
    }

    /// 按 `schedule` 重建扇出计数，复用已分配的缓冲。
    ///
    /// 重建会丢弃之前结算的进度——这正是"一个调度、多个账本"的用法：想重跑就再重建一次。
    pub fn rebuild(&mut self, schedule: &Schedule) {
        let graph = schedule.graph();
        self.state.clear();
        self.state
            .extend(graph.iter().map(|on_graph| match on_graph {
                true => State::Waiting,
                false => State::Outside,
            }));
        self.pending.clear();
        self.pending.resize(graph.len(), 0);
        self.steps = schedule.steps().len();
        self.settled = 0;
        for step in schedule.steps() {
            self.count(step);
        }
    }

    /// 还差几份梯度贡献；0 表示齐全。编号不在本次调度里时也返回 0。
    pub fn pending(&self, var: Var) -> u32 {
        self.pending.get(var.id() as usize).copied().unwrap_or(0)
    }

    /// 该变量的梯度是否已经齐全（它的反向步骤现在可以执行）。
    ///
    /// 不进图的变量恒为 `false`：它没有梯度，也就谈不上"齐全"——把常量报成就绪会让调用方
    /// 以为可以执行一步根本不存在的反向。
    pub fn ready(&self, var: Var) -> bool {
        self.graph(var) && self.pending(var) == 0
    }

    /// 已结算的步数。
    pub fn settled(&self) -> usize {
        self.settled
    }

    /// 是否每一步都已结算。
    pub fn finished(&self) -> bool {
        self.settled == self.steps
    }

    /// 结算一步，返回因此**可以执行**的变量（梯度刚收满的前驱，按步骤的输入顺序）。
    ///
    /// 每次调用返回一个新 `Vec`；训练循环里要不分配请用 [`Account::settle_into`]。
    /// 三种误用都会失败而不是算错：步骤不属于这份调度、步骤已结算、梯度还没收满。
    pub fn settle(&mut self, step: &Node) -> Result<Vec<Var>> {
        let mut ready = Vec::new();
        self.settle_into(step, &mut ready)?;
        Ok(ready)
    }

    /// 同 [`Account::settle`]，但把就绪变量写进调用方的 `ready`（先清空），**不分配**。
    pub fn settle_into(&mut self, step: &Node, ready: &mut Vec<Var>) -> Result<()> {
        ready.clear();
        self.enter(step)?;
        self.state[step.output().id() as usize] = State::Done;
        self.settled += 1;
        for input in step.inputs() {
            if let Some(var) = self.contribute(*input)? {
                ready.push(var);
            }
        }
        Ok(())
    }

    /// 统计一步的每个输入位置：一次调用里出现两次就是两份贡献。
    fn count(&mut self, step: &Node) {
        for input in step.inputs() {
            if self.graph(*input) {
                self.pending[input.id() as usize] += 1;
            }
        }
    }

    /// 变量是否在图上（要接收梯度）。
    fn graph(&self, var: Var) -> bool {
        self.state
            .get(var.id() as usize)
            .is_some_and(|state| *state != State::Outside)
    }

    /// 结算前的校验：步骤在计划里、没结算过、梯度已收满。
    fn enter(&self, step: &Node) -> Result<()> {
        let var = step.output();
        let index = var.id() as usize;
        match self.state.get(index) {
            Some(State::Waiting) => {}
            Some(State::Done) => {
                return Err(invalid(
                    "autograd::Account::settle",
                    format!("变量 {} 的步骤已结算", var.id()),
                ));
            }
            _ => {
                return Err(invalid(
                    "autograd::Account::settle",
                    format!("变量 {} 不是本次调度的步骤", var.id()),
                ));
            }
        }
        match self.pending[index] {
            0 => Ok(()),
            pending => Err(invalid(
                "autograd::Account::settle",
                format!("变量 {} 的梯度还差 {pending} 份贡献", var.id()),
            )),
        }
    }

    /// 记一份贡献；不进图的输入不收账，收到最后一份时返回该变量。
    fn contribute(&mut self, input: Var) -> Result<Option<Var>> {
        if !self.graph(input) {
            return Ok(None);
        }
        let index = input.id() as usize;
        let Some(pending) = self.pending.get_mut(index) else {
            return Err(invalid(
                "autograd::Account::settle",
                format!("变量 {} 不在账本里", input.id()),
            ));
        };
        *pending = pending.checked_sub(1).ok_or_else(|| {
            internal(
                "autograd::Account::settle",
                format!("变量 {} 的扇出计数已为 0", input.id()),
            )
        })?;
        Ok((*pending == 0).then_some(input))
    }
}

#[cfg(test)]
mod tests {
    use diagnostic::Code;

    use super::*;
    use crate::graph::Requirement;
    use crate::tape::Tape;

    /// 菱形图：两条路径的贡献都到齐，x 的梯度才算齐全。
    #[test]
    fn a_diamond_waits_for_every_contribution() {
        let (x, left, right, joined, _, schedule) = crate::testutil::diamond();
        let mut account = schedule.account();
        assert_eq!(account.pending(x), 2);
        let steps = schedule.steps();
        assert_eq!(account.settle(&steps[0]).expect("loss"), [joined]);
        assert_eq!(account.settle(&steps[1]).expect("joined"), [left, right]);
        assert!(account.settle(&steps[3]).expect("left").is_empty());
        assert!(!account.ready(x), "还有一条路径没回传");
        assert_eq!(account.settle(&steps[2]).expect("right"), [x]);
        assert!(account.finished());
    }

    /// 同一个变量在一步里出现两次，就要收两份贡献，且只报一次齐全。
    #[test]
    fn a_repeated_input_receives_one_contribution_per_position() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let doubled = tape.record("add", &[x, x]).expect("记录");
        let scaled = tape.record("mul", &[doubled, x]).expect("记录");
        let loss = tape.record("reduce_sum", &[scaled]).expect("记录");
        let schedule = tape.backward(loss).expect("在图上");
        let mut account = schedule.account();
        assert_eq!(account.pending(x), 3, "add 里两次 + mul 里一次");
        let steps = schedule.steps();
        assert_eq!(account.settle(&steps[0]).expect("loss"), [scaled]);
        assert_eq!(account.settle(&steps[1]).expect("scaled"), [doubled]);
        assert!(!account.ready(x), "x 还差两份");
        assert_eq!(
            account.settle(&steps[2]).expect("doubled"),
            [x],
            "两个位置属于同一步，只报一次齐全"
        );
        assert!(account.ready(x));
    }

    /// 乱序执行、重复结算、外来步骤都必须失败，而不是悄悄算错。
    #[test]
    fn settling_out_of_order_or_twice_is_rejected() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let y = tape.record("relu", &[x]).expect("记录");
        let loss = tape.record("reduce_sum", &[y]).expect("记录");
        let schedule = tape.backward(loss).expect("在图上");
        let steps = schedule.steps();
        let mut account = schedule.account();
        assert_eq!(
            account.settle(&steps[1]).expect_err("y 还没收满").code(),
            Code::Invalid
        );
        account.settle(&steps[0]).expect("loss");
        assert_eq!(
            account.settle(&steps[0]).expect_err("已结算").code(),
            Code::Invalid
        );

        let pruned = tape.record("relu", &[x]).expect("记录");
        let foreign = tape.node(pruned).expect("节点").clone();
        assert_eq!(
            account.settle(&foreign).expect_err("不在计划里").code(),
            Code::Invalid
        );
        assert_eq!(account.settled(), 1);
    }

    /// 账本按计划重建：结算进度清零，扇出重新数一遍。
    #[test]
    fn rebuilding_starts_the_ledger_over() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let y = tape.record("relu", &[x]).expect("记录");
        let loss = tape.record("reduce_sum", &[y]).expect("记录");
        let schedule = tape.backward(loss).expect("在图上");
        let mut account = Account::new();
        account.rebuild(&schedule);
        assert_eq!(account.settled(), 0);
        assert_eq!(account.pending(y), 1);
        let step = &schedule.steps()[0];
        account.settle(step).expect("loss");
        assert_eq!(account.settled(), 1);
        account.rebuild(&schedule);
        assert_eq!(account.settled(), 0, "重建丢弃进度");
        assert_eq!(account.pending(y), 1, "扇出回到初始值");
        assert!(!account.finished());
    }

    /// `settle_into` 与 `settle` 结果一致，复用调用方的缓冲。
    #[test]
    fn settling_into_a_reused_buffer_reports_the_same_ready_vars() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let y = tape.parameter();
        let joined = tape.record("add", &[x, y]).expect("记录");
        let loss = tape.record("reduce_sum", &[joined]).expect("记录");
        let schedule = tape.backward(loss).expect("在图上");
        let mut account = schedule.account();
        let mut ready = Vec::with_capacity(8);
        let capacity = ready.capacity();
        account
            .settle_into(&schedule.steps()[0], &mut ready)
            .expect("loss");
        assert_eq!(ready, [joined]);
        account
            .settle_into(&schedule.steps()[1], &mut ready)
            .expect("joined");
        assert_eq!(ready, [x, y], "先清空再写，不残留上一步的就绪变量");
        assert_eq!(ready.capacity(), capacity, "缓冲没有重新分配");
        assert!(account.finished());
    }

    /// 一个账本只管一份调度：空账本拒绝一切，`finished` 也只是平凡真。
    #[test]
    fn an_empty_account_accepts_nothing() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let recorded = tape.record("relu", &[x]).expect("记录");
        let node = tape.node(recorded).expect("节点");
        let mut account = Account::new();
        assert!(account.finished());
        assert_eq!(
            account.settle(node).expect_err("没有调度").code(),
            Code::Invalid
        );
        assert!(
            account
                .settle(node)
                .expect_err("没有调度")
                .to_string()
                .contains("不是本次调度")
        );
    }

    /// 每个输入都有扇出，图上变量的计数与独立统计一致；不进图的变量不收账。
    #[test]
    fn fan_out_matches_an_independent_count() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let constant = tape.leaf();
        let a = tape.record("add", &[x, constant]).expect("记录");
        let b = tape.record("mul", &[a, a]).expect("记录");
        let loss = tape.record("reduce_sum", &[b, constant]).expect("记录");
        let schedule = tape.backward(loss).expect("在图上");
        let account = schedule.account();
        assert_eq!(account.pending(loss), 0, "loss 自己不是任何步骤的输入");
        assert_eq!(account.pending(b), 1);
        assert_eq!(account.pending(a), 2, "mul 里出现两次");
        assert_eq!(account.pending(x), 1);
        assert_eq!(account.pending(constant), 0, "常量不进图");
        assert_eq!(
            tape.requirement(constant),
            Some(Requirement::None),
            "常量确实在图外"
        );
        assert!(!account.ready(constant), "图外变量永远不会'就绪'");
    }
}
