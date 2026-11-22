//! 反向执行的**状态**：[`Account`] 数每个变量还差几份梯度贡献，并决定它的反向步骤何时可执行。

use diagnostic::Result;

use crate::graph::{Node, Var};
use crate::schedule::Schedule;
use crate::{internal, invalid};

/// 反向执行的账本：扇出计数与"谁的梯度已经齐全"。
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
        let idx = var.id() as usize;
        match self.state.get(idx) {
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
        match self.pending[idx] {
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
        let idx = input.id() as usize;
        let Some(pending) = self.pending.get_mut(idx) else {
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
#[path = "../tests/account_tests.rs"]
mod tests;
