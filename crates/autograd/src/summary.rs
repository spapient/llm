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

