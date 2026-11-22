//! 反向图的记录：[`Tape`] 按前向顺序追加节点，并可把 loss 的计划写进一个可复用的 [`Schedule`]。

use std::sync::Arc;

use diagnostic::Result;

use crate::graph::{Node, Requirement, Role, Var};
use crate::invalid;
use crate::plan::Config;
use crate::schedule::Schedule;

/// 反向图：按插入顺序记录节点，逆序就是反向顺序；四个平行数组共用一套稠密编号（编号即下标）。
#[derive(Debug, Clone, Default)]
pub struct Tape {
    nodes: Vec<Node>,
    requirement: Vec<Requirement>,
    depth: Vec<u32>,
    producer: Vec<Option<u32>>,
    /// 每个变量是否被任何节点当作输入引用过；`retain` 的 O(1) 判定。
    referenced: Vec<bool>,
    next: u32,
}

impl Tape {
    /// 空图。
    pub fn new() -> Self {
        Self::default()
    }

    /// 清空整张图但保留已分配的内存；编号从 0 重新开始，之前拿到的 [`Var`] 一律作废。
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.requirement.clear();
        self.depth.clear();
        self.producer.clear();
        self.referenced.clear();
        self.next = 0;
    }

    /// 新建一个不参与反向的叶子（输入 token、常量、权重之外的数据）。
    pub fn leaf(&mut self) -> Var {
        self.fresh(Requirement::None, 0)
    }

    /// 新建一个梯度终点叶子（可训练参数）。
    pub fn parameter(&mut self) -> Var {
        self.fresh(Requirement::Grad, 0)
    }

    /// 把已存在的叶子升级为梯度终点；只允许升级**还没被任何节点引用**的叶子（否则会漏掉梯度）。
    pub fn retain(&mut self, var: Var) -> Result<()> {
        let idx = self.check(var, "autograd::Tape::retain")?;
        if self.producer[idx].is_some() {
            return Err(invalid(
                "autograd::Tape::retain",
                format!(
                    "变量 {} 是算子输出，需求在记录时已确定；要终点请用 record_with",
                    var.id()
                ),
            ));
        }
        if self.requirement[idx].is_endpoint() {
            return Ok(());
        }
        if self.consumed(idx) {
            return Err(invalid(
                "autograd::Tape::retain",
                format!("变量 {} 已被记录为常量输入，升级为终点会漏掉梯度", var.id()),
            ));
        }
        self.requirement[idx] = Requirement::Grad;
        Ok(())
    }

    /// 记录一次算子调用，需求由输入推出：有输入要梯度就是反向步，全不要就不进图。
    pub fn record(&mut self, op: impl Into<String>, inputs: &[Var]) -> Result<Var> {
        let (requirement, deepest) = self.survey(inputs, "autograd::Tape::record")?;
        let op = op.into();
        self.named(&op)?;
        Ok(self.push(op, inputs, requirement, deepest))
    }

    /// 记录一次算子调用并显式声明输出的需求；`None` 即 detach，梯度到此为止。
    pub fn record_with(
        &mut self,
        op: impl Into<String>,
        inputs: &[Var],
        requirement: Requirement,
    ) -> Result<Var> {
        let op = op.into();
        let (_, deepest) = self.checked(&op, inputs, requirement)?;
        Ok(self.push(op, inputs, requirement, deepest))
    }

    /// 记录的公共校验：算子名非空、输入存在、声明的需求与输入相容。
    fn checked(
        &self,
        op: &str,
        inputs: &[Var],
        requirement: Requirement,
    ) -> Result<(Requirement, u32)> {
        self.named(op)?;
        let (inferred, deepest) = self.survey(inputs, "autograd::Tape::record_with")?;
        if requirement.requires() && !inferred.requires() {
            return Err(invalid(
                "autograd::Tape::record_with",
                format!(
                    "算子 {op} 声明为 {}，但没有任何输入需要梯度",
                    requirement.name()
                ),
            ));
        }
        Ok((inferred, deepest))
    }

    /// 算子名非空；空名会让诊断与调度都失去可读性。
    fn named(&self, op: &str) -> Result<()> {
        if op.trim().is_empty() {
            return Err(invalid("autograd::Tape::record_with", "算子名不能为空"));
        }
        Ok(())
    }

    /// 全部节点，按前向顺序。
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// 变量需求；编号不存在时为 `None`。
    pub fn requirement(&self, var: Var) -> Option<Requirement> {
        self.requirement.get(var.id() as usize).copied()
    }

    /// 变量出身（参数 / 常量 / 激活）；编号不存在时为 `None`。
    pub fn role(&self, var: Var) -> Option<Role> {
        let requirement = self.requirement(var)?;
        Some(Role::of(requirement, self.node(var).is_none()))
    }

    /// 整图审计：逐个节点复核输入存在、算子名非空、声明的需求与输入相容。
    pub fn validate(&self) -> Result<()> {
        for node in self.nodes() {
            self.checked(node.op(), node.inputs(), node.requirement())
                .map(|_| ())
                .map_err(|error| {
                    error.with(format!(
                        "节点 {}（算子 `{}`）",
                        node.output().id(),
                        node.op()
                    ))
                })?;
        }
        Ok(())
    }

    /// 变量深度；编号不存在时为 `None`。
    pub fn depth(&self, var: Var) -> Option<u32> {
        self.depth.get(var.id() as usize).copied()
    }

    /// 产出该变量的节点；叶子与不存在的编号都是 `None`。
    pub fn node(&self, var: Var) -> Option<&Node> {
        let idx = (*self.producer.get(var.id() as usize)?)?;
        self.nodes.get(idx as usize)
    }

    /// 变量总数（叶子加算子输出），也是下一个变量的编号。
    pub fn vars(&self) -> u32 {
        self.next
    }

    /// 由编号取回变量；编号不存在时为 `None`（缓冲按编号稠密布局，拿来回路转用）。
    pub fn var(&self, id: u32) -> Option<Var> {
        (id < self.next).then(|| Var::at(id))
    }

    /// 节点数（不含叶子）。
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// 是否没有任何算子调用。
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// 反向顺序：**全部**节点的逆序，不做可达性剪枝（执行反向请用 [`Tape::backward`]）。
    pub fn backward_order(&self) -> Vec<Var> {
        self.nodes.iter().rev().map(Node::output).collect()
    }

    /// 为 `loss` 生成反向调度：只含可达节点，按深度降序排列。
    pub fn backward(&self, loss: Var) -> Result<Schedule> {
        let mut schedule = Schedule::default();
        self.plan_into(&mut schedule, loss)?;
        Ok(schedule)
    }

    /// 把 `loss` 的反向调度写进已有的 `schedule`，**复用它的缓冲**（失败时它保持原样）。
    pub fn plan_into(&self, schedule: &mut Schedule, loss: Var) -> Result<()> {
        schedule.plan(self, loss)
    }

    /// 按 [`Config`] 里的排序策略生成调度，同样复用 `schedule` 的缓冲。
    pub fn plan_config(&self, schedule: &mut Schedule, loss: Var, config: &Config) -> Result<()> {
        schedule.plan_config(self, loss, config)
    }

    /// 分配一个新编号，需求与深度写进平行数组。
    fn fresh(&mut self, requirement: Requirement, depth: u32) -> Var {
        let var = Var::at(self.next);
        self.next += 1;
        self.requirement.push(requirement);
        self.depth.push(depth);
        self.producer.push(None);
        self.referenced.push(false);
        var
    }

    /// 分配编号并追加节点；输入边在这里一次标好，后续 `retain` 不再扫全图。
    fn push(&mut self, op: String, inputs: &[Var], requirement: Requirement, deepest: u32) -> Var {
        let depth = match requirement.requires() {
            true => deepest.saturating_add(1),
            false => 0,
        };
        let output = self.fresh(requirement, depth);
        let idx = self.nodes.len() as u32;
        self.producer[output.id() as usize] = Some(idx);
        for input in inputs {
            self.referenced[input.id() as usize] = true;
        }
        self.nodes.push(Node::new(
            output,
            Arc::from(op),
            Arc::from(inputs),
            depth,
            requirement,
        ));
        output
    }

    /// 校验输入变量并推出需求与最大深度；不存在的编号在这里失败，避免反向时"梯度莫名丢失"。
    fn survey(&self, inputs: &[Var], site: &'static str) -> Result<(Requirement, u32)> {
        let mut merged = Requirement::None;
        let mut deepest = 0;
        for input in inputs {
            let idx = self.check(*input, site)?;
            merged = merged.merged(self.requirement[idx]);
            deepest = deepest.max(self.depth[idx]);
        }
        Ok((merged, deepest))
    }

    /// 该变量是否被任何节点当作输入引用过。
    fn consumed(&self, idx: usize) -> bool {
        self.referenced.get(idx).copied().unwrap_or(false)
    }

    /// 编号校验：越界即"引用了一个还没创建的变量"。
    fn check(&self, var: Var, site: &'static str) -> Result<usize> {
        if var.id() as usize >= self.requirement.len() {
            return Err(invalid(
                site,
                format!(
                    "变量 {} 不存在（当前已创建 {} 个）",
                    var.id(),
                    self.requirement.len()
                ),
            ));
        }
        Ok(var.id() as usize)
    }
}

#[cfg(test)]
#[path = "../tests/tape_tests.rs"]
mod tests;
