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
        let idx = self.check_var(var, "autograd::Tape::retain")?;
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
        self.named(&op, "autograd::Tape::record")?;
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
        let (_, deepest) =
            self.check_inputs(&op, inputs, requirement, "autograd::Tape::record_with")?;
        Ok(self.push(op, inputs, requirement, deepest))
    }

    /// 记录的公共校验：算子名非空、输入存在、声明的需求与输入相容。`site` 标到实际调用口。
    fn check_inputs(
        &self,
        op: &str,
        inputs: &[Var],
        requirement: Requirement,
        site: &'static str,
    ) -> Result<(Requirement, u32)> {
        self.named(op, site)?;
        let (inferred, deepest) = self.survey(inputs, site)?;
        if requirement.requires() && !inferred.requires() {
            return Err(invalid(
                site,
                format!(
                    "算子 {op} 声明为 {}，但没有任何输入需要梯度",
                    requirement.name()
                ),
            ));
        }
        Ok((inferred, deepest))
    }

    /// 算子名非空；空名会让诊断与调度都失去可读性。`site` 标到实际调用口。
    fn named(&self, op: &str, site: &'static str) -> Result<()> {
        if op.trim().is_empty() {
            return Err(invalid(site, "算子名不能为空"));
        }
        Ok(())
    }

    /// 全部节点，按前向顺序。
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// 变量需求；编号不存在时为 `None`。
    #[must_use]
    pub fn requirement(&self, var: Var) -> Option<Requirement> {
        self.requirement.get(var.id() as usize).copied()
    }

    /// 变量出身（参数 / 常量 / 激活）；编号不存在时为 `None`。
    #[must_use]
    pub fn role(&self, var: Var) -> Option<Role> {
        let requirement = self.requirement(var)?;
        Some(Role::of(requirement, self.node(var).is_none()))
    }

    /// 整图审计：逐个节点复核输入存在、算子名非空、声明的需求与输入相容。
    pub fn validate(&self) -> Result<()> {
        for node in self.nodes() {
            self.check_inputs(
                node.op(),
                node.inputs(),
                node.requirement(),
                "autograd::Tape::validate",
            )
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
    #[must_use]
    pub fn depth(&self, var: Var) -> Option<u32> {
        self.depth.get(var.id() as usize).copied()
    }

    /// 产出该变量的节点；叶子与不存在的编号都是 `None`。
    #[must_use]
    pub fn node(&self, var: Var) -> Option<&Node> {
        let idx = (*self.producer.get(var.id() as usize)?)?;
        self.nodes.get(idx as usize)
    }

    /// 变量总数（叶子加算子输出），也是下一个变量的编号。
    #[must_use]
    pub fn vars(&self) -> u32 {
        self.next
    }

    /// 由编号取回变量；编号不存在时为 `None`（缓冲按编号稠密布局，拿来回路转用）。
    #[must_use]
    pub fn var(&self, id: u32) -> Option<Var> {
        (id < self.next).then(|| Var::at(id))
    }

    /// 节点数（不含叶子）。
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// 是否没有任何算子调用。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// 反向顺序：**全部**节点的逆序，不做可达性剪枝（执行反向请用 [`Tape::backward`]）。
    #[must_use]
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
            let idx = self.check_var(*input, site)?;
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
    fn check_var(&self, var: Var, site: &'static str) -> Result<usize> {
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
mod tests {
    use diagnostic::Code;

    use crate::{MemoryPolicy, Ordering};

    use super::*;

    testkit::test_cases! {
    fn ids_are_allocated_monotonically() {
        let mut tape = Tape::new();
        let a = tape.leaf();
        let b = tape.parameter();
        assert_eq!((a.id(), b.id()), (0, 1));
        assert_eq!(tape.vars(), 2);
        assert!(tape.is_empty(), "叶子不是节点");
        assert_eq!(tape.requirement(a), Some(Requirement::None));
        assert_eq!(tape.requirement(b), Some(Requirement::Grad));
    }

    fn clearing_resets_the_numbering() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        tape.record("relu", &[x]).expect("记录");
        tape.clear();
        assert!(tape.is_empty());
        assert_eq!(tape.vars(), 0);
        let fresh = tape.parameter();
        assert_eq!(fresh.id(), 0, "编号从 0 重新开始");
        assert_eq!(tape.requirement(fresh), Some(Requirement::Grad));
    }

    fn backward_order_is_the_reverse_of_forward() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let w = tape.parameter();
        let y = tape.record("gemv_i4", &[x, w]).expect("记录");
        let loss = tape.record("reduce_sum", &[y]).expect("记录");
        assert_eq!(tape.len(), 2);
        assert_eq!(tape.backward_order(), [loss, y]);
        assert_eq!(tape.nodes()[1].inputs(), [y]);
        assert_eq!(tape.nodes()[0].op(), "gemv_i4");
        assert_eq!(tape.depth(y), Some(1));
        assert_eq!(tape.depth(loss), Some(2));
    }

    fn referencing_a_missing_variable_is_rejected() {
        let mut tape = Tape::new();
        let x = tape.leaf();
        let ghost = Var::at(42);
        let error = tape.record("add", &[x, ghost]).expect_err("不存在的输入");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("42"), "{error}");
    }

    fn the_requirement_is_inferred_from_the_inputs() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let constant = tape.leaf();
        let from_constant = tape.record("scale", &[constant]).expect("记录");
        let from_parameter = tape.record("relu", &[x]).expect("记录");
        assert_eq!(tape.requirement(from_constant), Some(Requirement::None));
        assert_eq!(
            tape.requirement(from_parameter),
            Some(Requirement::Backward)
        );
        assert_eq!(tape.depth(from_constant), Some(0), "不进图的节点深度是 0");
    }

    fn detaching_cuts_the_path_to_the_inputs() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let frozen = tape
            .record_with("stop_gradient", &[x], Requirement::None)
            .expect("记录");
        let downstream = tape.record("relu", &[frozen]).expect("记录");
        assert_eq!(tape.requirement(downstream), Some(Requirement::None));
        assert_eq!(
            tape.backward(downstream).expect_err("不在图上").code(),
            Code::Invalid
        );
        assert!(tape.backward(x).expect("参数的梯度是 1").is_empty());
    }

    fn a_requiring_node_without_requiring_inputs_is_rejected() {
        let mut tape = Tape::new();
        let constant = tape.leaf();
        let error = tape
            .record_with("relu", &[constant], Requirement::Grad)
            .expect_err("没有输入要梯度");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("Grad"), "{error}");
        assert!(tape.is_empty(), "失败的记录不留节点");
    }

    fn an_empty_operator_name_is_rejected() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        assert_eq!(
            tape.record("  ", &[x]).expect_err("空算子名").code(),
            Code::Invalid
        );
        assert!(tape.is_empty());
    }

    fn plan_config_follows_the_configured_ordering() {
        let (tape, _, _, z) = crate::testutil::pair();
        let config = Config::new(Ordering::Insertion, MemoryPolicy::Auto);
        let mut schedule = Schedule::default();
        tape.plan_config(&mut schedule, z, &config).expect("规划");
        assert_eq!(schedule.ordering(), Ordering::Insertion);
    }

    fn roles_separate_parameters_constants_and_activations() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let c = tape.leaf();
        let y = tape.record("relu", &[x]).expect("记录");
        assert_eq!(tape.role(x), Some(Role::Parameter));
        assert_eq!(tape.role(c), Some(Role::Constant));
        assert_eq!(tape.role(y), Some(Role::Activation));
        assert!(tape.validate().is_ok());
    }

    fn retaining_an_unused_leaf_makes_it_an_endpoint() {
        let mut tape = Tape::new();
        let x = tape.leaf();
        tape.retain(x).expect("未被引用");
        assert_eq!(tape.requirement(x), Some(Requirement::Grad));
        tape.retain(x).expect("幂等");
        let y = tape.record("relu", &[x]).expect("记录");
        assert_eq!(
            tape.backward(y).expect("在图上").endpoints(),
            [x],
            "升级后的叶子是梯度终点"
        );
    }

    fn retaining_is_rejected_once_the_leaf_is_used_or_computed() {
        let mut tape = Tape::new();
        let x = tape.leaf();
        let y = tape.record("relu", &[x]).expect("记录");
        assert_eq!(tape.retain(x).expect_err("已被引用").code(), Code::Invalid);
        assert_eq!(tape.retain(y).expect_err("算子输出").code(), Code::Invalid);
        assert_eq!(
            tape.retain(Var::at(7)).expect_err("不存在").code(),
            Code::Invalid
        );
    }

    fn a_parameter_recorded_as_loss_has_an_empty_schedule() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let schedule = tape.backward(x).expect("参数在图上");
        assert!(schedule.is_empty());
        assert_eq!(schedule.endpoints(), [x]);
        assert_eq!(schedule.loss(), x);
    }

    fn backward_rejects_variables_that_are_not_on_the_graph() {
        let mut tape = Tape::new();
        let constant = tape.leaf();
        let computed = tape.record("scale", &[constant]).expect("记录");
        let ghost = Var::at(9);
        assert_eq!(
            tape.backward(constant).expect_err("不进图").code(),
            Code::Invalid
        );
        assert_eq!(
            tape.backward(computed).expect_err("不进图").code(),
            Code::Invalid
        );
        assert_eq!(
            tape.backward(ghost).expect_err("不存在").code(),
            Code::Invalid
        );
    }

    fn a_retained_intermediate_is_an_endpoint_with_a_step() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let hidden = tape
            .record_with("relu", &[x], Requirement::Grad)
            .expect("记录");
        let loss = tape.record("reduce_sum", &[hidden]).expect("记录");
        let schedule = tape.backward(loss).expect("在图上");
        assert_eq!(schedule.endpoints(), [x, hidden]);
        assert_eq!(schedule.len(), 2, "终点也要执行它自己的反向步骤");
    }
    }
}
