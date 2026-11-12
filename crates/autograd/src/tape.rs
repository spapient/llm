//! 反向图的记录：[`Tape`] 按前向顺序追加节点，并可把 loss 的计划写进一个可复用的 [`Schedule`]。
//!
//! 编号是稠密的 `u32`，所以"需求 / 深度 / 生产者"都是平行数组上的一次下标，没有哈希；
//! 新增一个节点是均摊 O(输入数)，已有节点一律不动。

use std::sync::Arc;

use diagnostic::Result;

use crate::error::invalid;
use crate::graph::{Node, Requirement, Var};
use crate::schedule::Schedule;

/// 反向图：按插入顺序记录节点，逆序就是反向顺序。
///
/// 四个平行数组是刻意的：编号就是下标，查需求、查深度、查生产者都不需要哈希表；
/// 叶子与算子输出共用一套编号，于是"输入必须已存在"只是一次越界检查。
#[derive(Debug, Clone, Default)]
pub struct Tape {
    nodes: Vec<Node>,
    requirement: Vec<Requirement>,
    depth: Vec<u32>,
    producer: Vec<Option<u32>>,
    next: u32,
}

impl Tape {
    /// 空图。
    pub fn new() -> Self {
        Self::default()
    }

    /// 清空整张图但保留已分配的内存。
    ///
    /// 训练循环每步复用一个 `Tape` 时用它，稳定态不会再有分配。清空后编号从 0 重新开始，
    /// 之前拿到的 [`Var`] 一律作废——它们会与新变量撞号，那种错不会报错，只会在反向时算错。
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.requirement.clear();
        self.depth.clear();
        self.producer.clear();
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

    /// 把已存在的叶子升级为梯度终点。
    ///
    /// 只允许升级**还没被任何节点引用**的叶子：父节点的需求在记录时就定了，事后再改会让
    /// 一个已经按"常量"剪掉的输入漏掉梯度，这种图看起来能跑但结果是错的，所以直接报错。
    /// 已经是终点时幂等返回。
    pub fn retain(&mut self, var: Var) -> Result<()> {
        let index = self.check(var, "autograd::Tape::retain")?;
        if self.producer[index].is_some() {
            return Err(invalid(
                "autograd::Tape::retain",
                format!(
                    "变量 {} 是算子输出，需求在记录时已确定；要终点请用 record_with",
                    var.id()
                ),
            ));
        }
        if self.requirement[index].is_endpoint() {
            return Ok(());
        }
        if self.consumed(index) {
            return Err(invalid(
                "autograd::Tape::retain",
                format!("变量 {} 已被记录为常量输入，升级为终点会漏掉梯度", var.id()),
            ));
        }
        self.requirement[index] = Requirement::Grad;
        Ok(())
    }

    /// 记录一次算子调用，需求由输入推出：有输入要梯度就是 [`Requirement::Backward`]，
    /// 全都不需要则是 [`Requirement::None`]（算子照常执行，只是不进反向图）。
    pub fn record(&mut self, op: impl Into<String>, inputs: &[Var]) -> Result<Var> {
        let requirement = self.infer(inputs, "autograd::Tape::record")?;
        self.record_with(op, inputs, requirement)
    }

    /// 记录一次算子调用并显式声明输出的需求。
    ///
    /// `None` 表示断开（detach）：输出与输入之间没有反向边，梯度到此为止。
    /// `Backward` / `Grad` 要求至少有一个输入要梯度——没有那条边就没有梯度可传，
    /// 声明成终点会得到一个"永远收不到贡献"的变量。
    pub fn record_with(
        &mut self,
        op: impl Into<String>,
        inputs: &[Var],
        requirement: Requirement,
    ) -> Result<Var> {
        let op = op.into();
        if op.trim().is_empty() {
            return Err(invalid("autograd::Tape::record_with", "算子名不能为空"));
        }
        let inferred = self.infer(inputs, "autograd::Tape::record_with")?;
        if requirement.requires() && !inferred.requires() {
            return Err(invalid(
                "autograd::Tape::record_with",
                format!(
                    "算子 {op} 声明为 {}，但没有任何输入需要梯度",
                    requirement.name()
                ),
            ));
        }
        let depth = match requirement.requires() {
            true => self.deepest(inputs) + 1,
            false => 0,
        };
        let output = self.fresh(requirement, depth);
        let index = self.nodes.len() as u32;
        self.producer[output.id() as usize] = Some(index);
        self.nodes.push(Node::new(
            output,
            Arc::from(op),
            Arc::from(inputs),
            depth,
            requirement,
        ));
        Ok(output)
    }

    /// 全部节点，按前向顺序。
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// 变量需求；编号不存在时为 `None`。
    pub fn requirement(&self, var: Var) -> Option<Requirement> {
        self.requirement.get(var.id() as usize).copied()
    }

    /// 变量深度；编号不存在时为 `None`。
    pub fn depth(&self, var: Var) -> Option<u32> {
        self.depth.get(var.id() as usize).copied()
    }

    /// 产出该变量的节点；叶子与不存在的编号都是 `None`。
    pub fn node(&self, var: Var) -> Option<&Node> {
        let index = (*self.producer.get(var.id() as usize)?)?;
        self.nodes.get(index as usize)
    }

    /// 变量总数（叶子加算子输出），也是下一个变量的编号。
    pub fn vars(&self) -> u32 {
        self.next
    }

    /// 由编号取回变量；编号不存在时为 `None`。
    ///
    /// 运行时缓冲按编号稠密布局，要在"编号"与"变量"之间来回换需要这个入口：
    /// 直接构造一个没校验过的 `Var` 会让打错的编号变成一张看起来正常的图。
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

    /// 反向顺序：**全部**节点的逆序，不做可达性剪枝。
    ///
    /// 要执行反向请用 [`Tape::backward`] / [`Tape::plan_into`]：它们剪掉与 loss 无关的分支。
    pub fn backward_order(&self) -> Vec<Var> {
        self.nodes.iter().rev().map(Node::output).collect()
    }

    /// 为 `loss` 生成反向调度：只含可达节点，按深度降序排列。
    ///
    /// 每次调用都会新建一个 [`Schedule`]；训练循环里要复用缓冲请用 [`Tape::plan_into`]。
    pub fn backward(&self, loss: Var) -> Result<Schedule> {
        let mut schedule = Schedule::default();
        self.plan_into(&mut schedule, loss)?;
        Ok(schedule)
    }

    /// 把 `loss` 的反向调度写进已有的 `schedule`，**复用它的缓冲**。
    ///
    /// 稳定态不分配：步骤表、可达位图、终点表都是 `clear` 后重填，容量留在原处。
    /// 失败时 `schedule` 保持原样——调用方拿到 `Err` 时不会得到半份计划。
    pub fn plan_into(&self, schedule: &mut Schedule, loss: Var) -> Result<()> {
        schedule.plan(self, loss)
    }

    /// 分配一个新编号，需求与深度写进平行数组。
    fn fresh(&mut self, requirement: Requirement, depth: u32) -> Var {
        let var = Var::at(self.next);
        self.next += 1;
        self.requirement.push(requirement);
        self.depth.push(depth);
        self.producer.push(None);
        var
    }

    /// 输入里的最大深度；没有输入时返回 0（叶子之上的一层）。
    fn deepest(&self, inputs: &[Var]) -> u32 {
        inputs
            .iter()
            .map(|input| self.depth[input.id() as usize])
            .max()
            .unwrap_or(0)
    }

    /// 校验输入变量并推出需求；不存在的编号在这里失败，避免反向时"梯度莫名丢失"。
    fn infer(&self, inputs: &[Var], site: &'static str) -> Result<Requirement> {
        let mut merged = Requirement::None;
        for input in inputs {
            let index = self.check(*input, site)?;
            merged = merged.merged(self.requirement[index]);
        }
        Ok(merged)
    }

    /// 该变量是否被任何节点当作输入引用过。
    fn consumed(&self, index: usize) -> bool {
        self.nodes.iter().any(|node| {
            node.inputs()
                .iter()
                .any(|input| input.id() as usize == index)
        })
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
mod tests {
    use diagnostic::Code;

    use super::*;

    #[test]
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

    #[test]
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

    #[test]
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

    #[test]
    fn referencing_a_missing_variable_is_rejected() {
        let mut tape = Tape::new();
        let x = tape.leaf();
        let ghost = Var::at(42);
        let error = tape.record("add", &[x, ghost]).expect_err("不存在的输入");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("42"), "{error}");
    }

    #[test]
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

    #[test]
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

    #[test]
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

    #[test]
    fn an_empty_operator_name_is_rejected() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        assert_eq!(
            tape.record("  ", &[x]).expect_err("空算子名").code(),
            Code::Invalid
        );
        assert!(tape.is_empty());
    }

    #[test]
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

    #[test]
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

    #[test]
    fn a_parameter_recorded_as_loss_has_an_empty_schedule() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let schedule = tape.backward(x).expect("参数在图上");
        assert!(schedule.is_empty());
        assert_eq!(schedule.endpoints(), [x]);
        assert_eq!(schedule.loss(), x);
    }

    #[test]
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

    #[test]
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
