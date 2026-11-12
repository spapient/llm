//! 反向图：记录（[`Tape`]、[`Var`]、[`Node`]）与调度（[`Schedule`]、[`Account`]）。
//!
//! **拥有**：`Requirement`、`Var`、`Node`、`Tape`、`Schedule`、`Account`。**不拥有**：梯度数值与前向
//! 激活（运行时缓冲）、具体算子的反向公式（P1：算子反向表会引用算子名，但记录结构不会变）、
//! 前向计算（`nn` 与 `cutile` 生成）。**依赖**：`tensor`、`diagnostic`。**被依赖**：`train`。
//!
//! 这一层只回答三个问题，且答案都是**数据**，不含数值：
//!
//! 1. **谁在图上**：只有 [`Requirement`] 不是 [`Requirement::None`] 的变量参与反向。`record` 从输入
//!    推出需求，`record_with(.., None)` 是显式断开（detach），`retain` 把已存在的叶子升级为梯度终点。
//! 2. **按什么顺序执行**：[`Tape::backward`] 只排 loss 可达的节点，顺序是"深度降序、同深度编号降序"。
//!    它同时是合法的反向拓扑序（输出总比输入深）与并行提示（同深度互不依赖，可合批或并发）。
//! 3. **梯度什么时候齐**：每个变量的扇出（在可达子图里被引用的次数）记在 [`Account`] 里，调用方
//!    不必自己数——`settle` 收满贡献的变量才允许执行它的反向步骤，顺序错了直接报错。
//!
//! # 例子
//!
//! ```
//! use llm_autograd::Tape;
//!
//! let mut tape = Tape::new();
//! let x = tape.parameter(); // 要算梯度的叶子
//! let w = tape.leaf(); // 常量输入，不进图
//! let y = tape.record("mul", &[x, w]).expect("输入已存在");
//! let loss = tape.record("reduce_sum", &[y]).expect("输入已存在");
//!
//! let schedule = tape.backward(loss).expect("loss 在图上");
//! assert_eq!(schedule.len(), 2, "只有 loss 与 y 要执行反向");
//! assert_eq!(schedule.endpoints(), [x], "梯度终点只有参数 x");
//!
//! let mut account = schedule.account();
//! let ready = account.settle(&schedule.steps()[0]).expect("loss 的梯度是 1");
//! assert_eq!(ready, [y], "y 只被 loss 消费一次，这一步之后它的梯度就齐了");
//! account.settle(&schedule.steps()[1]).expect("y 已就绪");
//! assert!(account.finished());
//! ```

#![deny(unsafe_code)]

mod sched;

use std::sync::Arc;

use diagnostic::{Code, Error, Result};

pub use sched::{Account, Schedule};

/// 一个变量对反向传播的要求。
///
/// `None` 与另外两种是**图内/图外**的分界：不进图的变量既不产生反向步骤，也不接收梯度，
/// 因此它的"梯度为 0"不是算出来的结果，而是根本没有这个量。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Requirement {
    /// 不参与反向：常量、输入 token、detach 之后的输出。
    #[default]
    None,
    /// 只为反传存在：它的梯度要传给输入，但它本身不是梯度终点。
    Backward,
    /// 梯度终点：反向结束后调用方要取走它的梯度（参数、显式 retain 的变量）。
    Grad,
}

impl Requirement {
    /// 是否在图上（参与反向）。
    pub fn requires(self) -> bool {
        !self.is_none()
    }

    /// 是否完全在反向之外。
    pub fn is_none(self) -> bool {
        matches!(self, Self::None)
    }

    /// 诊断里的短名，与变体一一对应。
    fn name(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Backward => "Backward",
            Self::Grad => "Grad",
        }
    }

    /// 是否是梯度终点。
    pub fn is_endpoint(self) -> bool {
        matches!(self, Self::Grad)
    }

    /// 合并两个需求：都不需要梯度才是 `None`，否则输出只能是中间节点。
    ///
    /// 用"输出"而不是"两边的较大者"：两个梯度终点的和仍然不是梯度终点——终点由调用方
    /// 显式声明（`record_with(.., Requirement::Grad)`），不从输入继承。
    pub fn merged(self, other: Self) -> Self {
        match self.is_none() && other.is_none() {
            true => Self::None,
            false => Self::Backward,
        }
    }
}

/// 一个可微变量（只存编号，数值在运行时缓冲里）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var {
    id: u32,
}

impl Var {
    /// 编号。编号按创建顺序单调递增，因此"编号大"就是"更晚创建"。
    pub fn id(&self) -> u32 {
        self.id
    }
}

/// 反向图的一个节点：一次算子调用产生的变量。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    output: Var,
    op: Arc<str>,
    inputs: Arc<[Var]>,
    depth: u32,
    requirement: Requirement,
}

impl Node {
    /// 产出的变量。
    pub fn output(&self) -> Var {
        self.output
    }

    /// 算子名（与 `op::Op::name` 一致，便于把反向公式与声明对上）。
    pub fn op(&self) -> &str {
        &self.op
    }

    /// 输入变量。
    pub fn inputs(&self) -> &[Var] {
        &self.inputs
    }

    /// 深度：输入的最大深度加一，叶子为 0。反向顺序按它降序排。
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// 产出变量的需求。
    pub fn requirement(&self) -> Requirement {
        self.requirement
    }
}

/// 反向图：按插入顺序记录节点，逆序就是反向顺序。
///
/// 三个按变量编号索引的平行数组是刻意的：编号是稠密的 `u32`，查需求、查深度、查生产者都是
/// 一次下标，没有任何哈希。新增一个节点是均摊 O(输入数)，不需要重排已有节点。
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
            return Err(self.reject(format!(
                "变量 {} 是算子输出，需求在记录时已确定；要终点请用 record_with",
                var.id()
            )));
        }
        if self.requirement[index].is_endpoint() {
            return Ok(());
        }
        if self.consumed(index) {
            return Err(self.reject(format!(
                "变量 {} 已被记录为常量输入，升级为终点会漏掉梯度",
                var.id()
            )));
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
            return Err(self.reject("算子名不能为空".to_string()));
        }
        let inferred = self.infer(inputs, "autograd::Tape::record_with")?;
        if requirement.requires() && !inferred.requires() {
            return Err(self.reject(format!(
                "算子 {op} 声明为 {}，但没有任何输入需要梯度",
                requirement.name()
            )));
        }
        let depth = if requirement.requires() {
            inputs
                .iter()
                .map(|input| self.depth[input.id() as usize])
                .max()
                .unwrap_or(0)
                + 1
        } else {
            0
        };
        let output = self.fresh(requirement, depth);
        let index = self.nodes.len() as u32;
        self.producer[output.id() as usize] = Some(index);
        self.nodes.push(Node {
            output,
            op: Arc::from(op),
            inputs: Arc::from(inputs),
            depth,
            requirement,
        });
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
    /// 要执行反向请用 [`Tape::backward`]：它剪掉与 loss 无关的分支，并按深度排序。
    pub fn backward_order(&self) -> Vec<Var> {
        self.nodes.iter().rev().map(Node::output).collect()
    }

    /// 为 `loss` 生成反向调度：只含可达节点，按深度降序排列。
    pub fn backward(&self, loss: Var) -> Result<Schedule> {
        let index = self.check(loss, "autograd::Tape::backward")?;
        if self.requirement[index].is_none() {
            return Err(self.reject(format!("变量 {} 不参与反向", loss.id())));
        }
        let reachable = self.reachable(loss);
        let mut steps: Vec<Node> = self
            .nodes
            .iter()
            .filter(|node| reachable[node.output.id() as usize])
            .cloned()
            .collect();
        steps.sort_by(|left, right| {
            right
                .depth()
                .cmp(&left.depth())
                .then(right.output().id().cmp(&left.output().id()))
        });
        let endpoints = reachable
            .iter()
            .enumerate()
            .filter(|(_, seen)| **seen)
            .filter(|(index, _)| self.requirement[*index].is_endpoint())
            .map(|(index, _)| Var { id: index as u32 })
            .collect();
        Ok(Schedule::new(loss, self.next, steps, endpoints))
    }

    /// 分配一个新编号，需求与深度写进平行数组。
    fn fresh(&mut self, requirement: Requirement, depth: u32) -> Var {
        let var = Var { id: self.next };
        self.next += 1;
        self.requirement.push(requirement);
        self.depth.push(depth);
        self.producer.push(None);
        var
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

    /// 从 `loss` 出发沿边反向走一遍，返回"变量是否可达"的位图。
    ///
    /// 显式栈而不是递归：图深与调用栈深无关，一万层的前向也能排。
    fn reachable(&self, loss: Var) -> Vec<bool> {
        let mut seen = vec![false; self.requirement.len()];
        let mut stack = vec![loss];
        seen[loss.id() as usize] = true;
        while let Some(var) = stack.pop() {
            let Some(node) = self.node(var) else {
                continue;
            };
            for input in node.inputs() {
                let index = input.id() as usize;
                if self.requirement[index].requires() && !seen[index] {
                    seen[index] = true;
                    stack.push(*input);
                }
            }
        }
        seen
    }

    /// 该变量是否被任何节点当作输入引用过。
    fn consumed(&self, index: usize) -> bool {
        self.nodes
            .iter()
            .any(|node| node.inputs.iter().any(|input| input.id() as usize == index))
    }

    /// 编号校验：越界即"引用了一个还没创建的变量"。
    fn check(&self, var: Var, site: &'static str) -> Result<usize> {
        if var.id() as usize >= self.requirement.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "变量 {} 不存在（当前已创建 {} 个）",
                    var.id(),
                    self.requirement.len()
                ),
            )
            .with(site));
        }
        Ok(var.id() as usize)
    }

    /// 本层的参数错误：调用顺序或声明写错了，不是缺能力。
    fn reject(&self, message: impl Into<String>) -> Error {
        Error::new(Code::Invalid, message).with("autograd::Tape")
    }
}

#[cfg(test)]
mod tests {
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
        let ghost = Var { id: 42 };
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
            tape.retain(Var { id: 7 }).expect_err("不存在").code(),
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
        let ghost = Var { id: 9 };
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
