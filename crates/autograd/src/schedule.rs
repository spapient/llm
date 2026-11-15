//! 反向调度的**事实**：[`Schedule`] 是一次 `backward` 的结果——哪些步骤、什么顺序、谁是终点。
//!
//! 调度只读、可重复使用：把同一张图的下一次前向重新规划进同一个 `Schedule`，缓冲留在原处。

use diagnostic::Result;

use crate::account::Account;
use crate::config::Config;
use crate::error::invalid;
use crate::graph::{Node, Requirement, Var};
use crate::tape::Tape;

/// 反向步骤的排序策略：两种都是合法的反向拓扑序，差别只在同层之间的次序。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Ordering {
    /// 深度降序、同深度编号降序：同层互不依赖，分层 [`Schedule::levels`] 直接可用。
    #[default]
    Depth,
    /// 记录编号降序：与前向磁带严格逆序，不重算深度，图很深时规划更省一次比较。
    Insertion,
}

impl Ordering {
    /// 策略短名：诊断与报告里的稳定标签。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Depth => "depth",
            Self::Insertion => "insertion",
        }
    }
}

/// 一次反向传播的计划：只含 loss 可达的节点，按反向拓扑序排列。
///
/// 默认顺序是"深度降序、同深度编号降序"：
///
/// - 深度降序保证输出总在输入之前（向后一步），因此是合法的反向拓扑序；
/// - 同深度互不依赖，编号降序只是让同一种图每次排出来的顺序都一样，便于缓存与对拍。
///
/// 训练循环里应当复用一个 `Schedule`（[`Tape::plan_into`]）：`clear` 之后重填，
/// 稳定态不再分配。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    loss: Var,
    vars: u32,
    ordering: Ordering,
    graph: Vec<bool>,
    steps: Vec<Node>,
    endpoints: Vec<Var>,
}

impl Default for Schedule {
    fn default() -> Self {
        // 空计划还没有起点：`loss` 只是占位，规划一次之前它不对应任何变量。
        Self {
            loss: Var::at(0),
            vars: 0,
            ordering: Ordering::default(),
            graph: Vec::new(),
            steps: Vec::new(),
            endpoints: Vec::new(),
        }
    }
}

impl Schedule {
    /// 空的计划：还没有规划过任何 loss，也没有分配缓冲。
    ///
    /// 它存在的意义是给训练循环一个占位符——[`Tape::plan_into`] 可以反复往里写，
    /// 第一次规划才分配，之后都复用容量。
    pub fn new() -> Self {
        Self::default()
    }

    /// 反向的起点：它的梯度是 1，不参与累加。
    pub fn loss(&self) -> Var {
        self.loss
    }

    /// 反向步骤，按执行顺序。
    pub fn steps(&self) -> &[Node] {
        &self.steps
    }

    /// 需要取走的梯度：loss 可达、需求为 [`Requirement::Grad`] 的变量，按编号升序。
    ///
    /// 不可达的终点不出现在这里——这正是剪枝的价值：没参与 loss 的参数不该被优化器更新，
    /// 也不该被分配梯度缓冲。
    pub fn endpoints(&self) -> &[Var] {
        &self.endpoints
    }

    /// 步骤数。
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// 是否没有步骤（loss 本身就是叶子）。
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// 按深度分层：同一层的步骤互不依赖，可以并行/合批执行。
    ///
    /// 层按执行顺序给出，第 0 层是 loss 所在深度、最先执行。分层不是随意的：边总是从
    /// 浅层指向深层，所以一个深度为 `d` 的变量只会被深度 `> d` 的步骤消费——等前面的层
    /// 全部结算完，这一层的每一步都必然已经收满梯度。
    pub fn levels(&self) -> impl Iterator<Item = &[Node]> {
        self.steps
            .chunk_by(|left, right| left.depth() == right.depth())
    }

    /// 本计划采用的排序策略。
    pub fn ordering(&self) -> Ordering {
        self.ordering
    }

    /// 按这个计划执行反向的账本。
    pub fn account(&self) -> Account {
        let mut account = Account::new();
        account.rebuild(self);
        account
    }

    /// 图上的变量位图：终点与每个步骤的输出。
    pub(crate) fn graph(&self) -> &[bool] {
        &self.graph
    }

    /// 把 `loss` 的规划写进本对象（默认 [`Ordering::Depth`]）；失败时保持原样。
    pub(crate) fn plan(&mut self, tape: &Tape, loss: Var) -> Result<()> {
        self.plan_with(tape, loss, Ordering::default())
    }

    /// 按指定排序策略把 `loss` 的规划写进本对象；失败时保持原样。
    pub(crate) fn plan_with(&mut self, tape: &Tape, loss: Var, ordering: Ordering) -> Result<()> {
        Self::validate(tape, loss)?;
        self.reset(loss, tape.vars());
        self.ordering = ordering;
        self.mark(tape, loss);
        self.harvest(tape);
        Ok(())
    }

    /// 按配置里的排序策略规划；失败时保持原样。
    pub(crate) fn plan_config(&mut self, tape: &Tape, loss: Var, config: &Config) -> Result<()> {
        self.plan_with(tape, loss, config.ordering())
    }

    /// 校验起点：它必须存在，而且必须在图上。
    fn validate(tape: &Tape, loss: Var) -> Result<()> {
        match tape.requirement(loss) {
            Some(requirement) if requirement.requires() => Ok(()),
            Some(_) => Err(invalid(
                "autograd::Tape::plan_into",
                format!("变量 {} 不参与反向", loss.id()),
            )),
            None => Err(invalid(
                "autograd::Tape::plan_into",
                format!("变量 {} 不存在（当前已创建 {} 个）", loss.id(), tape.vars()),
            )),
        }
    }

    /// 复用缓冲：位图按变量数补零，表清空但保留容量。
    fn reset(&mut self, loss: Var, vars: u32) {
        self.loss = loss;
        self.vars = vars;
        self.graph.clear();
        self.graph.resize(vars as usize, false);
        self.steps.clear();
        self.endpoints.clear();
    }

    /// 从 `loss` 出发沿边反向走一遍，把"变量是否可达"写进位图。
    ///
    /// 边只能从小编号指向大编号（输入先于输出记录），因此**按编号降序遍历节点**时，
    /// 一个节点的可达性已经由它全部消费者决定：这正是拓扑序的逆序，不需要显式栈，
    /// 也不需要递归——两万层的前向与两层一样走一遍。
    fn mark(&mut self, tape: &Tape, loss: Var) {
        self.graph.fill(false);
        self.graph[loss.id() as usize] = true;
        for node in tape.nodes().iter().rev() {
            if !self.graph[node.output().id() as usize] {
                continue;
            }
            self.inherit(tape, node);
        }
    }

    /// 一个可达节点把可达性传给它的每个"要梯度"的输入。
    fn inherit(&mut self, tape: &Tape, node: &Node) {
        for input in node.inputs() {
            let index = input.id() as usize;
            if tape.requirement(*input).is_some_and(Requirement::requires) {
                self.graph[index] = true;
            }
        }
    }

    /// 把可达节点按选定策略搬进步骤表，并收集终点。
    fn harvest(&mut self, tape: &Tape) {
        let Self {
            graph,
            steps,
            endpoints,
            ordering,
            ..
        } = self;
        steps.extend(
            tape.nodes()
                .iter()
                .filter(|node| graph[node.output().id() as usize])
                .cloned(),
        );
        steps.sort_by(|left, right| compare(*ordering, left, right));
        endpoints.extend(graph.iter().enumerate().filter_map(|(index, seen)| {
            let var = Var::at(index as u32);
            let endpoint = *seen && tape.requirement(var) == Some(Requirement::Grad);
            endpoint.then_some(var)
        }));
    }
}

/// 两种策略各自的比较方向；都是合法反向拓扑序（输出编号必大于输入）。
fn compare(ordering: Ordering, left: &Node, right: &Node) -> core::cmp::Ordering {
    match ordering {
        Ordering::Depth => right
            .depth()
            .cmp(&left.depth())
            .then(right.output().id().cmp(&left.output().id())),
        Ordering::Insertion => right.output().id().cmp(&left.output().id()),
    }
}

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod tests;
