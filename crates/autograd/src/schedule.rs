//! 反向调度的**事实**：[`Schedule`] 是一次 `backward` 的结果——哪些步骤、什么顺序、谁是终点。

use diagnostic::Result;

use crate::account::Account;
use crate::graph::{Node, Requirement, Role, Var};
use crate::invalid;
use crate::plan::Config;
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
    fn mark(&mut self, tape: &Tape, loss: Var) {
        // `reset` 已把位图补成等长的全 false，这里不再整表清零。
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
            let idx = input.id() as usize;
            if tape.requirement(*input).is_some_and(Requirement::requires) {
                self.graph[idx] = true;
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
        match ordering {
            // 插入序就是前向逆序，倒着过滤一遍即可，省掉一次排序。
            Ordering::Insertion => steps.extend(
                tape.nodes()
                    .iter()
                    .rev()
                    .filter(|node| graph[node.output().id() as usize])
                    .cloned(),
            ),
            Ordering::Depth => {
                steps.extend(
                    tape.nodes()
                        .iter()
                        .filter(|node| graph[node.output().id() as usize])
                        .cloned(),
                );
                steps.sort_unstable_by(|left, right| compare(*ordering, left, right));
            }
        }
        endpoints.extend(graph.iter().enumerate().filter_map(|(idx, seen)| {
            let var = Var::at(idx as u32);
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

/// 一张图与一次反向规划的计数快照。
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
#[path = "../tests/schedule_tests.rs"]
mod tests;
