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
mod tests {
use std::collections::{BTreeMap, BTreeSet};

use diagnostic::Code;
use testkit::Rng;

use super::*;

#[test]
fn insertion_order_is_the_reverse_record_order() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let y = tape.record("mul", &[x]).expect("记录");
    let z = tape.record("add", &[y]).expect("记录");
    let mut schedule = Schedule::default();
    schedule
        .plan_with(&tape, z, Ordering::Insertion)
        .expect("规划");
    let outputs: Vec<_> = schedule.steps().iter().map(|node| node.output()).collect();
    assert_eq!(outputs, [z, y]);
    assert_eq!(schedule.ordering(), Ordering::Insertion);
}

/// 分支不参与 loss 时不进计划、也不是梯度终点。
#[test]
fn branches_that_miss_the_loss_are_pruned() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let used = tape.record("relu", &[x]).expect("记录");
    let _unused = tape.record("relu", &[x]).expect("记录");
    let loss = tape.record("reduce_sum", &[used]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    assert_eq!(
        schedule
            .steps()
            .iter()
            .map(Node::output)
            .collect::<Vec<_>>(),
        [loss, used]
    );
    assert_eq!(schedule.endpoints(), [x]);
}

/// 深度相同的步骤按编号降序排，顺序可复现。
#[test]
fn steps_are_ordered_by_depth_then_by_id() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let left = tape.record("relu", &[x]).expect("记录");
    let right = tape.record("relu", &[x]).expect("记录");
    let joined = tape.record("add", &[left, right]).expect("记录");
    let loss = tape.record("reduce_sum", &[joined]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    assert_eq!(
        schedule
            .steps()
            .iter()
            .map(Node::output)
            .collect::<Vec<_>>(),
        [loss, joined, right, left]
    );
    assert_eq!(
        schedule.steps().iter().map(Node::depth).collect::<Vec<_>>(),
        [3, 2, 1, 1]
    );
}

/// 深链不递归：两万层也不会栈溢出。
#[test]
fn a_deep_chain_does_not_recurse() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let mut current = x;
    for _ in 0..20_000 {
        current = tape.record("relu", &[current]).expect("记录");
    }
    let loss = tape.record("reduce_sum", &[current]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    assert_eq!(schedule.len(), 20_001);
    let mut account = schedule.account();
    for step in schedule.steps() {
        account.settle(step).expect("拓扑序保证每一步都就绪");
    }
    assert!(account.finished());
    assert_eq!(schedule.endpoints(), [x]);
}

/// 同一张图反复 backward，计划逐字段一致。
#[test]
fn the_schedule_is_deterministic() {
    fn build() -> (Tape, Var) {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let y = tape.parameter();
        let a = tape.record("mul", &[x, y]).expect("记录");
        let b = tape.record("relu", &[a]).expect("记录");
        let loss = tape.record("reduce_sum", &[a, b]).expect("记录");
        (tape, loss)
    }
    let (tape, loss) = build();
    let first = tape.backward(loss).expect("在图上");
    let second = tape.backward(loss).expect("在图上");
    assert_eq!(first, second);
    assert_eq!(first.account().pending(loss), 0);
}

/// 复用同一个 `Schedule`：步骤表的缓冲地址不变，说明没有重新分配。
#[test]
fn planning_into_an_existing_schedule_reuses_its_buffers() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let left = tape.record("relu", &[x]).expect("记录");
    let right = tape.record("relu", &[x]).expect("记录");
    let loss = tape.record("add", &[left, right]).expect("记录");
    let mut schedule = Schedule::new();
    tape.plan_into(&mut schedule, loss).expect("在图上");
    let steps = schedule.steps().as_ptr();
    let endpoints = schedule.endpoints().as_ptr();
    let graph = schedule.graph().as_ptr();
    tape.plan_into(&mut schedule, loss).expect("在图上");
    assert_eq!(schedule.steps().as_ptr(), steps, "步骤表没有重新分配");
    assert_eq!(
        schedule.endpoints().as_ptr(),
        endpoints,
        "终点表没有重新分配"
    );
    assert_eq!(schedule.graph().as_ptr(), graph, "可达位图没有重新分配");
}

/// 同一张图换一个 loss 重新规划：旧步骤不复用、终点跟着变。
#[test]
fn replanning_with_another_loss_replaces_the_whole_plan() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let y = tape.parameter();
    let branch = tape.record("relu", &[x]).expect("记录");
    let loss = tape.record("add", &[branch, y]).expect("记录");
    let mut schedule = tape.backward(loss).expect("在图上");
    assert_eq!(schedule.endpoints(), [x, y]);
    tape.plan_into(&mut schedule, branch).expect("在图上");
    assert_eq!(schedule.loss(), branch);
    assert_eq!(schedule.len(), 1, "只剩 branch 自己的反向步骤");
    assert_eq!(schedule.endpoints(), [x], "y 不在这个 loss 的图上");
}

/// 分层：同层互不依赖，前面的层结算完，这一层的每一步都已就绪。
#[test]
fn levels_partition_the_schedule_and_each_level_is_ready() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let shared = tape.record("relu", &[x]).expect("记录");
    let left = tape.record("mul", &[shared, x]).expect("记录");
    let right = tape.record("add", &[shared, x]).expect("记录");
    let loss = tape.record("add", &[left, right]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    let mut account = schedule.account();
    let mut steps = 0;
    for level in schedule.levels() {
        let depth = level[0].depth();
        assert!(
            level.iter().all(|node| node.depth() == depth),
            "同层深度一致"
        );
        for step in level {
            assert!(
                account.ready(step.output()),
                "变量 {} 在进入它的层时梯度应当已收满",
                step.output().id()
            );
            account.settle(step).expect("同层可按序结算");
            steps += 1;
        }
    }
    assert_eq!(steps, schedule.len(), "分层覆盖全部步骤且不重复");
}

/// 失败的规划不改变已有计划：调用方拿到 `Err` 时不会得到半份计划。
#[test]
fn a_failed_plan_keeps_the_previous_schedule() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let loss = tape.record("relu", &[x]).expect("记录");
    let mut schedule = tape.backward(loss).expect("在图上");
    let before = schedule.clone();
    for bad in [Var::at(99), tape.leaf()] {
        let error = tape.plan_into(&mut schedule, bad).expect_err("不可规划");
        assert_eq!(error.code(), Code::Invalid);
        assert_eq!(schedule, before, "失败的规划不留半份计划");
    }
}

/// 用最小标量引擎跑完整反向：调度的顺序与扇出足够复现解析梯度。
///
/// 数值由这里给出的反向公式累积，和运行时一样：本层只提供顺序与"何时齐全"，不提供数。
#[test]
fn a_toy_engine_reproduces_analytic_gradients() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let y = tape.parameter();
    let sum = tape.record("add", &[x, y]).expect("记录");
    let product = tape.record("mul", &[x, y]).expect("记录");
    let loss = tape.record("add", &[sum, product]).expect("记录");
    // 前向取值：x = 2、y = 3，于是 loss = (x + y) + x * y = 11。
    let schedule = tape.backward(loss).expect("在图上");
    let mut account = schedule.account();
    let mut grads: BTreeMap<u32, f32> = BTreeMap::from([(loss.id(), 1.0)]);
    for step in schedule.steps() {
        account.settle(step).expect("按序");
        let incoming = grads[&step.output().id()];
        match step.op() {
            "add" => {
                for input in step.inputs() {
                    *grads.entry(input.id()).or_insert(0.0) += incoming;
                }
            }
            "mul" => {
                let values = [2.0, 3.0]; // 与 x、y 的顺序一致
                for (slot, input) in step.inputs().iter().enumerate() {
                    *grads.entry(input.id()).or_insert(0.0) += incoming * values[1 - slot];
                }
            }
            op => panic!("未定义的算子 {op}"),
        }
    }
    assert_eq!(schedule.endpoints(), [x, y]);
    assert_eq!(grads[&x.id()], 4.0, "d/dx = y + 1");
    assert_eq!(grads[&y.id()], 3.0, "d/dy = x + 1");
    assert!(account.finished());
}

/// 随机图上逐条核对不变量：可达性、拓扑序、终点与分层。
#[test]
fn random_graphs_keep_every_invariant() {
    let mut rng = Rng::new(0x5EED_1234);
    for round in 0..400 {
        let Some((tape, loss)) = random_graph(&mut rng) else {
            continue;
        };
        let (nodes, vars) = reference(&tape, loss);
        let schedule = tape.backward(loss).expect("loss 在图上");
        let position = positions(&schedule);
        assert_eq!(
            position.keys().copied().collect::<BTreeSet<_>>(),
            nodes,
            "第 {round} 轮：可达节点与计划不一致"
        );
        let expected_endpoints: Vec<u32> = vars
            .iter()
            .copied()
            .filter(|id| tape.requirement(Var::at(*id)) == Some(Requirement::Grad))
            .collect();
        assert_eq!(
            schedule
                .endpoints()
                .iter()
                .map(|var| var.id())
                .collect::<Vec<_>>(),
            expected_endpoints,
            "第 {round} 轮：终点与可达的 Grad 变量不一致"
        );
        check_order(&tape, &schedule, &position, round);
        check_levels(&schedule, round);
    }
}

/// 拓扑序：每一步的入边都必须排在它后面，深度不得回升。
fn check_order(tape: &Tape, schedule: &Schedule, position: &BTreeMap<u32, usize>, round: u32) {
    let mut previous = u32::MAX;
    for step in schedule.steps() {
        assert!(step.depth() <= previous, "第 {round} 轮：深度回升");
        previous = step.depth();
        for input in step.inputs() {
            if !tape.requirement(*input).is_some_and(Requirement::requires) {
                continue;
            }
            if let Some(next) = position.get(&input.id()) {
                assert!(
                    *next > position[&step.output().id()],
                    "第 {round} 轮：输入 {} 排在了输出 {} 之前",
                    input.id(),
                    step.output().id()
                );
            }
        }
    }
}

/// 分层：每层深度一致、颜色连续，且层数等于不同深度的个数。
fn check_levels(schedule: &Schedule, round: u32) {
    let mut levels = schedule.levels().peekable();
    let mut seen = BTreeSet::new();
    let mut total = 0;
    while let Some(level) = levels.next() {
        let depth = level[0].depth();
        assert!(seen.insert(depth), "第 {round} 轮：深度 {depth} 的层重复");
        assert!(
            level.iter().all(|node| node.depth() == depth),
            "第 {round} 轮：一层里混了不同深度"
        );
        if let Some(next) = levels.peek() {
            assert!(
                next[0].depth() < depth,
                "第 {round} 轮：层的顺序不是深度降序"
            );
        }
        total += level.len();
    }
    assert_eq!(total, schedule.len(), "第 {round} 轮：分层没有覆盖全部步骤");
}

/// 计划里每一步的位置。
fn positions(schedule: &Schedule) -> BTreeMap<u32, usize> {
    schedule
        .steps()
        .iter()
        .enumerate()
        .map(|(index, step)| (step.output().id(), index))
        .collect()
}

/// 独立实现的可达性：从 loss 沿需求为真的边反向走。
fn reference(tape: &Tape, loss: Var) -> (BTreeSet<u32>, BTreeSet<u32>) {
    let mut vars = BTreeSet::new();
    let mut nodes = BTreeSet::new();
    let mut stack = vec![loss];
    while let Some(var) = stack.pop() {
        if !vars.insert(var.id()) {
            continue;
        }
        if let Some(node) = tape.node(var) {
            nodes.insert(var.id());
            stack.extend(
                node.inputs()
                    .iter()
                    .copied()
                    .filter(|input| tape.requirement(*input).is_some_and(Requirement::requires)),
            );
        }
    }
    (nodes, vars)
}

/// 随机前向：参数、常量、detach 与显式终点混在一起，loss 取最后一个要梯度的变量。
fn random_graph(rng: &mut Rng) -> Option<(Tape, Var)> {
    let mut tape = Tape::new();
    let mut vars: Vec<Var> = (0..1 + rng.below(5)).map(|_| tape.parameter()).collect();
    vars.extend((0..rng.below(4)).map(|_| tape.leaf()));
    for _ in 0..rng.below(20) {
        let width = 1 + rng.below(3) as usize;
        let inputs: Vec<Var> = (0..width)
            .map(|_| vars[rng.below(vars.len() as u32) as usize])
            .collect();
        let recorded = match rng.below(10) {
            0 => tape.record_with("detach", &inputs, Requirement::None),
            1 => tape.record_with("retain", &inputs, Requirement::Grad),
            _ => tape.record("edge", &inputs),
        };
        if let Ok(var) = recorded {
            vars.push(var);
        }
    }
    let loss = *vars
        .iter()
        .rev()
        .find(|var| tape.requirement(**var).is_some_and(Requirement::requires))?;
    Some((tape, loss))
}
}

