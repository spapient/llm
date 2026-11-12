//! 反向调度：[`Schedule`] 是 [`crate::Tape::backward`] 的产物，[`Account`] 是执行它的账本。
//!
//! 两件事必须分开：调度是**事实**（哪些节点、什么顺序、谁是梯度终点），账本是**运行期状态**
//! （每个变量还差几份梯度贡献）。分开的好处是同一次前向可以反复反向——调度只读，账本可重建。

use crate::{Node, Var};
use diagnostic::{Code, Error, Result};

/// 一次反向传播的计划：只含 loss 可达的节点，按反向拓扑序排列。
///
/// 顺序是"深度降序、同深度编号降序"：
///
/// - 深度降序保证输出总在输入之前（向后一步），因此是合法的反向拓扑序；
/// - 同深度互不依赖，编号降序只是让同一种图每次排出来的顺序都一样，便于缓存与对拍。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    loss: Var,
    graph: Vec<bool>,
    steps: Vec<Node>,
    endpoints: Vec<Var>,
}

impl Schedule {
    /// 由 [`crate::Tape::backward`] 构造；顺序已经定好，这里只装载并算出图上变量位图。
    pub(crate) fn new(loss: Var, vars: u32, steps: Vec<Node>, endpoints: Vec<Var>) -> Self {
        let mut graph = vec![false; vars as usize];
        graph[loss.id() as usize] = true;
        for step in &steps {
            graph[step.output().id() as usize] = true;
        }
        for endpoint in &endpoints {
            graph[endpoint.id() as usize] = true;
        }
        Self {
            loss,
            graph,
            steps,
            endpoints,
        }
    }

    /// 反向的起点：它的梯度是 1，不参与累加。
    pub fn loss(&self) -> Var {
        self.loss
    }

    /// 反向步骤，按执行顺序。
    pub fn steps(&self) -> &[Node] {
        &self.steps
    }

    /// 需要取走的梯度：loss 可达、需求为 [`crate::Requirement::Grad`] 的变量，按编号升序。
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

    /// 按这个计划执行反向的账本。
    pub fn account(&self) -> Account {
        Account::new(self)
    }
}

/// 反向执行的账本：扇出计数与"谁的梯度已经齐全"。
///
/// 一个变量被多少个**可达**步骤当作输入引用，就要接收多少份贡献；收满之前的步骤不允许执行。
/// 计数一次建好（O(输入数)），之后每一步只做减法，没有哈希、没有扫描。
#[derive(Debug, Clone)]
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
    /// 统计每个变量的扇出：图上的每个输入位置算一份贡献。
    fn new(schedule: &Schedule) -> Self {
        let state: Vec<State> = schedule
            .graph
            .iter()
            .map(|on_graph| match on_graph {
                true => State::Waiting,
                false => State::Outside,
            })
            .collect();
        let mut account = Self {
            pending: vec![0u32; state.len()],
            state,
            steps: schedule.steps.len(),
            settled: 0,
        };
        for step in &schedule.steps {
            for input in step.inputs() {
                if account.graph(*input) {
                    account.pending[input.id() as usize] += 1;
                }
            }
        }
        account
    }

    /// 变量是否在图上（要接收梯度）。
    fn graph(&self, var: Var) -> bool {
        self.state
            .get(var.id() as usize)
            .is_some_and(|state| *state != State::Outside)
    }

    /// 还差几份梯度贡献；0 表示齐全。编号不在本次调度里时也返回 0。
    pub fn pending(&self, var: Var) -> u32 {
        self.pending.get(var.id() as usize).copied().unwrap_or(0)
    }

    /// 该变量的梯度是否已经齐全（它的反向步骤现在可以执行）。
    pub fn ready(&self, var: Var) -> bool {
        self.pending(var) == 0
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
    /// 三种误用都会失败而不是算错：步骤不属于这份调度、步骤已结算、梯度还没收满。
    pub fn settle(&mut self, step: &Node) -> Result<Vec<Var>> {
        let index = step.output().id() as usize;
        self.enter(step)?;
        self.state[index] = State::Done;
        self.settled += 1;
        let mut ready = Vec::with_capacity(step.inputs().len());
        for input in step.inputs() {
            if let Some(var) = self.contribute(*input)? {
                ready.push(var);
            }
        }
        Ok(ready)
    }

    /// 结算前的校验：步骤在计划里、没结算过、梯度已收满。
    fn enter(&self, step: &Node) -> Result<()> {
        let var = step.output();
        let index = var.id() as usize;
        match self.state.get(index) {
            Some(State::Waiting) => {}
            Some(State::Done) => {
                return Err(self.field(format!("变量 {} 的步骤已结算", var.id())));
            }
            _ => return Err(self.field(format!("变量 {} 不是本次调度的步骤", var.id()))),
        }
        let pending = self.pending[index];
        if pending > 0 {
            return Err(self.field(format!("变量 {} 的梯度还差 {pending} 份贡献", var.id())));
        }
        Ok(())
    }

    /// 记一份贡献；不进图的输入不收账，收到最后一份时返回该变量。
    fn contribute(&mut self, input: Var) -> Result<Option<Var>> {
        let index = input.id() as usize;
        if !self.graph(input) {
            return Ok(None);
        }
        let Some(pending) = self.pending.get_mut(index) else {
            return Err(self.field(format!("变量 {} 不在账本里", input.id())));
        };
        *pending = pending.checked_sub(1).ok_or_else(|| {
            Error::new(
                Code::Internal,
                format!("变量 {} 的扇出计数已为 0", input.id()),
            )
            .with("autograd::Account::settle")
        })?;
        Ok((*pending == 0).then_some(input))
    }

    /// 本层的参数错误：执行顺序与调度不一致。
    fn field(&self, message: impl Into<String>) -> Error {
        Error::new(Code::Invalid, message).with("autograd::Account::settle")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::{Requirement, Tape};

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

    /// 菱形图：两条路径的贡献都到齐，x 的梯度才算齐全。
    #[test]
    fn a_diamond_waits_for_every_contribution() {
        let mut tape = Tape::new();
        let x = tape.parameter();
        let left = tape.record("relu", &[x]).expect("记录");
        let right = tape.record("relu", &[x]).expect("记录");
        let joined = tape.record("add", &[left, right]).expect("记录");
        let loss = tape.record("reduce_sum", &[joined]).expect("记录");
        let schedule = tape.backward(loss).expect("在图上");
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
        let strategy = tape.backward(loss).expect("在图上");
        let steps = strategy.steps();
        let mut account = strategy.account();
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

    /// 随机图上逐条核对不变量：可达性、拓扑序、扇出计数与账本推进。
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
                .filter(|id| tape.requirement(Var { id: *id }) == Some(Requirement::Grad))
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
            check_account(&tape, &schedule, &vars, round);
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

    /// 账本：扇出计数与独立统计一致，按序结算总能走完。
    fn check_account(tape: &Tape, schedule: &Schedule, vars: &BTreeSet<u32>, round: u32) {
        let mut expected = vec![0u32; tape.vars() as usize];
        for step in schedule.steps() {
            for input in step.inputs() {
                if vars.contains(&input.id()) {
                    expected[input.id() as usize] += 1;
                }
            }
        }
        let mut account = schedule.account();
        for id in 0..tape.vars() {
            assert_eq!(
                account.pending(Var { id }),
                expected[id as usize],
                "第 {round} 轮：变量 {id} 的扇出不一致"
            );
        }
        for step in schedule.steps() {
            for ready in account.settle(step).expect("按序结算") {
                assert!(
                    tape.requirement(ready).is_some_and(Requirement::requires),
                    "第 {round} 轮：不进图的变量不该收到梯度"
                );
                assert_eq!(account.pending(ready), 0);
            }
        }
        assert!(account.finished(), "第 {round} 轮：还有步骤没结算");
        assert_eq!(account.settled(), schedule.len());
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
                    node.inputs().iter().copied().filter(|input| {
                        tape.requirement(*input).is_some_and(Requirement::requires)
                    }),
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

    /// 确定性 PRNG：不给依赖表加 `rand`，同时让失败可复现。
    struct Rng {
        state: u32,
    }

    impl Rng {
        fn new(seed: u32) -> Self {
            Self { state: seed }
        }

        fn below(&mut self, bound: u32) -> u32 {
            self.state ^= self.state << 13;
            self.state ^= self.state >> 17;
            self.state ^= self.state << 5;
            self.state % bound.max(1)
        }
    }
}
