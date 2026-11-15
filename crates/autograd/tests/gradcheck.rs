//! 数值对拍：用调度与账本驱动一套**测试自带**的标量反向公式，梯度与中心差分逐参数比对。
//!
//! 数值属于测试而不是本层：这里只验证"顺序 + 扇出 + 终点剪枝"足以复现正确的梯度，
//! 口径与 `llm.u` 的 gradcheck 一致——`|Δ| ≤ 2e-3 + 0.05·max(|fd|, 1)`，不允许 NaN/Inf，
//! 同一前向重复反向必须逐位一致，图外参数的梯度必须恰好为 0。

use llm_autograd::{Node, Requirement, Schedule, Tape, Var};
use testkit::Rng;

/// 绝对项（`llm.u` 的 gradcheck 口径）。
const ABSOLUTE: f64 = 2e-3;
/// 相对项：大梯度允许按比例放宽。
const RELATIVE: f64 = 0.05;
/// 中心差分的步长：f64 前向，`1e-6` 已在舍入误差与截断误差之间。
const STEP: f64 = 1e-6;

/// 一个随机的标量图：变量值按编号稠密存放，叶子与参数的值由用例给出。
struct Case {
    tape: Tape,
    values: Vec<f64>,
    loss: Var,
}

impl Case {
    /// 前向：节点按记录顺序求值，因此每个输入都已经有值。
    fn forward(&self, values: &[f64]) -> Vec<f64> {
        let mut out = values.to_vec();
        for node in self.tape.nodes() {
            let args: Vec<f64> = node
                .inputs()
                .iter()
                .map(|input| out[input.id() as usize])
                .collect();
            out[node.output().id() as usize] = apply(node.op(), &args);
        }
        out
    }

    /// loss 作为一个纯函数：把某个变量的值换掉再重算。
    fn loss_at(&self, var: Var, value: f64) -> f64 {
        let mut values = self.values.clone();
        values[var.id() as usize] = value;
        self.forward(&values)[self.loss.id() as usize]
    }

    /// 中心差分：不用解析公式，只反复前向。
    fn difference(&self, var: Var) -> f64 {
        let value = self.values[var.id() as usize];
        let plus = self.loss_at(var, value + STEP);
        let minus = self.loss_at(var, value - STEP);
        (plus - minus) / (2.0 * STEP)
    }

    /// 参数：所有需求为终点的叶子。
    fn parameters(&self) -> Vec<Var> {
        (0..self.tape.vars())
            .filter_map(|id| self.tape.var(id))
            .filter(|var| self.tape.requirement(*var) == Some(Requirement::Grad))
            .filter(|var| self.tape.node(*var).is_none())
            .collect()
    }
}

/// 五个光滑算子的前向：没有不可导点，中心差分才有意义。
fn apply(op: &str, args: &[f64]) -> f64 {
    match op {
        "add" => args[0] + args[1],
        "mul" => args[0] * args[1],
        "scale" => 2.0 * args[0],
        "square" => args[0] * args[0],
        "neg" => -args[0],
        other => panic!("未定义算子 {other}"),
    }
}

/// 五个光滑算子的反向：把 `incoming` 按各自的解析公式分给输入。
///
/// 只往图上的变量累加，和运行时一样：常量没有梯度缓冲，也没人取它的梯度。
fn accumulate(tape: &Tape, node: &Node, values: &[f64], grads: &mut [f64]) {
    let incoming = grads[node.output().id() as usize];
    let inputs = node.inputs();
    let first = inputs[0].id() as usize;
    let a = values[first];
    match node.op() {
        "add" => {
            feed(tape, inputs[0], incoming, grads);
            feed(tape, inputs[1], incoming, grads);
        }
        "mul" => {
            feed(
                tape,
                inputs[0],
                incoming * values[inputs[1].id() as usize],
                grads,
            );
            feed(tape, inputs[1], incoming * a, grads);
        }
        "scale" => feed(tape, inputs[0], 2.0 * incoming, grads),
        "square" => feed(tape, inputs[0], 2.0 * a * incoming, grads),
        "neg" => feed(tape, inputs[0], -incoming, grads),
        other => panic!("未定义算子 {other}"),
    }
}

/// 记一份贡献；不在图上的输入没有梯度缓冲，直接丢弃。
fn feed(tape: &Tape, input: Var, contribution: f64, grads: &mut [f64]) {
    let receives = tape.requirement(input).is_some_and(|r| !r.is_none());
    if receives {
        grads[input.id() as usize] += contribution;
    }
}

/// 依次结算整个计划，返回每个变量的梯度。
fn gradients(case: &Case, schedule: &Schedule) -> Vec<f64> {
    let values = case.forward(&case.values);
    let mut grads = vec![0.0; case.tape.vars() as usize];
    grads[schedule.loss().id() as usize] = 1.0;
    let mut account = schedule.account();
    for step in schedule.steps() {
        accumulate(&case.tape, step, &values, &mut grads);
        account.settle(step).expect("按序");
    }
    assert!(account.finished(), "计划里的每一步都应结算");
    grads
}

/// 按 `levels()` 分批结算：同层顺序随意，结果必须与顺序执行逐位一致。
fn gradients_by_level(case: &Case, schedule: &Schedule) -> Vec<f64> {
    let values = case.forward(&case.values);
    let mut grads = vec![0.0; case.tape.vars() as usize];
    grads[schedule.loss().id() as usize] = 1.0;
    let mut account = schedule.account();
    for level in schedule.levels() {
        for step in level.iter().rev() {
            assert!(account.ready(step.output()), "进入本层时梯度应当已齐");
            accumulate(&case.tape, step, &values, &mut grads);
            account.settle(step).expect("层内顺序不影响正确性");
        }
    }
    assert!(account.finished());
    grads
}

/// 一个用例的全部断言。
fn check(case: &Case, round: u32) {
    let schedule = case.tape.backward(case.loss).expect("loss 在图上");
    let analytic = gradients(case, &schedule);
    let levels = gradients_by_level(case, &schedule);
    for (var, (left, right)) in analytic.iter().zip(&levels).enumerate() {
        let slack = 1e-9 * left.abs().max(right.abs()).max(1.0);
        assert!(
            (left - right).abs() <= slack,
            "第 {round} 轮：变量 {var} 的分层梯度 {right} 与顺序梯度 {left} 不一致"
        );
    }
    assert_eq!(
        analytic,
        gradients(case, &schedule),
        "第 {round} 轮：重复反向不逐位一致"
    );
    for value in &analytic {
        assert!(value.is_finite(), "第 {round} 轮：解析梯度出现 NaN/Inf");
    }
    for parameter in case.parameters() {
        let numeric = case.difference(parameter);
        let actual = analytic[parameter.id() as usize];
        let slack = ABSOLUTE + RELATIVE * numeric.abs().max(1.0);
        assert!(
            (actual - numeric).abs() <= slack,
            "第 {round} 轮：变量 {} 的解析梯度 {actual} 与差分 {numeric} 超差 {slack}",
            parameter.id()
        );
    }
    for id in 0..case.tape.vars() {
        let var = case.tape.var(id).expect("编号存在");
        if !on_graph(&case.tape, var, case.loss) {
            assert_eq!(
                analytic[var.id() as usize],
                0.0,
                "第 {round} 轮：不在图上的变量 {} 收到了梯度",
                var.id()
            );
            assert!(
                !schedule.endpoints().contains(&var),
                "第 {round} 轮：图外变量 {} 被当成终点",
                var.id()
            );
        }
    }
}

/// 变量是否在 loss 的图上：从 loss 沿"要梯度"的边能走到它。
fn on_graph(tape: &Tape, target: Var, loss: Var) -> bool {
    let mut seen = vec![false; tape.vars() as usize];
    let mut stack = vec![loss];
    seen[loss.id() as usize] = true;
    while let Some(var) = stack.pop() {
        if var == target {
            return true;
        }
        let Some(node) = tape.node(var) else {
            continue;
        };
        for input in node.inputs() {
            let on_graph = tape.requirement(*input).is_some_and(|r| !r.is_none());
            if on_graph && !seen[input.id() as usize] {
                seen[input.id() as usize] = true;
                stack.push(*input);
            }
        }
    }
    false
}

#[test]
fn random_graphs_match_central_differences() {
    let mut rng = Rng::new(0xC0FF_EE12);
    for round in 0..200 {
        let case = random_case(&mut rng);
        check(&case, round);
    }
}

#[test]
fn a_shared_input_is_accumulated_once_per_edge() {
    // loss = (x * x) + (x + x)：d/dx = 2x + 2 = 6（x = 2）。
    let mut tape = Tape::new();
    let x = tape.parameter();
    let square = tape.record("square", &[x]).expect("记录");
    let doubled = tape.record("add", &[x, x]).expect("记录");
    let loss = tape.record("add", &[square, doubled]).expect("记录");
    let mut values = vec![0.0; tape.vars() as usize];
    values[x.id() as usize] = 2.0;
    let case = Case { tape, values, loss };
    check(&case, 0);
    let analytic = gradients(&case, &case.tape.backward(loss).expect("在图上"));
    assert_eq!(analytic[x.id() as usize], 6.0);
}

#[test]
fn a_detached_branch_contributes_nothing() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let frozen = tape
        .record_with("square", &[x], Requirement::None)
        .expect("记录");
    let loss = tape.record("add", &[frozen, x]).expect("记录");
    let mut values = vec![0.0; tape.vars() as usize];
    values[x.id() as usize] = 3.0;
    let case = Case { tape, values, loss };
    let schedule = case.tape.backward(loss).expect("在图上");
    assert_eq!(schedule.len(), 1, "断开的分支排在计划之外");
    let analytic = gradients(&case, &schedule);
    assert_eq!(analytic[x.id() as usize], 1.0, "d/dx (x) = 1，平方项已断开");
}

/// 随机前向：参数、常量与五个算子混在一起，loss 取最后一个要梯度的变量。
fn random_case(rng: &mut Rng) -> Case {
    let mut tape = Tape::new();
    let mut vars: Vec<Var> = (0..2 + rng.below(4)).map(|_| tape.parameter()).collect();
    vars.extend((0..rng.below(3)).map(|_| tape.leaf()));
    for _ in 0..rng.below(12) {
        let node = match rng.below(5) {
            0 => tape.record("add", &[pick(rng, &vars), pick(rng, &vars)]),
            1 => tape.record("mul", &[pick(rng, &vars), pick(rng, &vars)]),
            2 => tape.record("scale", &[pick(rng, &vars)]),
            3 => tape.record("square", &[pick(rng, &vars)]),
            _ => tape.record("neg", &[pick(rng, &vars)]),
        };
        if let Ok(var) = node {
            vars.push(var);
        }
    }
    let loss = vars
        .iter()
        .rev()
        .find(|var| tape.requirement(**var).is_some_and(|r| !r.is_none()))
        .copied()
        .unwrap_or_else(|| tape.record("scale", &[vars[0]]).expect("记录"));
    let mut values = vec![0.0; tape.vars() as usize];
    for var in 0..tape.vars() {
        values[var as usize] = rng.value();
    }
    Case { tape, values, loss }
}

/// 从已有变量里随机挑一个。
fn pick(rng: &mut Rng, vars: &[Var]) -> Var {
    vars[rng.below(vars.len() as u32) as usize]
}
