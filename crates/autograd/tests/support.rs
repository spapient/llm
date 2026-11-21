#![allow(dead_code)]

use testkit::Rng;

use crate::{Requirement, Schedule, Tape, Var};

/// 两条 `relu` 从同一个参数分叉；磁带留给调用方接下游。
pub(crate) fn fork() -> (Tape, Var, Var, Var) {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let left = tape.record("relu", &[x]).expect("记录");
    let right = tape.record("relu", &[x]).expect("记录");
    (tape, x, left, right)
}

/// 菱形图 `loss = reduce_sum(add(relu(x), relu(x)))`：两条路径的贡献都回到 `x`，它的梯度
/// 才算齐全。建好即 backward 出执行序，返回 `(x, left, right, joined, loss, 调度)`——
/// 调度自带节点，磁带用完即可丢掉。
pub(crate) fn diamond() -> (Var, Var, Var, Var, Var, Schedule) {
    let (mut tape, x, left, right) = fork();
    let joined = tape.record("add", &[left, right]).expect("记录");
    let loss = tape.record("reduce_sum", &[joined]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    (x, left, right, joined, loss, schedule)
}

/// 两级线性链 `x → mul → add`：排序与配置测试的固定形状。
pub(crate) fn pair() -> (Tape, Var, Var, Var) {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let y = tape.record("mul", &[x]).expect("记录");
    let z = tape.record("add", &[y]).expect("记录");
    (tape, x, y, z)
}

/// `parameter → relu → reduce_sum`：一个中间激活、一个 loss，账本测试的最小图。
pub(crate) fn linear() -> (Tape, Var, Var, Var, Schedule) {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let y = tape.record("relu", &[x]).expect("记录");
    let loss = tape.record("reduce_sum", &[y]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    (tape, x, y, loss, schedule)
}

/// 随机前向：`rounds` 轮里每轮取 1..=3 个已记录的变量作输入，按 `rng.below(10)` 的落点
/// 决定记成什么——0 是 detach，`upgrades` 为真时 1 是 retain（显式梯度终点），其余是普通
/// 算子；记录成功就进变量栈。`rng` 的取数顺序与调用方写开的循环逐次一致。
pub(crate) fn random_edges(
    tape: &mut Tape,
    vars: &mut Vec<Var>,
    rng: &mut Rng,
    rounds: u32,
    upgrades: bool,
) {
    for _ in 0..rounds {
        let width = 1 + rng.below(3) as usize;
        let inputs: Vec<Var> = (0..width)
            .map(|_| vars[rng.below(vars.len() as u32) as usize])
            .collect();
        let recorded = match rng.below(10) {
            0 => tape.record_with("detach", &inputs, Requirement::None),
            1 if upgrades => tape.record_with("retain", &inputs, Requirement::Grad),
            _ => tape.record("edge", &inputs),
        };
        if let Ok(var) = recorded {
            vars.push(var);
        }
    }
}

/// 从变量栈反向找第一个要求梯度的叶子作 loss，并 backward 出执行序；无 loss 或 backward
/// 失败返回 `None`（随机图测试据此跳过本轮）。
pub(crate) fn schedule_of(tape: &Tape, vars: &[Var]) -> Option<Schedule> {
    let loss = vars
        .iter()
        .rev()
        .find(|var| tape.requirement(**var).is_some_and(Requirement::requires))
        .copied()?;
    tape.backward(loss).ok()
}
