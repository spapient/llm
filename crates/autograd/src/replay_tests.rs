use diagnostic::{Code, Error};

use super::*;
use crate::{Budget, Choice, Requirement};
use testkit::Rng;

/// 标量表：值的真值 + 当前活着的值 + 峰值（逐位对拍、读已释放值即报错、活着的字节数）。
struct Table {
    truth: Vec<f32>,
    bytes: Vec<u64>,
    values: Vec<Option<f32>>,
    live: u64,
    peak: u64,
    reads_of_dead: usize,
}

impl Table {
    /// 前向刚结束时的状态：叶子常驻，计划保留的激活都还在（叶子不计入 `live`）。
    fn new(tape: &Tape, plan: &Plan, truth: &[f32], bytes: &[u64]) -> Self {
        let mut values = vec![None; truth.len()];
        let mut live = 0u64;
        for id in 0..tape.vars() {
            let var = tape.var(id).expect("编号有效");
            let leaf = tape.node(var).is_none();
            values[id as usize] = (leaf || plan.keep(var)).then_some(truth[id as usize]);
            if plan.keep(var) {
                live += bytes[id as usize];
            }
        }
        Self {
            truth: truth.to_vec(),
            bytes: bytes.to_vec(),
            values,
            live,
            peak: live,
            reads_of_dead: 0,
        }
    }

    fn value(&self, var: Var) -> Option<f32> {
        self.values.get(var.id() as usize).copied().flatten()
    }
}

impl Store for Table {
    fn compute(&mut self, op: &str, inputs: &[Var], output: Var) -> Result<()> {
        let mut args = [0.0f32; 4];
        if inputs.len() > args.len() {
            return Err(Error::new(
                Code::Unsupported,
                format!("测试引擎最多支持四个输入，实际 {}", inputs.len()),
            ));
        }
        for (slot, input) in inputs.iter().enumerate() {
            let Some(value) = self.value(*input) else {
                self.reads_of_dead += 1;
                return Err(Error::new(
                    Code::Internal,
                    format!("重算 {} 时读到已释放的变量 {}", output.id(), input.id()),
                ));
            };
            args[slot] = value;
        }
        let value = eval(op, &args[..inputs.len()])?;
        let expected = self.truth[output.id() as usize];
        if value != expected {
            return Err(Error::new(
                Code::Internal,
                format!("重算 {} 得到 {value}，前向是 {expected}", output.id()),
            ));
        }
        self.values[output.id() as usize] = Some(value);
        self.live += self.bytes[output.id() as usize];
        self.peak = self.peak.max(self.live);
        Ok(())
    }

    fn free(&mut self, var: Var) {
        if self.values[var.id() as usize].take().is_some() {
            self.live -= self.bytes[var.id() as usize];
        }
    }
}

/// 反向那一步：读全部输入，读不到就当成"读已释放"。
fn consume(table: &mut Table, step: &Node) -> Result<()> {
    for input in step.inputs() {
        if table.value(*input).is_none() {
            table.reads_of_dead += 1;
            return Err(Error::new(
                Code::Internal,
                format!(
                    "第 {} 步读到已释放的变量 {}",
                    step.output().id(),
                    input.id()
                ),
            ));
        }
    }
    Ok(())
}

/// 测试用的标量语义；与 `op` 里的真算子无关，只负责让数值随位置变化。
fn eval(op: &str, args: &[f32]) -> Result<f32> {
    match (op, args) {
        ("add", [a, b]) => Ok(a + b),
        ("relu", [a]) => Ok(a.max(0.0)),
        ("reduce_sum", [a]) => Ok(*a),
        other => Err(Error::new(
            Code::Unsupported,
            format!("测试引擎没有算子 {other:?}"),
        )),
    }
}

/// 前向真值：叶子取种子，节点按前向顺序算一遍。
fn forward(tape: &Tape, seeds: &[f32]) -> Vec<f32> {
    let mut values = vec![0.0f32; tape.vars() as usize];
    values[..seeds.len()].copy_from_slice(seeds);
    for node in tape.nodes() {
        let mut args = [0.0f32; 4];
        for (slot, input) in node.inputs().iter().enumerate() {
            args[slot] = values[input.id() as usize];
        }
        values[node.output().id() as usize] =
            eval(node.op(), &args[..node.inputs().len()]).expect("测试图只用已知算子");
    }
    values
}

/// 一条 `add(current, one)` 的链：每一步的值都不同，重算错了会被逐位对拍抓住。
fn chain(length: usize) -> (Tape, Var) {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let one = tape.leaf();
    let mut current = x;
    for _ in 0..length {
        current = tape.record("add", &[current, one]).expect("记录");
    }
    let loss = tape.record("reduce_sum", &[current]).expect("记录");
    (tape, loss)
}

fn run_once(tape: &Tape, schedule: &Schedule, plan: &Plan, bytes: &[u64]) -> (u64, Table) {
    let seeds = [1.0, 1.0];
    let truth = forward(tape, &seeds);
    let mut table = Table::new(tape, plan, &truth, bytes);
    let replay = Replay::of(tape, schedule, plan).expect("可执行");
    assert_eq!(replay.expected(), plan.replayed(), "承诺值与计划同源");
    let computed = replay
        .run(tape, schedule, &mut table, consume)
        .expect("重放");
    (computed, table)
}

#[test]
fn a_tight_budget_replays_exactly_what_the_plan_promised() {
    let (tape, loss) = chain(16);
    let schedule = tape.backward(loss).expect("在图上");
    let bytes = vec![8u64; tape.vars() as usize];
    let plan = Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(96)).expect("可分段");
    assert_eq!(plan.choice(), Choice::Recompute);

    let (computed, table) = run_once(&tape, &schedule, &plan, &bytes);
    assert_eq!(computed, plan.replayed(), "链上不共享前缀，实际就是承诺值");
    assert!(
        table.peak <= plan.peak(),
        "实测峰值 {} 超过计划 {}",
        table.peak,
        plan.peak()
    );
    assert!(table.peak <= plan.budget().bytes(), "峰值超预算");
    assert_eq!(table.live, 0, "执行完不该留下任何激活");
    assert_eq!(table.reads_of_dead, 0);
}

#[test]
fn replay_counts_listed_instructions() {
    let (tape, loss) = chain(16);
    let schedule = tape.backward(loss).expect("在图上");
    let bytes = vec![8u64; tape.vars() as usize];
    let plan = Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(96)).expect("可分段");
    let replay = Replay::of(&tape, &schedule, &plan).expect("可执行");
    assert_eq!(replay.compute_count(), plan.replayed() as usize);
    assert!(
        replay.free_count() > 0,
        "到寿激活逐块交还，至少一条释放指令"
    );

    let (tape, loss) = chain(4);
    let schedule = tape.backward(loss).expect("在图上");
    let bytes = vec![8u64; tape.vars() as usize];
    let plan = Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(1 << 20)).expect("放得下");
    let replay = Replay::of(&tape, &schedule, &plan).expect("可执行");
    assert_eq!(replay.compute_count(), 0, "全留路径不编入重算");
    assert!(replay.free_count() > 0, "不重放也要按时编入释放");
}

#[test]
fn a_plan_that_fits_replays_nothing_but_still_frees_on_time() {
    let (tape, loss) = chain(4);
    let schedule = tape.backward(loss).expect("在图上");
    let bytes = vec![8u64; tape.vars() as usize];
    let plan = Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(1 << 20)).expect("放得下");
    assert_eq!(plan.choice(), Choice::KeepAll);

    let (computed, table) = run_once(&tape, &schedule, &plan, &bytes);
    assert_eq!(computed, 0, "装得下就不重放");
    assert_eq!(
        table.live, 0,
        "不重放也要按释放点还缓冲，否则显存收益就没了"
    );
    assert_eq!(table.reads_of_dead, 0);
}

#[test]
fn a_checkpoint_lives_until_its_last_need() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let one = tape.leaf();
    let shared = tape.record("relu", &[x]).expect("记录");
    let left = tape.record("add", &[shared, one]).expect("记录");
    let right = tape.record("add", &[shared, one]).expect("记录");
    let joined = tape.record("add", &[left, right]).expect("记录");
    let loss = tape.record("reduce_sum", &[joined]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    let bytes = vec![8u64; tape.vars() as usize];
    let plan = Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(1 << 20)).expect("放得下");

    let truth = forward(&tape, &[1.0, 1.0]);
    let mut table = Table::new(&tape, &plan, &truth, &bytes);
    let replay = Replay::of(&tape, &schedule, &plan).expect("可执行");
    let mut live = Vec::new();
    replay
        .run(&tape, &schedule, &mut table, |table, step| {
            live.push(table.value(shared).is_some());
            consume(table, step)
        })
        .expect("重放");

    assert_eq!(
        live,
        [true, true, true, true, false],
        "共享的激活活到较晚的那一支，之后立刻还回去"
    );
    assert_eq!(plan.release(shared), Some(3));
}

/// 一个检查点可能既是深层节点的直接输入、又是更浅的被丢弃值的祖先：后者排得更晚。
#[test]
fn a_checkpoint_that_anchors_a_later_replay_is_not_freed_early() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let one = tape.leaf();
    let v = tape.record("relu", &[x]).expect("记录");
    let w = tape.record("add", &[v, one]).expect("记录");
    let c1 = tape.record("add", &[w, one]).expect("记录");
    let c2 = tape.record("add", &[c1, one]).expect("记录");
    let deep = tape.record("add", &[v, c2]).expect("记录");
    let loss = tape.record("reduce_sum", &[deep]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    let bytes = vec![8u64; tape.vars() as usize];

    let mut saw_drop = false;
    for budget in [8u64, 16, 24, 32, 40, 56] {
        let Ok(plan) = Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(budget)) else {
            continue;
        };
        if !plan.keep(w) {
            saw_drop = true;
        }
        let (computed, table) = run_once(&tape, &schedule, &plan, &bytes);
        assert_eq!(table.reads_of_dead, 0, "预算 {budget}：释放早了");
        assert!(table.peak <= plan.peak(), "预算 {budget}：峰值超计划");
        assert_eq!(
            computed,
            plan.replayed(),
            "预算 {budget}：重算步数与承诺不符"
        );
    }
    assert!(saw_drop, "这组预算里必须有一个真丢掉了 w，否则用例没有意义");
}

#[test]
fn a_deep_chain_neither_recurses_nor_blows_the_budget() {
    let (tape, loss) = chain(20_000);
    let schedule = tape.backward(loss).expect("在图上");
    let bytes = vec![8u64; tape.vars() as usize];
    let plan = Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(16_000)).expect("可分段");
    let (computed, table) = run_once(&tape, &schedule, &plan, &bytes);
    assert!(computed > 0);
    assert!(computed <= plan.replayed());
    assert!(
        table.peak <= plan.peak(),
        "峰值 {} 超过计划 {}",
        table.peak,
        plan.peak()
    );
    assert_eq!(table.live, 0);
}

#[test]
fn a_sequence_from_another_schedule_is_rejected() {
    let (tape, loss) = chain(3);
    let schedule = tape.backward(loss).expect("在图上");
    let plan = Plan::of(
        &tape,
        &schedule,
        vec![8u64; tape.vars() as usize],
        Budget::new(1 << 20),
    )
    .expect("放得下");
    let replay = Replay::of(&tape, &schedule, &plan).expect("可执行");

    let mut other = Tape::new();
    let y = other.parameter();
    let long = other.record("relu", &[y]).expect("记录");
    let other_loss = other.record("reduce_sum", &[long]).expect("记录");
    let other_schedule = other.backward(other_loss).expect("在图上");
    let mut table = Table::new(&other, &plan, &[1.0, 1.0, 1.0], &[8, 8, 8]);
    let error = replay
        .run(&other, &other_schedule, &mut table, consume)
        .expect_err("步骤数不同");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("不是同一张图"), "{error}");
}

/// `Store` 的失败原样上抛：重放不能把"算不出来"悄悄当成"算完了"。
#[test]
fn a_store_failure_stops_the_replay() {
    struct Refuse;
    impl Store for Refuse {
        fn compute(&mut self, _: &str, _: &[Var], _: Var) -> Result<()> {
            Err(Error::new(Code::Unsupported, "设备拒绝"))
        }
        fn free(&mut self, _: Var) {}
    }

    let (tape, loss) = chain(16);
    let schedule = tape.backward(loss).expect("在图上");
    let bytes = vec![8u64; tape.vars() as usize];
    let plan = Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(96)).expect("可分段");
    let replay = Replay::of(&tape, &schedule, &plan).expect("可执行");
    let error = replay
        .run(&tape, &schedule, &mut Refuse, |_, _| Ok(()))
        .expect_err("重算失败");
    assert_eq!(error.code(), Code::Unsupported);
}

/// 随机图：释放时机、重算正确性、峰值与承诺值一起对拍。
#[test]
fn random_graphs_replay_without_reading_a_released_value() {
    let mut rng = Rng::new(0x1234_5678);
    let mut replayed_rounds = 0;
    for round in 0..200 {
        let mut tape = Tape::new();
        let mut vars: Vec<Var> = (0..1 + rng.below(3)).map(|_| tape.parameter()).collect();
        vars.extend((0..rng.below(2)).map(|_| tape.leaf()));
        let leaves = vars.len();
        for _ in 0..1 + rng.below(18) {
            let width = 1 + rng.below(2) as usize;
            let inputs: Vec<Var> = (0..width)
                .map(|_| vars[rng.below(vars.len() as u32) as usize])
                .collect();
            let op = match width {
                1 => "relu",
                _ => "add",
            };
            let recorded = match rng.below(10) {
                0 => tape.record_with(op, &inputs, Requirement::None),
                _ => tape.record(op, &inputs),
            };
            if let Ok(var) = recorded {
                vars.push(var);
            }
        }
        let Some(loss) = vars
            .iter()
            .rev()
            .find(|var| tape.requirement(**var).is_some_and(Requirement::requires))
            .copied()
        else {
            continue;
        };
        let Ok(schedule) = tape.backward(loss) else {
            continue;
        };
        let seeds: Vec<f32> = (0..leaves).map(|_| 1.0 + rng.below(4) as f32).collect();
        let truth = forward(&tape, &seeds);
        let bytes: Vec<u64> = (0..tape.vars())
            .map(|_| 8 * u64::from(rng.below(8)))
            .collect();
        let budget = Budget::new(8 * u64::from(rng.below(64)));
        let Ok(plan) = Plan::of(&tape, &schedule, bytes.as_slice(), budget) else {
            continue;
        };
        let mut table = Table::new(&tape, &plan, &truth, &bytes);
        let replay = Replay::of(&tape, &schedule, &plan).expect("可执行");
        let computed = replay
            .run(&tape, &schedule, &mut table, consume)
            .unwrap_or_else(|error| panic!("第 {round} 轮：{error}"));
        assert_eq!(table.reads_of_dead, 0, "第 {round} 轮：读到已释放的值");
        assert!(computed <= plan.replayed(), "第 {round} 轮：重算超过承诺");
        assert!(
            table.peak <= plan.peak(),
            "第 {round} 轮：峰值 {} 超过计划 {}",
            table.peak,
            plan.peak()
        );
        assert_eq!(table.live, 0, "第 {round} 轮：执行完还留着激活");
        replayed_rounds += u32::from(computed > 0);
    }
    assert!(
        replayed_rounds > 0,
        "随机图里至少要有一轮真的重算了，否则用例是空的"
    );
}
