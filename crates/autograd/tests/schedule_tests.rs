use std::collections::{BTreeMap, BTreeSet};

use diagnostic::Code;
use testkit::Rng;

use super::*;

#[test]
fn insertion_order_is_the_reverse_record_order() {
    let (tape, _, y, z) = crate::testutil::pair();
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
    let (_, left, right, joined, loss, schedule) = crate::testutil::diamond();
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
fn planning_into_an_existing_schedule_reuses_its_bufs() {
    let (mut tape, _, left, right) = crate::testutil::fork();
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
        .map(|(idx, step)| (step.output().id(), idx))
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
    let rounds = rng.below(20);
    crate::testutil::random_edges(&mut tape, &mut vars, rng, rounds, true);
    let loss = *vars
        .iter()
        .rev()
        .find(|var| tape.requirement(**var).is_some_and(Requirement::requires))?;
    Some((tape, loss))
}

#[test]
fn counts_separate_roles_and_graph_edges() {
    let mut tape = Tape::new();
    let w = tape.parameter();
    let c = tape.leaf();
    let hidden = tape.record("relu", &[w]).expect("记录");
    let _unused = tape.record("relu", &[c]).expect("记录");
    let loss = tape.record("add", &[hidden]).expect("记录");

    let summary = Summary::of(&tape);
    assert_eq!(summary.vars(), 5);
    assert_eq!(summary.nodes(), 3);
    assert_eq!(summary.edges(), 3);
    assert_eq!(summary.parameters(), 1);
    assert_eq!(summary.constants(), 1);
    assert_eq!(summary.activations(), 3);
    assert_eq!(summary.max_depth(), 2);
    assert_eq!(summary.steps(), 0);

    let schedule = tape.backward(loss).expect("在图上");
    let planned = Summary::for_schedule(&tape, &schedule);
    assert_eq!(planned.steps(), 2, "loss 可达 hidden 与 loss 两步");
    assert_eq!(planned.endpoints(), 1);
    assert!(planned.reachable() >= planned.endpoints());
    assert_eq!(
        planned.dead(),
        summary.vars() as usize - planned.reachable()
    );
}
#[test]
fn a_diamond_waits_for_every_contribution() {
    let (x, left, right, joined, _, schedule) = crate::testutil::diamond();
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
    let (mut tape, x, _, _, schedule) = crate::testutil::linear();
    let steps = schedule.steps();
    let mut account = schedule.account();
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

/// 账本按计划重建：结算进度清零，扇出重新数一遍。
#[test]
fn rebuilding_starts_the_ledger_over() {
    let (_, _, y, _, schedule) = crate::testutil::linear();
    let mut account = Account::new();
    account.rebuild(&schedule);
    assert_eq!(account.settled(), 0);
    assert_eq!(account.pending(y), 1);
    let step = &schedule.steps()[0];
    account.settle(step).expect("loss");
    assert_eq!(account.settled(), 1);
    account.rebuild(&schedule);
    assert_eq!(account.settled(), 0, "重建丢弃进度");
    assert_eq!(account.pending(y), 1, "扇出回到初始值");
    assert!(!account.finished());
}

/// `settle_into` 与 `settle` 结果一致，复用调用方的缓冲。
#[test]
fn settling_into_a_reused_buf_reports_the_same_ready_vars() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let y = tape.parameter();
    let joined = tape.record("add", &[x, y]).expect("记录");
    let loss = tape.record("reduce_sum", &[joined]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    let mut account = schedule.account();
    let mut ready = Vec::with_capacity(8);
    let capacity = ready.capacity();
    account
        .settle_into(&schedule.steps()[0], &mut ready)
        .expect("loss");
    assert_eq!(ready, [joined]);
    account
        .settle_into(&schedule.steps()[1], &mut ready)
        .expect("joined");
    assert_eq!(ready, [x, y], "先清空再写，不残留上一步的就绪变量");
    assert_eq!(ready.capacity(), capacity, "缓冲没有重新分配");
    assert!(account.finished());
}

/// 一个账本只管一份调度：空账本拒绝一切，`finished` 也只是平凡真。
#[test]
fn an_empty_account_accepts_nothing() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let recorded = tape.record("relu", &[x]).expect("记录");
    let node = tape.node(recorded).expect("节点");
    let mut account = Account::new();
    assert!(account.finished());
    assert_eq!(
        account.settle(node).expect_err("没有调度").code(),
        Code::Invalid
    );
    assert!(
        account
            .settle(node)
            .expect_err("没有调度")
            .to_string()
            .contains("不是本次调度")
    );
}

/// 每个输入都有扇出，图上变量的计数与独立统计一致；不进图的变量不收账。
#[test]
fn fan_out_matches_an_independent_count() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let constant = tape.leaf();
    let a = tape.record("add", &[x, constant]).expect("记录");
    let b = tape.record("mul", &[a, a]).expect("记录");
    let loss = tape.record("reduce_sum", &[b, constant]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    let account = schedule.account();
    assert_eq!(account.pending(loss), 0, "loss 自己不是任何步骤的输入");
    assert_eq!(account.pending(b), 1);
    assert_eq!(account.pending(a), 2, "mul 里出现两次");
    assert_eq!(account.pending(x), 1);
    assert_eq!(account.pending(constant), 0, "常量不进图");
    assert_eq!(
        tape.requirement(constant),
        Some(Requirement::None),
        "常量确实在图外"
    );
    assert!(!account.ready(constant), "图外变量永远不会'就绪'");
}
