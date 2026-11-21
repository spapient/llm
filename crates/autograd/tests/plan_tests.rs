use crate::{
    Budget, Choice, Config, Footprint, MemoryPolicy, Ordering, Plan, Requirement, Schedule, Sizes,
    Tape, Var,
};
use diagnostic::Code;
use tensor::{DType, Shape};
use testkit::Rng;

/// 从 `current` 再往下接 `length` 个 `relu`，以 `reduce_sum` 收尾并 backward。
fn extend(tape: &mut Tape, current: &mut Var, length: usize) -> Schedule {
    for _ in 0..length {
        *current = tape.record("relu", &[*current]).expect("记录");
    }
    let loss = tape.record("reduce_sum", &[*current]).expect("记录");
    tape.backward(loss).expect("在图上")
}

/// 一条链：`length` 个 `relu` 加一个 `reduce_sum`，反向顺序与深度都已知。
fn chain(length: usize) -> (Tape, Schedule) {
    let mut tape = Tape::new();
    let mut current = tape.parameter();
    let schedule = extend(&mut tape, &mut current, length);
    (tape, schedule)
}

fn uniform(tape: &Tape, bytes: u64) -> Vec<u64> {
    vec![bytes; tape.vars() as usize]
}

#[test]
fn keep_all_when_the_budget_fits_everything() {
    let (tape, schedule) = chain(4);
    let plan = Plan::of(&tape, &schedule, uniform(&tape, 8), Budget::new(1 << 20)).expect("放得下");
    assert_eq!(plan.choice(), Choice::KeepAll);
    assert_eq!(plan.replayed(), 0);
    // 四个 relu 的输出被消费，loss 的输出没人消费：峰值就是这四块。
    assert_eq!(plan.saved().len(), 4);
    assert_eq!(plan.retained(), 32);
    assert_eq!(plan.peak(), 32);
    assert!(plan.peak() >= plan.floor(), "峰值不能低于下界");
}

#[test]
fn forced_keep_all_keeps_everything_even_under_pressure() {
    let (tape, schedule) = chain(16);
    let plan = Plan::of_policy(
        &tape,
        &schedule,
        uniform(&tape, 16),
        Budget::new(200),
        MemoryPolicy::KeepAll,
    )
    .expect("全留");
    assert_eq!(plan.policy(), MemoryPolicy::KeepAll);
    assert_eq!(plan.choice(), Choice::KeepAll);
    assert_eq!(plan.replayed(), 0);
}

#[test]
fn forced_recompute_segments_even_with_room_to_spare() {
    let (tape, schedule) = chain(8);
    let plan = Plan::of_policy(
        &tape,
        &schedule,
        uniform(&tape, 16),
        Budget::new(1 << 20),
        MemoryPolicy::Recompute,
    )
    .expect("分段");
    assert_eq!(plan.policy(), MemoryPolicy::Recompute);
    assert_eq!(plan.choice(), Choice::Recompute);
    assert!(plan.replayed() > 0, "强制重放必须留下段内重算");
}

#[test]
fn of_config_forwards_the_configured_memory_policy() {
    let (tape, schedule) = chain(8);
    let config = Config::new(Ordering::Depth, MemoryPolicy::Recompute);
    let plan = Plan::of_config(
        &tape,
        &schedule,
        uniform(&tape, 16),
        Budget::new(1 << 20),
        &config,
    )
    .expect("分段");
    assert_eq!(plan.policy(), MemoryPolicy::Recompute);
    assert_eq!(plan.choice(), Choice::Recompute);

    let auto = Config::default();
    let plan = Plan::of_config(
        &tape,
        &schedule,
        uniform(&tape, 16),
        Budget::new(1 << 20),
        &auto,
    )
    .expect("放得下");
    assert_eq!(plan.policy(), MemoryPolicy::Auto);
    assert_eq!(plan.choice(), Choice::KeepAll);
}

#[test]
fn a_budget_below_the_floor_is_unsupported() {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let q = tape.parameter();
    let r = tape.parameter();
    let a = tape.record("relu", &[p]).expect("记录");
    let b = tape.record("relu", &[q]).expect("记录");
    let c = tape.record("relu", &[r]).expect("记录");
    let joined = tape.record("add", &[a, b, c]).expect("记录");
    let loss = tape.record("reduce_sum", &[joined]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");

    let mut bytes = vec![0u64; tape.vars() as usize];
    for var in [a, b, c] {
        bytes[var.id() as usize] = 100;
    }
    let error =
        Plan::of(&tape, &schedule, bytes, Budget::new(200)).expect_err("三个输入要同时在场");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("300"), "{error}");
}

#[test]
fn a_tight_budget_buys_replay_instead_of_memory() {
    let (tape, schedule) = chain(16);
    let plan = Plan::of(&tape, &schedule, uniform(&tape, 16), Budget::new(200)).expect("可分段");
    assert_eq!(plan.choice(), Choice::Recompute);
    assert!(plan.peak() <= 200, "峰值 {} 超预算", plan.peak());
    assert!(plan.replayed() > 0, "省下的显存必须用重放换");
    assert!(
        plan.saved().windows(2).all(|pair| pair[0] < pair[1]),
        "检查点按前向顺序且不重复"
    );
    assert!(plan.saved().len() >= 2, "至少要留第一个激活和段界");
    assert!(plan.peak() >= plan.floor());
}

#[test]
fn release_points_follow_the_last_consumer_not_the_scope() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let shared = tape.record("relu", &[x]).expect("记录");
    let left = tape.record("mul", &[shared]).expect("记录");
    let right = tape.record("mul", &[shared]).expect("记录");
    let joined = tape.record("add", &[left, right]).expect("记录");
    let loss = tape.record("reduce_sum", &[joined]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    let plan = Plan::of(&tape, &schedule, uniform(&tape, 4), Budget::new(1 << 20)).expect("放得下");

    // 反向顺序是 loss(0)、joined(1)、right(2)、left(3)、shared(4)：共享的激活活到第 3 步。
    assert_eq!(
        schedule
            .steps()
            .iter()
            .map(|step| step.output())
            .collect::<Vec<_>>(),
        [loss, joined, right, left, shared]
    );
    assert_eq!(plan.release(shared), Some(3), "活到较晚的那一支");
    assert_eq!(plan.release(loss), None, "没人消费就不必保留");
    assert!(!plan.keep(loss));
    assert!(!plan.keep(x), "叶子是常驻的权重，不进激活预算");
}

#[test]
fn a_zero_budget_cannot_hold_anything() {
    let (tape, schedule) = chain(3);
    let error = Plan::of(&tape, &schedule, uniform(&tape, 8), Budget::new(0)).expect_err("放不下");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("预算 0"), "{error}");
}

#[test]
fn the_plan_is_deterministic() {
    let (tape, schedule) = chain(12);
    let first = Plan::of(&tape, &schedule, uniform(&tape, 16), Budget::new(160)).expect("可分段");
    let second = Plan::of(&tape, &schedule, uniform(&tape, 16), Budget::new(160)).expect("可分段");
    assert_eq!(first, second);
}

#[test]
fn the_size_table_turns_shapes_into_bytes() {
    let mut sizes = Sizes::new(1);
    assert!(!sizes.is_empty());
    sizes.set(Var::at(0), Shape::new([2, 3]), DType::F32);
    assert_eq!(sizes.bytes(Var::at(0)), 24);
    sizes.set(Var::at(0), Shape::new([4]), DType::U4);
    assert_eq!(sizes.bytes(Var::at(0)), 2, "4 位元素按位打包");
    sizes.set(Var::at(9), Shape::new([5]), DType::F16);
    assert_eq!(sizes.len(), 10, "跳跃登记会补齐中间的空位");
    assert_eq!(sizes.bytes(Var::at(8)), 0, "没登记的变量按 0 计");
}

#[test]
fn a_size_table_drives_the_plan() {
    let (tape, schedule) = chain(4);
    let mut sizes = Sizes::new(tape.vars());
    for step in schedule.steps() {
        sizes.set(step.output(), Shape::new([2, 2]), DType::F32);
    }
    let plan = Plan::of(&tape, &schedule, &sizes, Budget::new(1 << 20)).expect("放得下");
    assert_eq!(plan.retained(), 4 * 16, "四个 2×2 的 f32 激活");
}

#[test]
fn a_dead_branch_stays_out_of_the_plan() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let dead = tape
        .record_with("stop_gradient", &[x], Requirement::None)
        .expect("记录");
    let _hidden = tape.record("relu", &[dead]).expect("记录");
    let live = tape.record("relu", &[x]).expect("记录");
    let loss = tape.record("reduce_sum", &[live]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    let plan = Plan::of(&tape, &schedule, uniform(&tape, 8), Budget::new(1 << 20)).expect("放得下");
    assert!(
        !plan.keep(dead),
        "不进图的分支没有反向步骤，也就不进显存计划"
    );
    assert!(plan.keep(live));
    assert_eq!(plan.saved().len(), 1);
}

#[test]
fn a_detached_value_consumed_by_a_live_step_is_kept_and_never_dropped() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let frozen = tape
        .record_with("stop_gradient", &[x], Requirement::None)
        .expect("记录");
    let mut current = tape.record("relu", &[frozen, x]).expect("记录");
    let schedule = extend(&mut tape, &mut current, 8);

    // 10 个 8 字节的激活共 80 字节；预算 64 时必然要丢一些，但 frozen 无处可重放。
    let plan = Plan::of(&tape, &schedule, uniform(&tape, 8), Budget::new(64)).expect("可分段");
    assert_eq!(plan.choice(), Choice::Recompute);
    assert!(
        plan.keep(frozen),
        "detach 出去的值仍被活着的步骤消费，只能留"
    );
    assert!(plan.saved().contains(&frozen));
    assert!(plan.peak() <= 64, "峰值 {} 超预算", plan.peak());
}

#[test]
fn a_deep_chain_does_not_recurse_and_fits_a_tight_budget() {
    let (tape, schedule) = chain(20_000);
    let bytes = uniform(&tape, 1);
    let large = Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(1 << 20)).expect("全留");
    assert_eq!(large.choice(), Choice::KeepAll);

    let tight = Plan::of(&tape, &schedule, bytes.as_slice(), Budget::new(2_000)).expect("可分段");
    assert!(tight.peak() <= 2_000, "峰值 {} 超预算", tight.peak());
    assert!(tight.replayed() > 0);
}

#[test]
fn random_graphs_never_exceed_their_budget() {
    let mut rng = Rng::new(0x9E37_79B9);
    for round in 0..200 {
        let mut tape = Tape::new();
        let mut vars: Vec<Var> = (0..1 + rng.below(4)).map(|_| tape.parameter()).collect();
        let rounds = 1 + rng.below(24);
        crate::testutil::random_edges(&mut tape, &mut vars, &mut rng, rounds, false);
        let Some(schedule) = crate::testutil::schedule_of(&tape, &vars) else {
            continue;
        };
        let bytes: Vec<u64> = (0..tape.vars()).map(|_| u64::from(rng.below(64))).collect();
        let budget = Budget::new(u64::from(rng.below(512)));
        match Plan::of(&tape, &schedule, bytes.as_slice(), budget) {
            Ok(plan) => {
                assert!(
                    plan.peak() <= budget.bytes(),
                    "第 {round} 轮：峰值 {} 超预算 {}",
                    plan.peak(),
                    budget.bytes()
                );
                assert!(plan.peak() >= plan.floor(), "第 {round} 轮：峰值低于下界");
                for var in plan.saved() {
                    assert!(plan.keep(*var), "第 {round} 轮：保留集与位图不一致");
                    assert!(plan.release(*var).is_some(), "第 {round} 轮：保留却不释放");
                    assert!(
                        tape.node(*var).is_some(),
                        "第 {round} 轮：叶子（参数/输入）不该进激活预算"
                    );
                }
            }
            Err(error) => {
                assert_eq!(error.code(), Code::Invalid, "第 {round} 轮：{error}");
            }
        }
    }
}

/// 固定图形与预算下的计划摘要：`saved`/`release`/`replayed`/`peak` 逐值冻结。
fn summarize(case: &str, tape: &Tape, schedule: &Schedule, bytes: &[u64], budget: u64) -> String {
    let Ok(plan) = Plan::of(tape, schedule, bytes, Budget::new(budget)) else {
        return format!("{case}/{budget}: 放不下");
    };
    let saved: Vec<u32> = plan.saved().iter().map(|var| var.id()).collect();
    let release: Vec<Option<usize>> = plan.saved().iter().map(|var| plan.release(*var)).collect();
    format!(
        "{case}/{budget}: {:?} saved={saved:?} release={release:?} replayed={} peak={}",
        plan.choice(),
        plan.replayed(),
        plan.peak()
    )
}

/// 改造重构前的实测值：同一组图形与预算必须逐字相同。
const FROZEN: [&str; 3] = [
    "chain16/1048576: KeepAll saved=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] release=[Some(15), Some(14), Some(13), Some(12), Some(11), Some(10), Some(9), Some(8), Some(7), Some(6), Some(5), Some(4), Some(3), Some(2), Some(1), Some(0)] replayed=0 peak=256",
    "chain16/200: Recompute saved=[1, 3, 6, 8, 11, 14, 16] release=[Some(15), Some(13), Some(10), Some(8), Some(5), Some(2), Some(0)] replayed=12 peak=144",
    "chain16/0: 放不下",
];

#[test]
fn the_plan_numbers_are_frozen() {
    let mut lines = Vec::new();
    let (tape, schedule) = chain(16);
    for budget in [1 << 20, 200, 0] {
        let bytes = uniform(&tape, 16);
        lines.push(summarize("chain16", &tape, &schedule, &bytes, budget));
    }
    println!("{}", lines.join("\n"));
    assert_eq!(lines, FROZEN);
}
