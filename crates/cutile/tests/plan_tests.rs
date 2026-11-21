use super::*;
use crate::prof::Profile;
use diagnostic::data::Value;

fn profile() -> Profile {
    Profile::parse(prof::embedded("cpu").expect("cpu 档案")).expect("档案")
}

fn op() -> OpDecl {
    let source = &op::builtin_named("gemv_i4").expect("内置");
    OpDecl::parse(source.declaration(), source.template()).expect("声明")
}

fn path(mma_share: f64) -> Path {
    Path {
        variant: "pipe2".to_string(),
        index: 0,
        schedule: crate::sch::Schedule::parse("pipe<2>").expect("调度"),
        mma_share,
    }
}

/// 降级链的最后一个变体：`direct` 没有流水，掩盖账必然超出阈值。
fn fallback(mma_share: f64) -> Path {
    Path {
        variant: "direct".to_string(),
        index: 1,
        schedule: crate::sch::Schedule::Direct,
        mma_share,
    }
}

/// 与算子无关的代价：`plan` 只消费这个数对，不认识任何流量公式。
fn cost() -> Cost {
    Cost {
        flops: 2.0 * 4096.0 * 4096.0,
        read: 8 * 1024 * 1024,
        write: 0,
    }
}

#[test]
fn roofline_is_the_bottleneck_plus_unmasked_part() {
    let profile = profile();
    let cost = cost();
    // `mma_share = 1` 时未掩盖项为 0：剩下的只有两个下界里的大者。
    let masked = Masking::of(1.0, &crate::sch::Schedule::Direct);
    let compute_only = predict(&profile, cost.flops, 0, masked).expect("可预测");
    let memory_only = predict(&profile, 0.0, cost.bytes(), masked).expect("可预测");
    let both = predict(&profile, cost.flops, cost.bytes(), masked).expect("可预测");
    assert!((both - compute_only.max(memory_only)).abs() < 1e-18);
}

#[test]
fn select_takes_the_declared_tile() {
    let plan = select(&op(), &profile(), &path(0.62), cost()).expect("可选择");
    assert_eq!(plan.variant, "pipe2");
    assert_eq!(plan.tile.summary(), "GMAX=32");
    assert!(plan.predicted_s > 0.0);
    assert_eq!(plan.flops, cost().flops);
    assert_eq!(plan.bytes, cost().bytes());
    assert!(plan.masking.within_limit(), "首选变体应当达标");
}

#[test]
fn the_plan_line_reports_the_judgement_it_used() {
    // 计划行里的百分比与阈值结论必须是判定的那两个数，而不是另算一份。
    let plan = select(&op(), &profile(), &path(0.62), cost()).expect("可选择");
    let line = plan.to_string();
    assert!(line.contains("未掩盖 19%（阈值 20%，达标）"), "{line}");
}

#[test]
fn a_falling_back_variant_is_reported_not_rejected() {
    // 兜底变体的掩盖账超阈值是事实，不是错误：`select` 照旧出计划，行里写明它是降级兜底。
    let plan = select(&op(), &profile(), &fallback(0.35), cost()).expect("兜底也应有计划");
    assert_eq!(plan.variant, "direct");
    assert!(!plan.masking.within_limit(), "{}", plan.masking);
    assert!(
        plan.to_string()
            .contains("未掩盖 65%（阈值 20%，未达标：降级兜底）"),
        "{plan}"
    );
}

#[test]
fn declaration_order_decides_the_tile_not_its_size() {
    // 轴取值顺序即优先级：把 64 写在前面就取 64，即使它更大。
    let source = &op::builtin_named("gemv_i4").expect("内置");
    let mut reordered = source.declaration().clone();
    reordered.set(
        "axes.GMAX",
        Value::array(vec![Value::int(64), Value::int(32)]),
    );
    assert_ne!(&reordered, source.declaration(), "声明里的轴写法变了");
    let op = OpDecl::parse(&reordered, source.template()).expect("声明");
    let plan = select(&op, &profile(), &path(0.62), cost()).expect("可选择");
    assert_eq!(plan.tile.summary(), "GMAX=64");
}

#[test]
fn prediction_is_the_same_for_every_tile() {
    // 模型里没有可比较的量：两个候选的预测必须相同，否则说明有人给 predict 编了常数。
    let source = &op::builtin_named("gemv_i4").expect("内置");
    let mut swapped = source.declaration().clone();
    swapped.set(
        "axes.GMAX",
        Value::array(vec![Value::int(32), Value::int(64)]),
    );
    let wide = OpDecl::parse(&swapped, source.template()).expect("声明");
    let narrow = select(&op(), &profile(), &path(0.62), cost()).expect("可选择");
    let wide = select(&wide, &profile(), &path(0.62), cost()).expect("可选择");
    assert_eq!(narrow.predicted_s, wide.predicted_s);
    assert_eq!(narrow.bytes, wide.bytes);
}

#[test]
fn profile_without_peaks_is_an_error() {
    let mut profile = profile();
    profile.peak = crate::prof::Peak::default();
    let error = select(&op(), &profile, &path(0.62), cost()).unwrap_err();
    assert_eq!(error.code(), Code::Unsupported);
    assert!(error.to_string().contains("roofline"), "{error}");
}

#[test]
fn the_window_radius_is_solved_from_the_budget() {
    assert_eq!(window_radius(1), 1);
    assert_eq!(window_radius(4), 2);
    assert_eq!(window_radius(10), 4, "√10 = 3.16 → 向上取整 4");
}

#[test]
fn pruning_caps_at_the_budget_and_keeps_a_composable_pair() {
    let parse = |name: &str| {
        let source = op::builtin_named(name).expect("内置");
        OpDecl::parse(source.declaration(), source.template()).expect("声明")
    };
    let gemv = parse("gemv_f32");
    let gelu = parse("gelu_f32");
    let ops = [gemv, gelu];
    let accepted = fusions(&ops, 2);
    assert_eq!(accepted.len(), 2, "预算 2 就只接受 2 条");
    // 窗口 `ops[lo..hi]` 必然含生产者自己：自融合（`gemv_f32` 的 `y` 写回自己的 `w`/`x`）在槽位
    // 上看起来合法，只能由 `compose` 排除。断言"没有自融合"才真正锁住剪枝质量。
    assert!(
        accepted.iter().all(|fusion| !fusion.is_self()),
        "自融合是循环携带依赖，不能进候选表：{accepted:?}"
    );
    assert!(
        accepted
            .iter()
            .any(|fusion| fusion.producer == "gemv_f32" && fusion.consumer == "gelu_f32"),
        "可拼的生产者→消费者对必须留下"
    );
}
