use super::*;
use crate::prof::Profile;

fn profile() -> Profile {
    Profile::parse(prof::embedded("cpu").expect("cpu 档案")).expect("档案")
}

fn op() -> OpDecl {
    prof::catalogue::named("gemv_i4")
        .expect("内置")
        .to_decl()
        .expect("声明")
}

fn path(mma_share: f64) -> Path {
    Path {
        variant: "pipe2".to_string(),
        idx: 0,
        schedule: crate::sch::Schedule::parse("pipe<2>").expect("调度"),
        mma_share,
    }
}

/// 降级链的最后一个变体：`direct` 没有流水，掩盖账必然超出阈值。
fn fallback(mma_share: f64) -> Path {
    Path {
        variant: "direct".to_string(),
        idx: 1,
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
    let mut op = op();
    op.axes[0].values = vec![64, 32];
    let plan = select(&op, &profile(), &path(0.62), cost()).expect("可选择");
    assert_eq!(plan.tile.summary(), "GMAX=64");
}

#[test]
fn prediction_is_the_same_for_every_tile() {
    // 模型里没有可比较的量：两个候选的预测必须相同，否则说明有人给 predict 编了常数。
    let mut wide = op();
    wide.axes[0].values = vec![32, 64];
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
        prof::catalogue::named(name)
            .expect("内置")
            .to_decl()
            .expect("声明")
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

/// 内置声明表：`fusions` 与 `ranking` 的输入就是它。
/// 语义表：注册表在 `nn`——只有它持有内核类型，`Model` 才装得进去。
fn models() -> &'static [prof::catalogue::Registration] {
    &nn::registry::REGISTRY
}

fn decls() -> Vec<OpDecl> {
    prof::catalogue::all()
        .iter()
        .map(|source| source.to_decl().expect("声明自洽"))
        .collect()
}

/// 一条真实候选：`adaptive_pool_f32` 的 `y` 接 `attention_f32` 的输入（生产形状对得上）。
fn pool_into_attention() -> Fusion {
    let parse = |name: &str| {
        prof::catalogue::named(name)
            .expect("内置")
            .to_decl()
            .expect("声明")
    };
    compose(&parse("adaptive_pool_f32"), &parse("attention_f32")).expect("能拼")
}

#[test]
fn a_fusion_never_slows_down() {
    let profile = profile();
    let ranked = ranking(&profile, &decls(), 12, models());
    let mut scored = 0;
    for (fusion, verdict) in &ranked {
        if let Verdict::Scored(payoff) = verdict {
            scored += 1;
            // `max` 次可加：融合省的只有流量，算力不变，所以预测只会更小。
            assert!(
                payoff.speedup() >= 1.0 && payoff.fused_s <= payoff.unfused_s,
                "{} → {} 越融越慢：{:?}",
                fusion.producer,
                fusion.consumer,
                payoff
            );
            assert!(payoff.saved_bytes > 0, "融合必须省下中间量的往返");
            assert!(payoff.bound() == "compute" || payoff.bound() == "memory");
        }
    }
    assert!(scored > 0, "内置注册表上至少有一条候选打得出分");
}

#[test]
fn the_payoff_is_the_intermediate_round_trip() {
    let profile = profile();
    let fusion = pool_into_attention();
    let Verdict::Scored(payoff) = score(&profile, &fusion, models()) else {
        panic!(
            "这一对应当打得出分：{}",
            score(&profile, &fusion, models()).as_str()
        );
    };
    // 中间量=生产者唯一输出的字节数，独立算一遍：省的正是它的写 + 读。
    let producer = nn::registry::model(&fusion.producer).expect("语义投影");
    let (m, n) = producer.shape;
    let shape_ctx = (producer.shape_ctx)(m, n).expect("生产形状自洽");
    let cost = (producer.cost)(&shape_ctx).expect("代价");
    assert_eq!(payoff.saved_bytes, 2 * cost.write);
    assert!(payoff.saved_bytes <= cost.bytes() * 2, "省的不能超过往返");
}

#[test]
fn the_payoff_agrees_with_predict() {
    let profile = profile();
    let fusion = pool_into_attention();
    let Verdict::Scored(payoff) = score(&profile, &fusion, models()) else {
        panic!("这一对应当打得出分");
    };
    // 一台机器两个视角：不融合就是两次 `predict`（掩盖项为 0 时等价于裸 roofline）。
    let bare = Masking {
        mma_share: 1.0,
        stages: 1,
        unmasked: 0.0,
    };
    let side = |name: &str| {
        let model = nn::registry::model(name).expect("语义投影");
        let (m, n) = model.shape;
        let ctx = (model.shape_ctx)(m, n).expect("形状");
        let cost = (model.cost)(&ctx).expect("代价");
        predict(&profile, cost.flops, cost.bytes(), bare).expect("可预测")
    };
    let unfused = side(&fusion.producer) + side(&fusion.consumer);
    assert!((payoff.unfused_s - unfused).abs() < 1e-18, "{payoff:?}");
}

#[test]
fn scoring_refuses_without_peaks() {
    let mut profile = profile();
    profile.peak = crate::prof::Peak::default();
    // 峰值缺席是"无从判定"，不是"加速 1.0×"：不许猜一个数顶上。
    assert!(matches!(
        score(&profile, &pool_into_attention(), models()),
        Verdict::Unscorable(_)
    ));
}

#[test]
fn a_candidate_without_a_semantic_model_is_unscorable() {
    let mut fusion = pool_into_attention();
    fusion.producer = "没有这个算子".to_string();
    assert!(matches!(
        score(&profile(), &fusion, models()),
        Verdict::Unscorable(_)
    ));
}

#[test]
fn a_multi_output_seam_cannot_be_attributed() {
    let mut fusion = pool_into_attention();
    let mut second = fusion.seams[0].clone();
    second.from = "另一条输出".to_string();
    fusion.seams.push(second);
    // 代价表的 `write` 是全部输出之和：接缝占两条时中间量无法从它归因。
    assert!(matches!(
        score(&profile(), &fusion, models()),
        Verdict::Unscorable(_)
    ));
}

#[test]
fn the_ranking_covers_every_candidate_and_orders_it() {
    let profile = profile();
    let ranked = ranking(&profile, &decls(), 12, models());
    assert_eq!(
        ranked.len(),
        fusions(&decls(), 12).len(),
        "每条候选必须落到某个判决上：不许悄悄丢掉"
    );
    let (mut ranked_slowdown, mut last_rank) = (f64::INFINITY, 0u8);
    for (fusion, verdict) in &ranked {
        let rank = match verdict {
            Verdict::Scored(payoff) => {
                assert!(
                    payoff.speedup() <= ranked_slowdown,
                    "{} → {} 破坏了加速比降序",
                    fusion.producer,
                    fusion.consumer
                );
                ranked_slowdown = payoff.speedup();
                0
            }
            Verdict::Unscorable(_) => 1,
            Verdict::Rejected(reason) => {
                assert!(!reason.is_empty(), "拒绝必须带原因");
                2
            }
        };
        assert!(
            rank >= last_rank,
            "判决分组必须是 scored → unscorable → rejected"
        );
        last_rank = rank;
    }
}
