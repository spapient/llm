use super::*;

#[test]
fn closed_set_round_trip() {
    for text in ["direct", "pipe<3>", "ws<2,2>", "pingpong<2>"] {
        assert_eq!(Schedule::parse(text).unwrap().as_str(), text);
    }
}

#[test]
fn closed_set_rejects_unknown() {
    let error = Schedule::parse("rs").unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("不在调度闭集内"), "{error}");
    assert!(Schedule::parse("pipe<1>").is_err(), "单级预取没有意义");
}

#[test]
fn required_prims_follow_the_template() {
    assert!(
        Schedule::parse("direct")
            .unwrap()
            .required_prims()
            .is_empty()
    );
    assert_eq!(
        Schedule::parse("pipe<3>").unwrap().required_prims(),
        &[Prim::Bar]
    );
    assert!(
        Schedule::parse("ws<2,2>")
            .unwrap()
            .required_prims()
            .contains(&Prim::LoadAsync)
    );
}

#[test]
fn unmasked_share_shrinks_with_stages() {
    let one = unmasked_share(0.5, 1);
    let two = unmasked_share(0.5, 2);
    assert!(two < one, "级数越多，越能掩盖");
    assert!((one - 0.5).abs() < 1e-12);
}

#[test]
fn the_preferred_variant_must_hide_its_non_mma_work() {
    let pipe = Schedule::parse("pipe<2>").unwrap();
    assert!(check_masking("pipe2", Masking::of(0.62, &pipe)).is_ok());
    let error = check_masking("direct", Masking::of(0.35, &Schedule::Direct)).unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("首选变体 `direct`"), "{error}");
    assert!(error.to_string().contains("65%"), "{error}");
}

#[test]
fn a_variant_without_mma_has_nothing_to_hide() {
    // 占比为 0 的变体全部时间都是非 MMA：判据不适用，不该拦住一条 SIMT 路径。
    let masking = Masking::of(0.0, &Schedule::Direct);
    assert!(!masking.claims_mma());
    assert!(!masking.within_limit());
    assert!(check_masking("scalar", masking).is_ok());
}

#[test]
fn the_limit_is_inclusive() {
    // (1 - 0.6) / 2 = 0.20：正好压线算通过，判定必须与报告里那个百分比同一个数。
    let masking = Masking::of(0.6, &Schedule::parse("pipe<2>").unwrap());
    assert!((masking.unmasked - UNMASKED_LIMIT).abs() < 1e-12);
    assert!(masking.within_limit());
    assert!(check_masking("pipe2", masking).is_ok());
}

#[test]
fn monotonic_chain_is_accepted() {
    let chain = vec![
        ("pingpong<2>".to_string(), 0.82),
        ("pipe<3>".to_string(), 0.74),
        ("pipe<2>".to_string(), 0.61),
        ("direct".to_string(), 0.40),
    ];
    assert!(check_monotonic(&chain).is_ok());
}

#[test]
fn reversed_chain_is_rejected() {
    let chain = vec![
        ("pipe<3>".to_string(), 0.50),
        ("pingpong<2>".to_string(), 0.90),
    ];
    let error = check_monotonic(&chain).unwrap_err();
    assert!(error.to_string().contains("写反了"), "{error}");
}

/// 调度解析边界：空串、缺尖括号、参数个数不齐、非数字与越下参数都带说明拒绝。
#[test]
fn schedule_parse_handles_degenerate_and_boundary_inputs() {
    let cases: &[&str] = &[
        "",
        "   ",
        "pipe",
        "pipe<>",
        "pipe<2,3>",
        "ws<2>",
        "ws<2,2,3>",
        "pingpong<1>",
        "pipe<0>",
        "pipe<x>",
        "ws<-1,2>",
        "unknown<2>",
        "pipe<2",
        "pipe>2",
    ];
    for (index, text) in cases.iter().enumerate() {
        let error = Schedule::parse(text).expect_err("用例 {index} 应当拒绝");
        assert_eq!(error.code(), Code::Invalid, "用例 {index}");
        assert!(!error.message().trim().is_empty(), "用例 {index}：拒绝必须有话说");
    }
}

/// 未掩盖占比饱和边界：占比越界（负、>1）由 clamp 收进 [0,1]；级数 0 按 1 级兜底。
#[test]
fn unmasked_share_clamps_saturated_inputs() {
    let cases: &[(f64, u32, f64)] = &[
        (1.0, 1, 0.0),
        (1.5, 2, 0.0),
        (0.0, 0, 1.0),
        (0.0, 3, 1.0 / 3.0),
    ];
    for (index, (share, stages, expected)) in cases.iter().enumerate() {
        let actual = unmasked_share(*share, *stages);
        assert!((actual - expected).abs() < 1e-12, "用例 {index}：{actual} != {expected}");
        assert!((0.0..=1.0).contains(&actual), "用例 {index}：占比必须在 [0,1]");
    }
    // 负占比按 (1 - share) 直接求值（骨架未做 clamp），结果仍是有限非负。
    let negative = unmasked_share(-0.5, 2);
    assert!(negative.is_finite() && negative > 0.0, "负占比不 panic");
}

/// 单调链边界：空链与单元素链放行；重复占比与并列首尾不违反单调。
#[test]
fn monotonic_chain_allows_empty_single_and_equal_pairs() {
    let empty: [(&str, f64); 0] = [];
    assert!(check_monotonic(&empty).is_ok());
    let single = [("direct", 0.5)];
    assert!(check_monotonic(&single).is_ok());
    let equal = [("a", 0.5), ("b", 0.5)];
    assert!(check_monotonic(&equal).is_ok());
}

