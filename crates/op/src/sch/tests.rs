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
