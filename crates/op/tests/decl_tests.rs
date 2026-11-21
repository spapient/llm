use super::*;
use diagnostic::data::Value as TValue;

fn builtin() -> OpDecl {
    // 按名字取 gemv_f32：注册表按名字排序，新增算子后不应再假设索引 0 是哪条。
    let op = crate::builtin_named("gemv_f32").expect("gemv_f32 内置");
    OpDecl::parse(op.declaration(), op.template()).expect("内置声明必须自洽")
}

/// 计数与展开必须同一个口径：装载期只数不物化，调用方仍然可以拿到组合。
#[test]
fn the_tile_count_matches_the_expanded_combinations() {
    let op = builtin();
    assert_eq!(
        tiles(&op).expect("未超限"),
        expand(&op).expect("可展开").len()
    );
}

#[test]
fn a_tile_overflow_names_the_axis_to_prune() {
    let op = crate::builtin_named("gemv_f32").expect("gemv_f32");
    let mut broken = op.declaration().clone();
    broken.set(
        "axes.GMAX",
        TValue::array(vec![
            TValue::int(1),
            TValue::int(2),
            TValue::int(4),
            TValue::int(8),
            TValue::int(16),
            TValue::int(32),
            TValue::int(64),
            TValue::int(128),
            TValue::int(256),
        ]),
    );
    let error = OpDecl::parse(&broken, op.template()).expect_err("9 个候选超过上限 8");
    assert_eq!(error.code(), Code::Invalid);
    let text = error.to_string();
    assert!(text.contains("GMAX"), "{text}");
    assert!(text.contains("超过上限 8"), "{text}");
    assert!(text.contains("9 个 tile 候选"), "{text}");
}

/// 属性键属于原语自己的签名：`sat` 是 `cvt` 的参数，写在 `store` 上既不会被协商检查，
/// 也不会被片段消费。判据必须点名原语、键与该原语真正允许的参数。
#[test]
fn an_attr_key_must_belong_to_its_primitive() {
    let op = crate::builtin_named("gemv_f32").expect("gemv_f32");
    let mut broken = op.declaration().clone();
    broken.set("needs.store.sat", TValue::array(vec![TValue::text("off")]));
    let store = broken
        .table("needs")
        .and_then(|needs| needs.table("store"))
        .expect("store 需求");
    assert!(store.contains("sat"), "前提：测试要改的是 store 的签名");
    let error = OpDecl::parse(&broken, op.template()).expect_err("`store.sat` 不属于签名");
    let text = error.to_string();
    assert!(text.contains("`store`"), "{text}");
    assert!(text.contains("`sat`"), "{text}");
    assert!(text.contains("src | dst | align | exec"), "{text}");
}

/// 降级链的单调性判据接受借用名：装载期不为它拷一份 `String`。
#[test]
fn the_monotonic_chain_is_checked_on_borrowed_names() {
    let chain = [("first", 0.5), ("second", 0.5)];
    assert!(sch::check_monotonic(&chain).is_ok());
    let reversed = [("first", 0.4), ("second", 0.9)];
    assert!(sch::check_monotonic(&reversed).is_err());
}
