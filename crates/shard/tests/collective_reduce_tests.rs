use super::*;

#[test]
fn every_kind_names_itself() {
    let names: Vec<&str> = Reduce::ALL.iter().map(|kind| kind.as_str()).collect();
    assert_eq!(names, ["sum", "max", "min", "prod"]);
}

/// 单位元必须中立：拿它去 `apply` 任何值都不能改变那个值。
///
/// `Max` 的单位元写成 `f32::MIN` 时这条会在负值上失败——那正是最容易被漏掉的输入。
#[test]
fn the_identity_is_neutral_for_every_kind() {
    let samples = [
        -1.5e30,
        -3.0,
        -0.0,
        0.0,
        1.0e-30,
        7.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    for kind in Reduce::ALL {
        for value in samples {
            assert_eq!(
                kind.apply(kind.identity(), value),
                value,
                "{kind:?} 的单位元吃掉了 {value}"
            );
            assert_eq!(
                kind.apply(value, kind.identity()),
                value,
                "{kind:?} 的单位元吃掉了 {value}"
            );
        }
    }
}

#[test]
fn apply_is_the_expected_arithmetic() {
    assert_eq!(Reduce::Sum.apply(2.0, 3.0), 5.0);
    assert_eq!(Reduce::Prod.apply(2.0, 3.0), 6.0);
    assert_eq!(Reduce::Max.apply(2.0, 3.0), 3.0);
    assert_eq!(Reduce::Min.apply(2.0, 3.0), 2.0);
    // 负值上的极值：`0.0` 不能是任何一个极值的单位元。
    assert_eq!(Reduce::Max.apply(-9.0, -2.0), -2.0);
    assert_eq!(Reduce::Min.apply(-9.0, -2.0), -9.0);
}

/// NaN 必须传播，而不是被 `f32::max` / `f32::min` 静默吃掉。
#[test]
fn a_non_finite_value_is_never_silently_dropped() {
    assert!(Reduce::Max.apply(f32::NAN, 1.0).is_nan());
    assert!(Reduce::Min.apply(1.0, f32::NAN).is_nan());
    assert!(Reduce::Sum.apply(f32::NAN, 1.0).is_nan());
    assert!(Reduce::Prod.apply(1.0, f32::NAN).is_nan());
}

/// 空输入没有答案：不许拿单位元冒充"这一组算完了"。
#[test]
fn folding_nothing_is_not_a_value() {
    for kind in Reduce::ALL {
        assert_eq!(kind.fold(&[]), None, "{kind:?} 给空输入编了一个值");
    }
    assert_eq!(Reduce::Sum.fold(&[1.0]), Some(1.0));
    assert_eq!(Reduce::Max.fold(&[-4.0, -7.0]), Some(-4.0));
    assert_eq!(Reduce::Prod.fold(&[2.0, 3.0, 4.0]), Some(24.0));
}
