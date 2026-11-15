use super::*;

#[test]
fn lengths_are_looked_up_by_letter() {
    let extents = Extents::new([('n', 8), ('m', 64), ('g', 2)]).expect("合法");
    assert_eq!(extents.get('n'), Some(8));
    assert_eq!(extents.get('m'), Some(64));
    assert_eq!(extents.get('z'), None);
    assert_eq!(extents.len(), 3);
    assert!(!extents.is_empty());
}

#[test]
fn the_display_is_stable_and_sorted() {
    let extents = Extents::new([('n', 8), ('g', 2), ('m', 64)]).expect("合法");
    assert_eq!(extents.to_string(), "g=2 m=64 n=8");
}

#[test]
fn an_empty_table_is_allowed_because_a_scalar_has_no_axes() {
    let extents = Extents::new([]).expect("合法");
    assert!(extents.is_empty());
    assert_eq!(extents.to_string(), "");
}

#[test]
fn a_zero_length_axis_is_rejected() {
    let error = Extents::new([('n', 0)]).expect_err("空轴应当被拒绝");
    assert!(error.to_string().contains("长度为 0"), "{error}");
}

#[test]
fn a_duplicate_axis_is_rejected_instead_of_overwritten() {
    let error = Extents::new([('n', 8), ('n', 16)]).expect_err("重复轴应当被拒绝");
    assert!(error.to_string().contains("两次"), "{error}");
}

#[test]
fn a_non_letter_axis_is_rejected() {
    let error = Extents::new([('1', 8)]).expect_err("非字母应当被拒绝");
    assert!(error.to_string().contains("不是字母"), "{error}");
}

#[test]
fn a_grouped_axis_derives_its_length_and_remembers_its_base() {
    let extents = Extents::new([('m', 64)])
        .expect("合法")
        .group('g', 'm', 32)
        .expect("64 能被 32 整除");
    assert_eq!(extents.get('g'), Some(2));
    assert_eq!(extents.grouped('g'), Some(('m', 32)));
    assert_eq!(extents.grouped('m'), None);
    assert_eq!(extents.len(), 2);
    assert_eq!(extents.to_string(), "g=2 m=64");
}

#[test]
fn grouping_rejects_a_base_that_cannot_be_divided_evenly() {
    let error = Extents::new([('m', 33)])
        .expect("合法")
        .group('g', 'm', 32)
        .expect_err("33 不能被 32 整除");
    assert!(error.to_string().contains("整数倍"), "{error}");
}

#[test]
fn grouping_rejects_a_missing_base_and_a_zero_divisor() {
    let missing = Extents::new([('n', 8)])
        .expect("合法")
        .group('g', 'm', 32)
        .expect_err("没有 m");
    assert!(missing.to_string().contains("没有长度"), "{missing}");

    let zero = Extents::new([('m', 64)])
        .expect("合法")
        .group('g', 'm', 0)
        .expect_err("因子不能为 0");
    assert!(zero.to_string().contains("不能为 0"), "{zero}");
}

#[test]
fn grouping_rejects_a_taken_label_and_a_self_base() {
    let taken = Extents::new([('m', 64), ('g', 3)])
        .expect("合法")
        .group('g', 'm', 32)
        .expect_err("g 已经存在");
    assert!(taken.to_string().contains("两次"), "{taken}");

    let itself = Extents::new([('m', 64)])
        .expect("合法")
        .group('m', 'm', 32)
        .expect_err("不能以自己为基准");
    assert!(itself.to_string().contains("自己"), "{itself}");
}
