use super::*;

#[test]
fn max_matches_the_scalar_reference() {
    let sample = crate::reduce::sample::<Max>(128, 1);
    let expected = crate::reduce::reference::<Max>(&sample).expect("参考可计算");
    let mut out = vec![0.0];
    forward(sample.f32("a").expect("a"), &mut out).expect("单元素输出");
    assert_eq!(out, expected);
}

/// 归约输出不是单元素时拒绝；空输入折叠到初值 `f32::MIN`。
#[test]
fn max_rejects_non_singleton_output_and_handles_empty_input() {
    let mut wide = vec![0.0; 2];
    assert!(
        forward(&[1.0, 2.0], &mut wide).is_err(),
        "归约输出必须是单元素"
    );
    let mut out = vec![0.0];
    forward(&[], &mut out).expect("空输入折叠到初值");
    assert_eq!(out[0], f32::MIN);
}

/// 饱和与重复边界：极值、全零、重复元素与负数组，结果与标量参考一致。
#[test]
fn max_saturating_and_repeated_inputs_match_reference() {
    let cases: &[&[f32]] = &[
        &[0.0, 0.0, 0.0],
        &[f32::MIN, f32::MAX, -1.0],
        &[f32::MIN, f32::MIN],
        &[2.0, 2.0, 2.0],
        &[-3.0, -1.0, -2.0],
    ];
    for (index, input) in cases.iter().enumerate() {
        let mut sample = crate::reduce::sample::<Max>(input.len(), 1);
        sample.buffers[0].values = crate::model::Values::F32(input.to_vec());
        let expected = crate::reduce::reference::<Max>(&sample).expect("参考可计算");
        let mut out = vec![0.0];
        forward(input, &mut out).expect("单元素输出");
        assert_eq!(out, expected, "用例 {index}");
    }
}
