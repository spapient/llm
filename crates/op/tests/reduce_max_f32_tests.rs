use super::*;

crate::reduce::test_support::reduce_op_tests!(Max, forward, f32::MIN);

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
