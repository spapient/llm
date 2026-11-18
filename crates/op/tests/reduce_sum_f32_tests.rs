use super::*;

crate::reduce::test_support::reduce_op_tests!(Sum, forward, 0.0);

/// 饱和与退化输入：全零、同号大值、重复边界元素，结果与标量参考一致且不越界。
#[test]
fn sum_saturating_and_repeated_inputs_match_reference() {
    let cases: &[&[f32]] = &[
        &[0.0, 0.0, 0.0],
        &[f32::MAX, f32::MAX],
        &[f32::MIN, f32::MIN],
        &[1.0, 1.0, 1.0, 1.0],
        &[-3.0, 0.0, 3.0],
    ];
    for (index, input) in cases.iter().enumerate() {
        let mut sample = crate::reduce::sample::<Sum>(input.len(), 1);
        sample.buffers[0].values = crate::model::Values::F32(input.to_vec());
        let expected = crate::reduce::reference::<Sum>(&sample).expect("参考可计算");
        let mut out = vec![0.0];
        forward(input, &mut out).expect("单元素输出");
        assert_eq!(out, expected, "用例 {index}");
    }
}
