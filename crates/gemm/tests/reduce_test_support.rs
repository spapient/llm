/// 生成 reduce_max/reduce_sum 共有的两则测试：参考对拍、非单元素输出与空输入折叠。
macro_rules! reduce_op_tests {
    ($ty:ty, $forward:ident, $empty:expr) => {
        #[test]
        fn matches_the_scalar_reference() {
            let sample = crate::reduce::sample::<$ty>(128, 1);
            let expected = crate::reduce::reference::<$ty>(&sample).expect("参考可计算");
            let mut out = vec![0.0];
            $forward(sample.f32("a").expect("a"), &mut out).expect("单元素输出");
            assert_eq!(out, expected);
        }

        #[test]
        fn rejects_non_singleton_output_and_handles_empty_input() {
            let mut wide = vec![0.0; 2];
            assert!(
                $forward(&[1.0, 2.0], &mut wide).is_err(),
                "归约输出必须是单元素"
            );
            let mut out = vec![0.0];
            $forward(&[], &mut out).expect("空输入折叠到初值");
            assert_eq!(out[0], $empty);
        }
    };
}
/// 饱和与退化输入：极值、全零、重复元素与负数组，结果与标量参考一致且不越界。
macro_rules! reduce_saturating_tests {
    ($ty:ty, $forward:ident, $cases:expr) => {
        #[test]
        fn saturating_and_repeated_inputs_match_reference() {
            let cases: &[&[f32]] = $cases;
            for (idx, input) in cases.iter().enumerate() {
                let mut sample = crate::reduce::sample::<$ty>(input.len(), 1);
                sample.bufs[0].values = crate::model::Values::F32(input.to_vec());
                let expected = crate::reduce::reference::<$ty>(&sample).expect("参考可计算");
                let mut out = vec![0.0];
                $forward(input, &mut out).expect("单元素输出");
                assert_eq!(out, expected, "用例 {index}");
            }
        }
    };
}
pub(crate) use reduce_op_tests;
pub(crate) use reduce_saturating_tests;
