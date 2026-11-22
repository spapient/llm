/// 生成「参考对拍 + 长度负例」两则共有测试：不依赖算子的数值语义。
macro_rules! unary_pair_tests {
    ($ty:ty, $forward:ident) => {
        #[test]
        fn matches_the_scalar_reference() {
            let sample = crate::unary::sample::<$ty>(128, 1);
            let expected = crate::unary::reference::<$ty>(&sample).expect("参考可计算");
            let mut out = vec![0.0; 128];
            let a = sample.f32("a").expect("a 存在");
            $forward(a, &mut out).expect("缓冲等长");
            assert_eq!(out, expected);
        }

        #[test]
        fn rejects_mismatched_length_and_allows_empty() {
            let mut out = vec![0.0; 3];
            assert!($forward(&[1.0, 2.0], &mut out).is_err(), "输出比输入长");
            let mut empty = vec![];
            $forward(&[], &mut empty).expect("空行逐元素放行");
        }
    };
}

/// 生成 scale/silu 等一元算子共有的三则测试：参考对拍、长度负例、饱和与重复边界。
macro_rules! unary_op_tests {
    ($ty:ty, $forward:ident) => {
        $crate::unary::test_support::unary_pair_tests!($ty, $forward);

        #[test]
        fn saturating_and_repeated_inputs_match_reference() {
            let cases: &[&[f32]] = &[
                &[0.0, 0.0],
                &[1.0, 1.0, 1.0],
                &[-1.0, -2.0, -3.0],
                &[f32::MAX, f32::MAX],
                &[f32::MIN, f32::MIN],
            ];
            for (idx, a) in cases.iter().enumerate() {
                let mut sample = crate::unary::sample::<$ty>(a.len(), 1);
                sample.bufs[0].values = crate::model::Values::F32(a.to_vec());
                let expected = crate::unary::reference::<$ty>(&sample).expect("参考可计算");
                let mut out = vec![0.0; a.len()];
                $forward(a, &mut out).expect("等长");
                assert_eq!(out, expected, "用例 {idx}");
                assert!(out.iter().all(|v| v.is_finite()), "用例 {idx}：结果有限");
            }
        }
    };
}
pub(crate) use unary_op_tests;
pub(crate) use unary_pair_tests;
