/// 生成「参考对拍 + 长度负例」两则共有测试：不依赖算子的数值语义。
macro_rules! elementwise_pair_tests {
    ($ty:ty, $forward:ident) => {
        #[test]
        fn matches_the_scalar_reference() {
            let sample = crate::elementwise::sample::<$ty>(128, 1);
            let expected = crate::elementwise::reference::<$ty>(&sample).expect("参考可计算");
            let mut out = vec![0.0; 128];
            let a = sample.f32("a").expect("a 存在");
            let b = sample.f32("b").expect("b 存在");
            $forward(a, b, &mut out).expect("缓冲等长");
            assert_eq!(out, expected);
        }

        #[test]
        fn rejects_each_mismatched_pair_and_allows_empty() {
            let mut out = vec![0.0; 2];
            assert!(
                $forward(&[1.0], &[1.0, 2.0], &mut out).is_err(),
                "a 与 b 不等"
            );
            assert!(
                $forward(&[1.0, 2.0], &[1.0], &mut out).is_err(),
                "b 与 out 不等"
            );
            let mut empty = vec![];
            $forward(&[], &[], &mut empty).expect("空输入逐元素放行");
        }
    };
}

/// 生成 add/mul 等二元算子共有的三则测试：参考对拍、长度负例、饱和与重复边界。
macro_rules! elementwise_binary_tests {
    ($ty:ty, $forward:ident) => {
        $crate::elementwise::test_support::elementwise_pair_tests!($ty, $forward);

        #[test]
        fn saturating_and_repeated_inputs_match_reference() {
            let cases: &[(&[f32], &[f32])] = &[
                (&[0.0, 0.0], &[0.0, 0.0]),
                (&[1.0, 1.0], &[2.0, 2.0]),
                (&[f32::MAX, f32::MIN], &[0.0, 0.0]),
                (&[f32::MAX, f32::MAX], &[f32::MIN, f32::MIN]),
                (&[3.0, -1.0], &[-3.0, 1.0]),
            ];
            for (index, (a, b)) in cases.iter().enumerate() {
                let mut sample = crate::elementwise::sample::<$ty>(a.len(), 1);
                sample.buffers[0].values = crate::model::Values::F32(a.to_vec());
                sample.buffers[1].values = crate::model::Values::F32(b.to_vec());
                let expected = crate::elementwise::reference::<$ty>(&sample).expect("参考可计算");
                let mut out = vec![0.0; a.len()];
                $forward(a, b, &mut out).expect("等长");
                assert_eq!(out, expected, "用例 {index}");
            }
        }
    };
}
/// 除法专属：参考对拍加 IEEE-754 除零语义（符号决定 ±inf）。
macro_rules! elementwise_div_tests {
    ($ty:ty, $forward:ident) => {
        $crate::elementwise::test_support::elementwise_pair_tests!($ty, $forward);

        #[test]
        fn division_by_zero_matches_ieee() {
            let mut out = vec![0.0; 2];
            $forward(&[1.0, -1.0], &[0.0, 0.0], &mut out).expect("等长");
            assert!(out[0].is_sign_positive(), "1/0 = +inf");
            assert!(out[1].is_sign_negative(), "-1/0 = -inf");
            assert!(out.iter().all(|x| x.is_infinite()), "除零得无穷");
        }
    };
}
pub(crate) use elementwise_binary_tests;
pub(crate) use elementwise_div_tests;
pub(crate) use elementwise_pair_tests;
