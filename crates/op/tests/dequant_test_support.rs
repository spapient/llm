/// 生成 dequant_i2/i4 共有的三则测试：参考对拍、退化边界、仿射参数负例。
macro_rules! dequant_packed_tests {
    ($sample:ident, $reference:ident, $forward:ident, $scale:ident, $zero:ident,
         $cpb:expr, $domain:expr, $neg:expr, $nonfinite:expr) => {
        #[test]
        fn dequant_matches_the_scalar_reference() {
            let sample = $sample(128, 1);
            let expected = $reference(&sample).expect("参考可计算");
            let mut out = vec![0.0; 128];
            $forward($scale, $zero, sample.u8("q").expect("q"), &mut out).expect("等长");
            assert_eq!(out, expected);
        }

        #[test]
        fn handles_empty_mismatched_and_saturated_bytes() {
            let mut empty = vec![];
            $forward($scale, $zero, &[], &mut empty).expect("空行放行");

            let mut wrong = vec![0.0; 3];
            assert!(
                $forward($scale, $zero, &[0x12, 0x34], &mut wrong).is_err(),
                "输出长度必须是字节数的整数倍"
            );

            let saturated = [0x00u8, 0xFF];
            let mut out = vec![0.0; saturated.len() * $cpb];
            $forward($scale, $zero, &saturated, &mut out).expect("极值字节不越界");
            assert!(out.iter().all(|v| v.is_finite()), "结果有限");
        }

        #[test]
        fn rejects_zero_scale_and_out_of_domain_zero() {
            let q = [0x52u8];
            let mut out = [0.0; $cpb];

            assert!($forward(0.0, $zero, &q, &mut out).is_err(), "零标度拒绝");
            assert!(
                $forward($scale, $nonfinite, &q, &mut out).is_err(),
                "非有限零点拒绝"
            );
            assert!(
                $forward($scale, $domain, &q, &mut out).is_err(),
                "零点越出码域"
            );
            assert!(
                $forward($scale, $neg, &q, &mut out).is_err(),
                "负零点越出码域"
            );
        }
    };
}
pub(crate) use dequant_packed_tests;
