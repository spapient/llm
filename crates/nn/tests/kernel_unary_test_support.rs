/// 生成「参考对拍 + 长度负例」两则共有测试：不依赖算子的数值语义。
macro_rules! unary_pair_tests {
    ($ty:ty, $forward:ident) => {
        #[test]
        fn matches_the_scalar_reference() {
            let sample = crate::unary::sample::<$ty>(128, 1);
            let expected = crate::unary::reference::<$ty>(&sample).expect("参考可计算");
            let mut out = vec![0.0; 128];
            let a = sample.f32("a").expect("a 存在");
            $forward(tensor::View::one(a), &mut out).expect("缓冲等长");
            assert_eq!(out, expected);
        }

        #[test]
        fn rejects_mismatched_length_and_allows_empty() {
            let mut out = vec![0.0; 3];
            assert!(
                $forward(tensor::View::one(&[1.0, 2.0]), &mut out).is_err(),
                "输出比输入长"
            );
            let mut empty: Vec<f32> = Vec::new();
            $forward(tensor::View::one(&[]), &mut empty).expect("空行逐元素放行");
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
                $forward(tensor::View::one(a), &mut out).expect("等长");
                assert_eq!(out, expected, "用例 {idx}");
                assert!(out.iter().all(|v| v.is_finite()), "用例 {idx}：结果有限");
            }
        }
    };
}
/// 生成**指定精度档**一元算子的「参考对拍 + 长度负例」两则共有测试：样本/参考/标量类型/取值访问器
/// 全是参数，同一份测试骨架按档单态化（f16/bf16 共用），不必逐档逐算子重写。
macro_rules! unary_tier_tests {
    ($sample:path, $reference:path, $k:ty, $acc:ident, $forward:ident) => {
        #[test]
        fn matches_the_scalar_reference() {
            let sample = $sample(128, 1);
            let expected = $reference(&sample).expect("参考可计算");
            let a = sample.$acc("a").expect("a 存在");
            let mut out = vec![<$k>::from_f32(0.0); 128];
            $forward(tensor::View::one(a), &mut out).expect("缓冲等长");
            let got: Vec<f32> = out.iter().map(|value| value.to_f32()).collect();
            assert_eq!(got, expected);
        }

        #[test]
        fn rejects_mismatched_length_and_allows_empty() {
            let a = [<$k>::from_f32(1.0), <$k>::from_f32(2.0)];
            let mut out = vec![<$k>::from_f32(0.0); 3];
            assert!(
                $forward(tensor::View::one(&a), &mut out).is_err(),
                "输出比输入长"
            );
            let empty_in: &[$k] = &[];
            let mut empty: Vec<$k> = vec![];
            $forward(tensor::View::one(empty_in), &mut empty).expect("空行逐元素放行");
        }
    };
}

/// 按表核对一元边界：`tol = 0` 保持精确相等，其余元素保持严格绝对容差。
pub(crate) fn assert_unary_edge(
    forward: fn(tensor::View<'_, f32>, &mut [f32]) -> diagnostic::Result<()>,
    input: &[f32],
    want: &[f32],
    tol: &[f32],
) {
    assert_eq!(input.len(), want.len(), "输入与期望长度必须一致");
    assert_eq!(input.len(), tol.len(), "输入与容差长度必须一致");
    let mut got = vec![0.0; input.len()];
    forward(tensor::View::one(input), &mut got).expect("等长");
    for (idx, ((&got, &want), &tol)) in got.iter().zip(want).zip(tol).enumerate() {
        if tol == 0.0 {
            assert_eq!(got, want, "第 {idx} 项必须精确相等");
        } else {
            assert!((got - want).abs() < tol, "第 {idx} 项 {got} vs {want}（容差 {tol}）");
        }
    }
}

pub(crate) use unary_op_tests;
pub(crate) use unary_pair_tests;
pub(crate) use unary_tier_tests;
