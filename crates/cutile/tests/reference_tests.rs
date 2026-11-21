//! 对拍口径的单元测试：误差量级、长度路径、阈值分档与非有限值分类。

use super::*;

#[test]
fn relative_error_uses_the_biggest_magnitude() {
    assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    let err = max_rel_err(&[1.0, 2.1], &[1.0, 2.0]);
    assert!((err - 0.05).abs() < 1e-6, "{err}");
    assert!(within(&[1.0, 2.0], &[1.0, 2.0], DType::F32));
    assert!(!within(&[1.0, 2.5], &[1.0, 2.0], DType::F32));
}

#[test]
fn every_length_takes_the_same_path_to_the_same_answer() {
    // 8 路展开后有三条路径：整块、整块 + 尾部、只有尾部。三者对同一批数必须一致。
    let want: Vec<f32> = (0..19u32).map(|i| i as f32 * 0.5).collect();
    let mut got = want.clone();
    got[17] += 0.25;
    let expected = 0.25 / 9.0;
    assert!((max_rel_err(&got, &want) - expected).abs() < 1e-9);
    // 长度不足一组：只走尾部。
    assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    assert_eq!(max_rel_err(&[], &[]), 0.0);
    // 不是同一批数时是无穷，而不是某个"很大"的误差。
    assert_eq!(max_rel_err(&[1.0], &[1.0, 2.0]), f64::INFINITY);
}

#[test]
fn tolerance_follows_dtype() {
    assert_eq!(tolerance(DType::F16), 1e-3);
    assert_eq!(tolerance(DType::F32), 1e-5);
    assert_eq!(tolerance(DType::I8), 0.0);
}

#[test]
fn non_finite_values_are_classified_not_averaged_away() {
    // 同是 NaN、同号 Inf 算对上；其余非有限组合是不一致，不能被 max 当缺席吞掉。
    assert!(within(&[f32::NAN], &[f32::NAN], DType::F32));
    assert!(within(&[f32::INFINITY], &[f32::INFINITY], DType::F32));
    assert!(!within(&[f32::NAN], &[0.0], DType::F32));
    assert!(!within(&[f32::INFINITY], &[f32::NEG_INFINITY], DType::F32));
    assert!(!within(&[f32::INFINITY], &[1.0], DType::F32));
    assert_eq!(max_rel_err(&[f32::NAN], &[0.0]), f64::INFINITY);
}

#[test]
fn the_fused_oracle_runs_producer_then_consumer_on_one_sample() {
    let mul = op::model("mul_f32").expect("mul_f32 已登记");
    let add = op::model("add_f32").expect("add_f32 已登记");
    let mut sample = (mul.sample)(8, 8);
    let want: Vec<f32> = {
        let a = sample.f32("a").expect("样本有 a").to_vec();
        let b = sample.f32("b").expect("样本有 b").to_vec();
        // 独立算一遍：mul(a,b) 的结果作为 add 的第一路输入，再与 b 相加。
        a.iter().zip(&b).map(|(x, y)| (x * y) + y).collect()
    };
    let got = fused_reference(&mul, &add, "a", &mut sample).expect("融合参考可算");
    assert!(within(&got, &want, DType::F32), "对拍失败");
}
