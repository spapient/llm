use super::*;

#[test]
fn scan_matches_the_scalar_reference() {
    let sample = sample(128, 1);
    let expected = reference(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    forward(sample.f32("x").expect("x"), &mut out).expect("等长");
    assert_eq!(out, expected);
}

/// 输入与输出长度不一致时拒绝；非对齐长度不静默截断。
#[test]
fn scan_rejects_mismatched_output_length() {
    let mut out = vec![0.0; 3];
    assert!(
        forward(&[1.0, 2.0], &mut out).is_err(),
        "输出比输入长必须拒绝"
    );
    let mut short = vec![0.0; 1];
    assert!(
        forward(&[1.0, 2.0, 3.0], &mut short).is_err(),
        "输出比输入短必须拒绝"
    );
}

/// 递推初态为零：`y[t] = Σ_{k≤t} ā^(t-k)·b̄·x[k]`。取 `x = [1, 0, 0, …]` 得几何衰减
/// `1, ā, ā², …`，逐位验证半群语义与内核公式一致。
#[test]
fn scan_is_geometric_decay_for_impulse() {
    let x = [1.0f32, 0.0, 0.0, 0.0, 0.0];
    let expected = [1.0f32, 0.5, 0.25, 0.125, 0.0625];
    let mut out = vec![0.0; x.len()];
    forward(&x, &mut out).expect("等长");
    for (index, (got, want)) in out.iter().zip(expected).enumerate() {
        assert!((got - want).abs() < 1e-6, "第 {index} 步：{got} vs {want}");
    }
}

/// 空序列放行：空输入得空输出，扫描没有无定义的中间态。
#[test]
fn scan_allows_empty_sequence() {
    let mut out = vec![];
    forward(&[], &mut out).expect("空序列放行");
    assert_eq!(reference(&sample(0, 1)).expect("空参考"), Vec::<f32>::new());
}
