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
    for (idx, (got, want)) in out.iter().zip(expected).enumerate() {
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

/// M11 因果前缀不变性：给同一条输入追加未来 token 后，已生成前缀 `1..t` 的输出必须逐位不变。
///
/// 串行递推的算术对前缀是确定且相同的，因此这里是**严格 0 偏差**（不是容差比较），
/// 对应 §10.10 M11 卡「未来 token 改变不得影响前缀输出」。
#[test]
fn m11_prefix_is_invariant_to_future_tokens() {
    let mut seed = 0xfeed_beefu64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as u32 as f32 / (u32::MAX as f32)) * 2.0 - 1.0
    };
    let full: Vec<f32> = (0..256).map(|_| next()).collect();
    for t in [1usize, 7, 31, 64, 127, 255] {
        let mut prefix_out = vec![0.0f32; t];
        forward(&full[..t], &mut prefix_out).expect("等长");
        let mut full_out = vec![0.0f32; full.len()];
        forward(&full, &mut full_out).expect("等长");
        assert_eq!(
            &full_out[..t],
            prefix_out.as_slice(),
            "前缀 {t} 输出因未来 token 改变"
        );
    }
}
