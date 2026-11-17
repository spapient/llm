use super::*;

#[test]
fn rope_matches_the_scalar_reference() {
    let sample = sample(128, 1);
    let expected = reference(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    forward(7, sample.f32("x").expect("x"), &mut out).expect("等长");
    assert_eq!(out, expected);
}

#[test]
fn an_odd_length_passes_its_tail_through() {
    let out = rotate(&[1.0, 2.0, 3.0], 0.0).expect("奇数长度不报错");
    assert_eq!(out[2], 3.0, "末尾无法配对的元素原样透传");
}

/// 退化与重复边界：空行放行；长度不齐拒绝；重复偶数元素在同相位下结果对称。
#[test]
fn rope_handles_empty_mismatched_and_repeated_pairs() {
    let mut empty = vec![];
    forward(0, &[], &mut empty).expect("空行放行");

    let mut out = vec![0.0; 3];
    assert!(forward(0, &[1.0, 2.0], &mut out).is_err(), "长度不齐拒绝");

    let repeated = [1.0, 1.0, 1.0, 1.0];
    let mut out = vec![0.0; repeated.len()];
    forward(3, &repeated, &mut out).expect("重复对不越界");
    assert!(out.iter().all(|v| v.is_finite()), "结果有限");
}

#[test]
fn heads_split_rotation_recomputes_frequency_per_head() {
    let head_dim = 4;
    let heads = 2;
    let x: Vec<f32> = (0..heads * head_dim)
        .map(|i| i as f32 * 0.5 - 2.0)
        .collect();

    let mut out = vec![0.0; x.len()];
    forward_heads(7, &x, &mut out, head_dim).expect("分段旋转");

    // 参考：每头段独立按 head_dim 重计频率，段内下标从 0 起。
    let mut expected = vec![0.0; x.len()];
    for head in 0..heads {
        let span = head * head_dim..(head + 1) * head_dim;
        rotate_into(&x[span.clone()], 7.0, &mut expected[span]).expect("段内旋转");
    }
    assert_eq!(out, expected);
}

#[test]
fn one_head_rope_matches_the_single_head_forward() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut single = vec![0.0; x.len()];
    forward(3, &x, &mut single).expect("单头");
    let mut multi = vec![0.0; x.len()];
    forward_heads(3, &x, &mut multi, x.len()).expect("单头经多头入口");
    assert_eq!(single, multi, "heads=1 必须与单头入口逐元素等价");
}

#[test]
fn heads_rope_rejects_bad_head_dimension() {
    let mut out = vec![0.0; 8];
    // 头宽为 0。
    assert!(forward_heads(0, &[1.0; 8], &mut out, 0).is_err());
    // 长度不是头宽的整数倍。
    assert!(forward_heads(0, &[1.0; 7], &mut out, 4).is_err());
    // 输入输出不等长。
    let mut short = vec![0.0; 7];
    assert!(forward_heads(0, &[1.0; 8], &mut short, 4).is_err());
}
