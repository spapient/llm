use super::*;
use testkit::{COSINE_GUARD, Rng, cosine_similarity};

#[test]
fn forward_matches_a_hand_written_sum() {
    // 一行、一组：点积先按 int8 位模式累加，整组只乘一次标度。
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let w: Vec<u8> = [1i8, 2, 3, 4].iter().map(|value| *value as u8).collect();
    let scales = [0.5f32];
    let mut out = [0.0f32; 1];

    forward(&x, &w, &scales, &mut out).expect("合法");
    let expected = 0.5 * (1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0);
    assert!((out[0] - expected).abs() < 1e-6, "{:?}", out);
}

#[test]
fn negative_bytes_are_interpreted_as_signed() {
    // `u8` 承载 int8 位模式：255 必须按 -1 解释，而不是 255。
    let x = [1.0f32, 1.0];
    let w = [0xFFu8, 2u8];
    let scales = [1.0f32];
    let mut out = [0.0f32; 1];

    forward(&x, &w, &scales, &mut out).expect("合法");
    assert!((out[0] - 1.0).abs() < 1e-6, "-1 + 2 = 1，实际 {}", out[0]);
}

/// 合成基准：f32 参考权重对称量化到 int8 后喂给 [`forward`]，logits 方向必须与 f32 参考一致。
///
/// 这是权重符号分流改动后的回归守卫：逐元素 int8 链路算出的 logits，余弦相似度不得低于
/// 99.9%。阈值只看方向（余弦尺度不变），不要求逐元素相等；累加在 `f64` 完成以保住参考真值。
#[test]
fn quantized_logits_keep_cosine_with_the_f32_reference() {
    const M: usize = 64;
    const N: usize = 4;
    const GROUP: usize = 32;
    const GROUPS: usize = M / GROUP;

    let mut rng = Rng::new(2026_1115);
    let between = |rng: &mut Rng| (rng.below(2001) as f32 / 1000.0) - 1.0;

    let x: Vec<f32> = (0..M).map(|_| between(&mut rng)).collect();
    let weights: Vec<f32> = (0..N * M).map(|_| between(&mut rng)).collect();

    // f32 参考：f64 累加，避免 f32 平方和把参考真值压低。
    let reference: Vec<f32> = (0..N)
        .map(|row| {
            let dot = (0..M)
                .map(|k| f64::from(weights[row * M + k]) * f64::from(x[k]))
                .sum::<f64>();
            dot as f32
        })
        .collect();

    // 按行、按组对称量化：scale = 组内最大绝对值 / 127，权重回写成 int8 位模式。
    let mut packed = vec![0u8; N * M];
    let mut scales = vec![0.0f32; N * GROUPS];
    for row in 0..N {
        for group in 0..GROUPS {
            let base = row * M + group * GROUP;
            let max_abs = weights[base..base + GROUP]
                .iter()
                .map(|value| value.abs())
                .fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            scales[row * GROUPS + group] = scale;
            for k in 0..GROUP {
                let quantized = (weights[base + k] / scale).round();
                packed[base + k] = quantized.clamp(-127.0, 127.0) as i8 as u8;
            }
        }
    }

    let mut quantized = vec![0.0f32; N];
    forward(&x, &packed, &scales, &mut quantized).expect("合成形状合法");

    let cosine = cosine_similarity(&reference, &quantized).expect("两条向量非零");
    assert!(
        cosine >= COSINE_GUARD,
        "int8 量化 logits 与 f32 参考余弦 {cosine:.5} < {COSINE_GUARD:.3}"
    );
}

/// 标度负例：非有限标度在分组守卫后、解包前拦下，不允许把整组乘积带进 NaN/Inf。
#[test]
fn gemv_i8_rejects_non_finite_scales() {
    let m = 4;
    let w = vec![0u8; m];
    let x = vec![1.0; m];
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = forward(&x, &w, &[bad], &mut [0.0; 1]).expect_err("非有限标度必须拒绝");
        assert_eq!(error.code(), diagnostic::Code::Invalid);
    }
}
