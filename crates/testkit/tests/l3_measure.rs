//! 真实权重 → 一段 L3 事实：两侧范数取 Frobenius 上界，`ε` 量在**将要使用的那份产物**上。
//!
//! 界偏松是取舍不是判据放宽：`‖W‖₂ ≤ ‖W‖_F` 恒成立、无需先做 SVD，而 `ε` 由量化后再解回的
//! 结果现算。`L_i` / `b_i` 不给默认值——`weights::l3` 刻意不提供任何"内置激活常量"，所以
//! [`rung`] 要调用方显式传。

use weights::l3::Rung;
use weights::{I4, dequantize, quantize};

/// 量化分组：32 元素一组、一个标度，与 `nn`/`serve` 的 4bit 权重口径一致。
const GROUP: usize = 32;

/// 确定性样本：`sample` 落在 `[-1, 1)`，乘一个小尺度让 `ρ < 1`（不放大误差的常规初始化）。
pub fn scaled(len: usize, salt: u32, scale: f32) -> Vec<f32> {
    crate::sample(len, salt)
        .iter()
        .map(|value| value * scale)
        .collect()
}

/// `‖W‖_F`：`‖W‖₂ ≤ ‖W‖_F` 对任何矩阵成立，拿它当 `ρ` 无需先分解。
pub fn frobenius(w: &[f32]) -> f64 {
    w.iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt()
}

/// 两列之差的欧氏范数：同一个算式既当 `ε = ‖W − Ŵ‖_F`，也当两条输出的实测差。
pub fn distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&p, &q)| ((p - q) as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// 行主序 `[rows, k]` × `[n, k]ᵀ`：走 `serve`/`train` 用的那份 TN 分块内核，不另写参考乘法。
pub fn forward(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    nn::matmul::forward_tn_tiled(x, w, rows, n, k, &mut out).expect("几何由用例自己保证");
    out
}

/// 两层 ReLU 网络 `max(0, x·W₁ᵀ)·W₂ᵀ`；第二层权重由调用方给（可以是量化后的那份）。
pub fn two_layer(x: &[f32], w1: &[f32], w2: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    let hidden: Vec<f32> = forward(x, w1, rows, n, k)
        .into_iter()
        .map(|value| value.max(0.0))
        .collect();
    forward(&hidden, w2, rows, n, n)
}

/// 4bit 逐组量化后再解回的那份权重：`ε` 量的是**将要使用的那份产物**，不是理论估计。
pub fn quantized(w: &[f32]) -> Vec<f32> {
    let (packed, scales) = quantize(I4, GROUP, w).expect("量化");
    let mut out = vec![0.0f32; w.len()];
    dequantize(I4, &packed, &scales, GROUP, &mut out).expect("反量化");
    out
}

/// 两版权重 → 一段 L3 事实：`ρ = ‖W‖_F`、`ρ̂ = ‖Ŵ‖_F`、`ε = ‖W − Ŵ‖_F`。
pub fn rung(w: &[f32], w_hat: &[f32], lipschitz: f64, offset: f64) -> Rung {
    Rung::new(
        frobenius(w),
        frobenius(w_hat),
        distance(w, w_hat),
        lipschitz,
        offset,
    )
    .expect("有限样本")
}
