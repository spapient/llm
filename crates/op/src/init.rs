//! 权重初始化：Xavier、Kaiming 与正交初始化（口径对齐 PyTorch 的 `nn.init`）。
//!
//! 全部只用 [`crate::rnd::SplitMix64`]，同一种子必得同一权重；函数签名是"形状 + 超参 + 种子"，
//! 返回按行主序排好的 `Vec<f32>`，调用方拿到即可直接当权重用。

use diagnostic::{Code, Error, Result};

use crate::rnd::SplitMix64;

/// Xavier 均匀：`U(-a, a)`，`a = sqrt(6 / (fan_in + fan_out))`。
///
/// # Errors
///
/// `fan_in`/`fan_out` 为 0 时返回说明。
pub fn xavier_uniform(fan_in: usize, fan_out: usize, seed: u64) -> Result<Vec<f32>> {
    let bound = xavier_bound(fan_in, fan_out)?;
    Ok(uniform(fan_in * fan_out, bound, seed))
}

/// Xavier 正态：`N(0, σ)`，`σ = sqrt(2 / (fan_in + fan_out))`。
///
/// # Errors
///
/// `fan_in`/`fan_out` 为 0 时返回说明。
pub fn xavier_normal(fan_in: usize, fan_out: usize, seed: u64) -> Result<Vec<f32>> {
    let std = xavier_std(fan_in, fan_out)?;
    Ok(normal(fan_in * fan_out, std, seed))
}

/// Kaiming 均匀：`U(-a, a)`，`a = sqrt(3)·gain / sqrt(fan_in)`，`gain = sqrt(2/(1+leak²))`。
///
/// # Errors
///
/// `fan_in` 为 0 时返回说明。
pub fn kaiming_uniform(fan_in: usize, fan_out: usize, leak: f32, seed: u64) -> Result<Vec<f32>> {
    let std = kaiming_std(fan_in, leak)?;
    Ok(uniform(fan_in * fan_out, bound_of(std), seed))
}

/// Kaiming 正态：`N(0, σ)`，`σ = gain / sqrt(fan_in)`。
///
/// # Errors
///
/// `fan_in` 为 0 时返回说明。
pub fn kaiming_normal(fan_in: usize, fan_out: usize, leak: f32, seed: u64) -> Result<Vec<f32>> {
    let std = kaiming_std(fan_in, leak)?;
    Ok(normal(fan_in * fan_out, std, seed))
}

/// 正交初始化：`rows ≥ cols` 时列正交，否则行正交；输出按 `[rows, cols]` 行主序，再乘 `gain`。
///
/// 用改进 Gram–Schmidt 对高斯矩阵正交化，数值上比拼 Givens 更稳。
///
/// # Errors
///
/// 任一维为 0 时返回说明。
pub fn orthogonal(rows: usize, cols: usize, gain: f32, seed: u64) -> Result<Vec<f32>> {
    if rows == 0 || cols == 0 {
        return Err(Error::new(Code::Invalid, "正交初始化不接受 0 维"));
    }
    let big = rows.max(cols);
    let small = rows.min(cols);
    let mat = orthonormalize(big, small, gain, seed);
    Ok(reshape_orthogonal(&mat, big, small, rows, cols))
}

/// 按 `[big, small]` 生成高斯列并用改进 Gram–Schmidt 正交化。
fn orthonormalize(big: usize, small: usize, gain: f32, seed: u64) -> Vec<f32> {
    let mut mat = vec![0.0f32; big * small];
    let mut rng = SplitMix64::new(seed);
    for v in mat.iter_mut() {
        *v = rng.next_normal();
    }
    for col in 0..small {
        subtract_prior(&mut mat, big, small, col);
        normalize_column(&mut mat, big, small, col, gain);
    }
    mat
}

/// 把列 `col` 对前面所有列做一次 Gram–Schmidt 减法。
fn subtract_prior(mat: &mut [f32], big: usize, small: usize, col: usize) {
    for prev in 0..col {
        let mut dot = 0.0f32;
        for r in 0..big {
            dot += mat[r * small + col] * mat[r * small + prev];
        }
        for r in 0..big {
            mat[r * small + col] -= dot * mat[r * small + prev];
        }
    }
}

/// 把列 `col` 归一化并乘 `gain`；零范数列保持为零。
fn normalize_column(mat: &mut [f32], big: usize, small: usize, col: usize, gain: f32) {
    let mut norm = 0.0f32;
    for r in 0..big {
        norm += mat[r * small + col] * mat[r * small + col];
    }
    let inv = if norm > 0.0 { gain / norm.sqrt() } else { 0.0 };
    for r in 0..big {
        mat[r * small + col] *= inv;
    }
}

/// `[big, small]` → `[rows, cols]`：`rows ≥ cols` 时列正交可直接拷贝，否则转置成行正交。
fn reshape_orthogonal(mat: &[f32], big: usize, small: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    if rows >= cols {
        out.copy_from_slice(mat);
        return out;
    }
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = mat[c * small + r];
        }
    }
    out
}

/// Xavier 的界 `sqrt(6/(fan_in+fan_out))`。
fn xavier_bound(fan_in: usize, fan_out: usize) -> Result<f32> {
    if fan_in == 0 || fan_out == 0 {
        return Err(Error::new(Code::Invalid, "Xavier 不接受 0 维"));
    }
    Ok((6.0 / (fan_in + fan_out) as f32).sqrt())
}

/// Xavier 的标准差 `sqrt(2/(fan_in+fan_out))`。
fn xavier_std(fan_in: usize, fan_out: usize) -> Result<f32> {
    if fan_in == 0 || fan_out == 0 {
        return Err(Error::new(Code::Invalid, "Xavier 不接受 0 维"));
    }
    Ok((2.0 / (fan_in + fan_out) as f32).sqrt())
}

/// Kaiming 的标准差 `gain/sqrt(fan_in)`。
fn kaiming_std(fan_in: usize, leak: f32) -> Result<f32> {
    if fan_in == 0 {
        return Err(Error::new(Code::Invalid, "Kaiming 不接受 0 维"));
    }
    let gain = (2.0 / (1.0 + leak * leak)).sqrt();
    Ok(gain / (fan_in as f32).sqrt())
}

/// 均匀分布的标准差 `σ = bound/√3` ⇒ `bound = √3·σ`。
fn bound_of(std: f32) -> f32 {
    3.0f32.sqrt() * std
}

/// `[−bound, bound)` 均匀抽样。
fn uniform(len: usize, bound: f32, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..len)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound)
        .collect()
}

/// `N(0, std)` 抽样。
fn normal(len: usize, std: f32, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..len).map(|_| rng.next_normal() * std).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xavier_uniform_within_bound() {
        let fan_in = 64;
        let fan_out = 128;
        let bound = (6.0 / 192.0f32).sqrt();
        let w = xavier_uniform(fan_in, fan_out, 5).unwrap();
        assert_eq!(w.len(), fan_in * fan_out);
        assert!(w.iter().all(|v| v.abs() <= bound + 1e-6), "越界");
    }

    #[test]
    fn kaiming_uniform_within_bound() {
        let fan_in = 256;
        let w = kaiming_uniform(fan_in, 8, 0.0, 6).unwrap();
        let std = (2.0 / fan_in as f32).sqrt();
        let bound = 3.0f32.sqrt() * std;
        assert!(w.iter().all(|v| v.abs() <= bound + 1e-6));
    }

    #[test]
    fn orthogonal_has_unit_columns() {
        let (rows, cols) = (32, 16);
        let w = orthogonal(rows, cols, 1.0, 7).unwrap();
        assert_eq!(w.len(), rows * cols);
        // 列正交：WᵀW ≈ I。
        for a in 0..cols {
            for b in 0..cols {
                let mut dot = 0.0f32;
                for r in 0..rows {
                    dot += w[r * cols + a] * w[r * cols + b];
                }
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-3, "({a},{b}) = {dot}");
            }
        }
    }

    #[test]
    fn orthogonal_has_unit_rows_when_fat() {
        let (rows, cols) = (8, 20);
        let w = orthogonal(rows, cols, 1.0, 8).unwrap();
        for a in 0..rows {
            for b in 0..rows {
                let mut dot = 0.0f32;
                for c in 0..cols {
                    dot += w[a * cols + c] * w[b * cols + c];
                }
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-3, "({a},{b}) = {dot}");
            }
        }
    }

    #[test]
    fn same_seed_reproduces_weights() {
        let a = kaiming_normal(512, 64, 0.01, 99).unwrap();
        let b = kaiming_normal(512, 64, 0.01, 99).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn zero_dims_are_rejected() {
        assert!(xavier_uniform(0, 8, 1).is_err());
        assert!(kaiming_uniform(0, 8, 0.0, 1).is_err());
        assert!(orthogonal(0, 4, 1.0, 1).is_err());
    }
}
