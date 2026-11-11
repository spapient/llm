//! 标量参考实现与对拍口径（DESIGN §11.1）。
//!
//! 片段只与这里的参考对拍，片段之间**不**互相对拍——否则一个错会传染成"两个都对"。

use crate::atom::DType;

/// weight-only int4 解码 GEMV 的标量参考。
///
/// 语义与 `ops/gemv_i4.k` 一致：
/// `y[n] = Σ_g Σ_k (nibble(w[n, g·G+k]) − 8) · scale[n, g] · x[g·G+k]`。
pub fn gemv_i4(
    x: &[f32],
    w: &[u8],
    scales: &[f32],
    m: usize,
    n: usize,
    group: usize,
) -> Result<Vec<f32>, String> {
    if group == 0 || !m.is_multiple_of(group) || !group.is_multiple_of(2) {
        return Err(format!("分组 {group} 必须能整除 {m} 且为偶数"));
    }
    if x.len() < m {
        return Err(format!("输入长度 {} < {m}", x.len()));
    }
    let groups = m / group;
    if w.len() < n * (m / 2) {
        return Err(format!("权重长度 {} < {}", w.len(), n * (m / 2)));
    }
    if scales.len() < n * groups {
        return Err(format!("scale 长度 {} < {}", scales.len(), n * groups));
    }

    let mut y = vec![0.0f32; n];
    for (row, out) in y.iter_mut().enumerate() {
        let base = row * (m / 2);
        let mut acc = 0.0f32;
        for g in 0..groups {
            let scale = scales[row * groups + g];
            let packed = &w[base + g * (group / 2)..base + (g + 1) * (group / 2)];
            let xs = &x[g * group..(g + 1) * group];
            for (index, byte) in packed.iter().enumerate() {
                let low = ((i32::from(byte & 0x0F)) - 8) as f32 * scale;
                let high = ((i32::from((byte >> 4) & 0x0F)) - 8) as f32 * scale;
                acc += low * xs[2 * index];
                acc += high * xs[2 * index + 1];
            }
        }
        *out = acc;
    }
    Ok(y)
}

/// 对拍口径：`max|got − want| / max|want|`。
pub fn max_rel_err(got: &[f32], want: &[f32]) -> f64 {
    if got.len() != want.len() {
        return f64::INFINITY;
    }
    let mut max_err = 0.0f64;
    let mut max_want = 0.0f64;
    for (a, b) in got.iter().zip(want.iter()) {
        max_err = max_err.max((f64::from(*a) - f64::from(*b)).abs());
        max_want = max_want.max(f64::from(*b).abs());
    }
    if max_want == 0.0 {
        return max_err;
    }
    max_err / max_want
}

/// 按数据类型的阈值（DESIGN §11.1）：整数精确、f16/bf16 ≤ 1e-3、f32 ≤ 1e-5。
pub fn tolerance(dt: DType) -> f64 {
    match dt {
        DType::F16 | DType::Bf16 => 1e-3,
        DType::F32 => 1e-5,
        _ => 0.0,
    }
}

/// 是否在阈值内。
pub fn within(got: &[f32], want: &[f32], dt: DType) -> bool {
    max_rel_err(got, want) <= tolerance(dt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(values: &[i8]) -> Vec<u8> {
        values
            .chunks(2)
            .map(|pair| {
                let low = (pair[0] + 8) as u8 & 0x0F;
                let high = (pair.get(1).copied().unwrap_or(0) + 8) as u8 & 0x0F;
                low | (high << 4)
            })
            .collect()
    }

    #[test]
    fn reference_dequantizes_and_dots() {
        // 1 行、1 组、4 个权重：[1, 2, 3, 4]，scale = 0.5，x = [1, 1, 1, 1]
        let w = packed(&[1, 2, 3, 4]);
        let y = gemv_i4(&[1.0, 1.0, 1.0, 1.0], &w, &[0.5], 4, 1, 4).expect("参考可算");
        assert_eq!(y.len(), 1);
        assert!((y[0] - 5.0).abs() < 1e-6, "0.5*(1+2+3+4)=5，实际 {}", y[0]);
    }

    #[test]
    fn reference_handles_multiple_groups_and_rows() {
        let m = 8;
        let n = 2;
        let group = 4;
        let w = packed(&[1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 2, 2, 2, 2]);
        let x: Vec<f32> = (0..m).map(|index| index as f32 + 1.0).collect();
        let scales = vec![0.25f32, 0.5, 1.0, 1.0];
        let y = gemv_i4(&x, &w, &scales, m, n, group).expect("参考可算");
        assert_eq!(y.len(), n);
        assert!(y[0] > 0.0);
        assert!(y[1] > y[0], "第二行权重更大");
    }

    #[test]
    fn shape_mismatch_is_an_error() {
        assert!(gemv_i4(&[1.0], &[0u8; 2], &[1.0], 4, 1, 4).is_err());
        assert!(gemv_i4(&[1.0; 4], &[0u8; 2], &[1.0], 4, 1, 3).is_err());
        assert!(gemv_i4(&[1.0; 4], &[0u8; 2], &[], 4, 1, 4).is_err());
    }

    #[test]
    fn relative_error_uses_the_biggest_magnitude() {
        assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        let err = max_rel_err(&[1.0, 2.1], &[1.0, 2.0]);
        assert!((err - 0.05).abs() < 1e-6, "{err}");
        assert!(within(&[1.0, 2.0], &[1.0, 2.0], DType::F32));
        assert!(!within(&[1.0, 2.5], &[1.0, 2.0], DType::F32));
    }

    #[test]
    fn tolerance_follows_dtype() {
        assert_eq!(tolerance(DType::F16), 1e-3);
        assert_eq!(tolerance(DType::F32), 1e-5);
        assert_eq!(tolerance(DType::I8), 0.0);
    }
}
