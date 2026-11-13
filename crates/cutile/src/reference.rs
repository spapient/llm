//! 对拍口径（DESIGN §11.1）。
//!
//! 标量参考由算子层提供（`op::model::Model::reference`）；本模块只回答"算不算对"：
//! 相对误差、按数据类型的阈值与判定。片段只与参考对拍，片段之间**不**互相对拍——否则一个
//! 错会传染成"两个都对"。

use crate::atom::DType;

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
        DType::F16 | DType::BF16 => 1e-3,
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
