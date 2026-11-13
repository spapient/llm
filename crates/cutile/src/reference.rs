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

/// 按数据类型的阈值（DESIGN §11.1）：浮点按尾数宽度分档，整数与布尔位精确。
///
/// 逐条列出而不是用 `_` 兜底：闭集新增一种格式时，这里必须表态。上一版的 `_ => 0.0` 会让
/// 新加的浮点格式静默落到「位精确」，于是正确的解码也会被判成错值，看起来像内核算错。
///
/// 「非零阈值的类型集合」是**契约**：它必须与注解层可声明的精度闭集（`annotation` 的
/// `DTYPES`）逐项相等，`tests/contracts.rs` 会对拍。所以 F4 / F8 族在这里是 `0.0`——
/// 它们还没有进入执行路径；一旦接入，改这里会同时把门禁叫醒，提醒去扩 `DTYPES`。
pub fn tolerance(dt: DType) -> f64 {
    match dt {
        DType::F16 | DType::BF16 => 1e-3,
        DType::F32 => 1e-5,
        // 浮点但当前无执行路径（MXFP4 的元素与 8 位浮点族）：阈值 0 表示"精确对拍"，
        // 接入解码/执行时与 `DTYPES` 一起改。
        DType::F4
        | DType::F8E4M3
        | DType::F8E5M2
        | DType::F8E4M3FNUZ
        | DType::F8E5M2FNUZ
        // 整数与布尔：位精确，任何差值都是错值。
        | DType::Bool
        | DType::U1
        | DType::U2
        | DType::U4
        | DType::U8
        | DType::U16
        | DType::U32
        | DType::U64
        | DType::I1
        | DType::I2
        | DType::I4
        | DType::I8
        | DType::I16
        | DType::I32
        | DType::I64 => 0.0,
        DType::F64 => 0.0,
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
