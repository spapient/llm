//! 对拍口径（README §11.1）。

use crate::atom::DType;

/// 对拍口径：`max|got − want| / max|want|`。
pub fn max_rel_err(got: &[f32], want: &[f32]) -> f64 {
    if got.len() != want.len() {
        return f64::INFINITY;
    }
    // 非有限值不能走下面的 `f64::max`：NaN 会被 max 当缺席，漏报成"对上了"。逐对先定性：
    // 同是 NaN、同号 Inf 算对上，其余非有限组合一律判不一致。
    if !got.iter().zip(want).all(|(&g, &w)| nonfinite_match(g, w)) {
        return f64::INFINITY;
    }
    let max_want = want
        .iter()
        .fold(0.0f64, |acc, value| acc.max(f64::from(*value).abs()));
    let max_err = max_diff(got, want);
    if max_want == 0.0 {
        return max_err;
    }
    max_err / max_want
}

/// 非有限值配对是否一致；有限值交给数值误差比较。
fn nonfinite_match(got: f32, want: f32) -> bool {
    match (got.is_nan(), want.is_nan()) {
        (true, true) => true,
        (true, false) | (false, true) => false,
        // 非 NaN：同号 Inf 相等，或两边都是有限值；其余（Inf 对有限、异号 Inf）不一致。
        _ => got == want || (!got.is_infinite() && !want.is_infinite()),
    }
}

/// 一次比较的元素个数。
const WIDTH: usize = 8;

/// 逐元素绝对误差的最大值：8 路展开，尾部按剩余元素收口。
fn max_diff(got: &[f32], want: &[f32]) -> f64 {
    let mut lanes = [0.0f64; WIDTH];
    let block = got.len() - got.len() % WIDTH;
    for base in (0..block).step_by(WIDTH) {
        let a = &got[base..base + WIDTH];
        let b = &want[base..base + WIDTH];
        lanes[0] = lanes[0].max(diff(a[0], b[0]));
        lanes[1] = lanes[1].max(diff(a[1], b[1]));
        lanes[2] = lanes[2].max(diff(a[2], b[2]));
        lanes[3] = lanes[3].max(diff(a[3], b[3]));
        lanes[4] = lanes[4].max(diff(a[4], b[4]));
        lanes[5] = lanes[5].max(diff(a[5], b[5]));
        lanes[6] = lanes[6].max(diff(a[6], b[6]));
        lanes[7] = lanes[7].max(diff(a[7], b[7]));
    }
    let tail = got[block..]
        .iter()
        .zip(&want[block..])
        .fold(0.0f64, |acc, (a, b)| acc.max(diff(*a, *b)));
    lanes.into_iter().fold(tail, f64::max)
}

/// 一对元素的绝对误差，算在 `f64` 里：`f32` 相减在阈值附近会先丢掉要比较的那几位。
fn diff(got: f32, want: f32) -> f64 {
    (f64::from(got) - f64::from(want)).abs()
}

/// 按数据类型的阈值（README §11.1）：浮点按尾数宽度分档，整数与布尔位精确。
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
#[path = "../tests/reference_tests.rs"]
mod tests;
