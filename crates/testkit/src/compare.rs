//! 数值对拍的**逐元素与方向**判据：绝对误差、绝对+相对双项容差、有限值谓词与余弦方向一致性。

/// 量化 logits 与 f32 参考的余弦回归下界（方向一致性，尺度不变）。
pub const COSINE_GUARD: f32 = 0.999;

/// 逐个元素**绝对误差**比较：两套同尺度算法变体对拍走这一档。`NaN` 与 `NaN` 视为相等
/// （对拍时它两边同源，不该因为 `NaN != NaN` 失败）。
pub fn approx_eq(left: &[f32], right: &[f32], tolerance: f32) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter().zip(right).all(|(a, b)| {
        if a == b || (a.is_nan() && b.is_nan()) {
            return true;
        }
        (a - b).abs() <= tolerance
    })
}

/// 双项容差标量判定：`|a-b| ≤ abs + rel·max(|a|,|b|)`。
pub(crate) fn close_scalar(a: f32, b: f32, abs: f32, rel: f32) -> bool {
    if a == b {
        return true;
    }
    if a.is_nan() {
        return b.is_nan();
    }
    if a.is_infinite() {
        return false;
    }
    if b.is_infinite() {
        return false;
    }
    let tolerance = abs + rel * a.abs().max(b.abs());
    (a - b).abs() <= tolerance
}

/// 绝对 + 相对双项容差逐元素对拍：随量级缩放的比对（解析梯度 vs 数值梯度）走这一档。
pub fn close(left: &[f32], right: &[f32], abs: f32, rel: f32) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .all(|(a, b)| close_scalar(*a, *b, abs, rel))
}

/// 绝对 + 相对双项容差断言，相对项按**参考侧**（`want`）取分母：`|got−want| ≤ abs + rel·|want|`。
///
/// 与 [`close`] **不是同一条判据**，两者不可互换：`close` 的相对项取 `max(|got|,|want|)`
/// 的对称形式（"两边都可能有噪声"），本函数取单侧分母，与 `cutile::reference::max_rel_err`
/// 的 `max|got−want| / max|want|` 同一取向（"参考是标准答案"）。测试用例把两个公式各自钉住。
///
/// 失败时把首个越界元素的下标与两侧取值写进 panic 消息——只留一句 `false` 的断言
/// 在逐元素核上无法定位是哪一格写错。
///
/// # Panics
///
/// 长度不同或任一元素越界时 panic。
#[track_caller]
pub fn assert_close(left: &[f32], right: &[f32], abs: f32, rel: f32) {
    assert_eq!(
        left.len(),
        right.len(),
        "对拍两侧长度不同（got {}，want {}）",
        left.len(),
        right.len()
    );
    for (idx, (got, want)) in left.iter().zip(right).enumerate() {
        let tolerance = abs + rel * want.abs();
        assert!(
            (got - want).abs() <= tolerance,
            "第 {idx} 个元素 {got} vs {want}（容差 {tolerance}）"
        );
    }
}

/// 切片内全部为有限值（非 NaN、非无穷）；数值对拍前的前置谓词。
#[must_use]
pub fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

/// 两条同长向量的余弦相似度；长度不同、或任一向量范数为零时返回 [`None`]。
#[must_use]
pub fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() {
        return None;
    }
    let mut dot = 0.0f64;
    let mut left_sq = 0.0f64;
    let mut right_sq = 0.0f64;
    for (a, b) in left.iter().zip(right) {
        let a = f64::from(*a);
        let b = f64::from(*b);
        dot += a * b;
        left_sq += a * a;
        right_sq += b * b;
    }
    let denom = (left_sq * right_sq).sqrt();
    if denom == 0.0 {
        return None;
    }
    Some((dot / denom) as f32)
}

#[cfg(test)]
#[path = "../tests/compare_tests.rs"]
mod compare_tests;
