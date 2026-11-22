//! 数值对拍的判据：逐元素 / 方向档与整批相对误差档同住一处。
//!
//! 两档共用同一条双项容差的标量判定 [`close_scalar`]：逐元素档答「这一格差多少」，
//! 整批档答「整批里最差的那格相对多大」。分成两个文件只会让同一条对拍口径长出两份分母，
//! 所以这里只留一份实现与一份阈值投影。

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

/// 误差最大的那个元素：`idx` 是它在切片里的下标，`got` / `want` 是两边的原值。
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Worst {
    /// 元素在切片里的下标。
    pub idx: usize,
    /// 被测侧的值。
    pub got: f32,
    /// 参考侧的值。
    pub want: f32,
}

/// 一次整批对拍的读数：`ratio` 是判据本体，其余字段只为失败时定位。
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RelErr {
    /// 最大绝对误差 `max|got − want|`（分母另存 `scale`）。
    pub err: f64,
    /// 分母：误差相对于这个数是多少。整批口径下是整批最大参考幅度 `max|want|`（`want` 全零时
    /// 为 `0.0`），单位口径下是最差元素自己的 `max(|want|, 1)`。
    pub scale: f64,
    /// 误差最大的元素；整批逐元素相等（或无元素）时为 [`None`]。
    pub worst: Option<Worst>,
    /// 长度不等，或某个下标的非有限值分类不一致——这时 `err` 是无穷大且 `worst` 指向第一处。
    pub mismatched: bool,
}

impl RelErr {
    /// 判据本体：`err / scale`，即 `max|got − want| / max|want|`。
    #[must_use]
    pub fn ratio(&self) -> f64 {
        if self.mismatched || self.scale == 0.0 {
            self.err
        } else {
            self.err / self.scale
        }
    }
}

/// 分母策略：判据 `|got − want| / 分母` 里的「分母」按哪条规矩取。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    /// 整批最大参考幅度。
    Batch,
    /// 每元素幅度，且地板为 1。
    Unit,
}

impl Scale {
    /// 失败的读数里要写清用的是哪条分母，否则「差 0.5」会被读成同一个数。
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Batch => "整批 max|want|",
            Self::Unit => "元素 max(|want|, 1)",
        }
    }
}

/// 整批对拍的读数：最大误差、分母与它落在哪个元素上。
#[must_use]
pub(crate) fn rel_err(got: &[f32], want: &[f32]) -> RelErr {
    rel_err_with(got, want, Scale::Batch)
}

/// 指定分母策略的整批读数；最差元素按该策略自己的判据取（即 `|got − want| / 分母` 最大者）。
#[must_use]
pub(crate) fn rel_err_with(got: &[f32], want: &[f32], scale: Scale) -> RelErr {
    if got.len() != want.len() {
        return mismatched();
    }
    tally(got, want, scale, batch_magnitude(want))
}

/// 长度不等时的读数：`ratio()` 为无穷大且不指向任何元素——那是「不是同一批数」。
fn mismatched() -> RelErr {
    RelErr {
        err: f64::INFINITY,
        scale: 0.0,
        worst: None,
        mismatched: true,
    }
}

/// 整批参考幅度 `max|want|`；[`Scale::Batch`] 与 `cutile::reference` 的分母都取自它。
fn batch_magnitude(want: &[f32]) -> f64 {
    want.iter()
        .fold(0.0f64, |acc, value| acc.max(f64::from(*value).abs()))
}

/// 某元素自己的分母：整批口径取整批幅度，单位口径取 `max(|want|, 1)` 的绝对地板。
fn denominator(scale: Scale, batch: f64, want: f32) -> f64 {
    match scale {
        Scale::Batch => batch,
        Scale::Unit => f64::from(want.abs().max(1.0f32)),
    }
}

/// 非有限值是否配对：两边都有限时定性通过；否则要求与零容差的 [`close_scalar`] 一致
/// （NaN 对 NaN、同号 Inf 算对上，其余组合一律判不一致）。
fn pairing_agrees(got: f32, want: f32) -> bool {
    if got.is_finite() && want.is_finite() {
        return true;
    }
    close_scalar(got, want, 0.0, 0.0)
}

/// 单元素读数 `(判据值, 绝对误差, 分母, 该元素)`；逐元素相等时为 [`None`]（不占最差点）。
fn element_reading(idx: usize, got: f32, want: f32, denom: f64) -> Option<(f64, f64, f64, Worst)> {
    let diff = (f64::from(got) - f64::from(want)).abs();
    if diff == 0.0 {
        return None;
    }
    let value = if denom == 0.0 { diff } else { diff / denom };
    Some((value, diff, denom, Worst { idx, got, want }))
}

/// 长度已校验后的累加：第一处非有限值不一致立刻定论，否则取判据值最大的元素。
fn tally(got: &[f32], want: &[f32], scale: Scale, batch: f64) -> RelErr {
    let mut report = RelErr {
        err: 0.0,
        scale: denominator(scale, batch, 0.0),
        worst: None,
        mismatched: false,
    };
    let mut best = f64::NEG_INFINITY;
    for (idx, (&got, &want)) in got.iter().zip(want).enumerate() {
        if !pairing_agrees(got, want) {
            report.err = f64::INFINITY;
            report.worst = Some(Worst { idx, got, want });
            report.mismatched = true;
            return report;
        }
        let Some((value, diff, denom, worst)) =
            element_reading(idx, got, want, denominator(scale, batch, want))
        else {
            continue;
        };
        if value > best {
            best = value;
            report.err = diff;
            report.scale = denom;
            report.worst = Some(worst);
        }
    }
    report
}

/// 全局相对误差：`max|got − want| / max|want|`，与 `cutile::reference::max_rel_err` 同一口径。
#[must_use]
pub fn max_rel_err(got: &[f32], want: &[f32]) -> f64 {
    rel_err(got, want).ratio()
}

/// 整批对拍不通过时 panic；分母取 [`Scale::Batch`]，阈值给 `f32`。
#[track_caller]
pub fn assert_within_rel(got: &[f32], want: &[f32], tolerance: f32, what: &str) {
    assert_rel_with(Scale::Batch, got, want, f64::from(tolerance), what);
}

/// 指定分母策略的对拍断言：不通过时 panic，并把「差在哪」写进消息——判据读数、最差元素的
/// 标号与两侧取值、以及用的是哪条分母。通过时什么也不做。
#[track_caller]
pub fn assert_rel_with(scale: Scale, got: &[f32], want: &[f32], tolerance: f64, what: &str) {
    let report = rel_err_with(got, want, scale);
    assert!(
        report.ratio() <= tolerance,
        "{}",
        complaint(&report, got, want, scale, tolerance, what)
    );
}

/// 失败消息：先把「不是同一批数」的两类（长度不等、非有限值不一致）分开，再写判据读数、最差元素
/// 与用的是哪条分母——三者缺一都会让失败现场需要重新跑一遍。
fn complaint(
    report: &RelErr,
    got: &[f32],
    want: &[f32],
    scale: Scale,
    tolerance: f64,
    what: &str,
) -> String {
    if got.len() != want.len() {
        return format!(
            "{what} 对拍失败：长度不同（got {}，want {}）",
            got.len(),
            want.len()
        );
    }
    let Some(at) = report.worst else {
        return format!(
            "{what} 对拍失败：max_rel_err={:.3e} > {tolerance:.3e}",
            report.ratio()
        );
    };
    if report.mismatched {
        return format!(
            "{what} 对拍失败：第 {} 个元素非有限值不一致（got {:?}，want {:?}），max_rel_err=inf",
            at.idx, at.got, at.want
        );
    }
    format!(
        "{what} 对拍失败：max_rel_err={:.3e} > {tolerance:.3e}（第 {} 个元素 got={} want={}，\
         分母 {label}={:.3e}）",
        report.ratio(),
        at.idx,
        at.got,
        at.want,
        report.scale,
        label = scale.label()
    )
}

/// 全局相对误差是否在阈值内：`max|got − want| / max|want| <= tolerance`（`f32` 口径）。
///
/// 只服务本 crate 的对拍用例；外部一律走 [`assert_within_rel`] / `prop::assert_within_rel`。
#[cfg(test)]
#[must_use]
pub(crate) fn within_rel(got: &[f32], want: &[f32], tolerance: f32) -> bool {
    within_ratio(got, want, f64::from(tolerance))
}

/// 同一条判定的 `f64` 口径：精度分档表（`cutile::reference::tolerance`）以 `f64` 给出阈值，
/// 走这里就不必先降成 `f32` 再比。`within_rel` 是它的窄投影，两者不各算一遍。
#[cfg(test)]
#[must_use]
pub(crate) fn within_ratio(got: &[f32], want: &[f32], tolerance: f64) -> bool {
    rel_err(got, want).ratio() <= tolerance
}

#[cfg(test)]
#[path = "../tests/compare_tests.rs"]
mod compare_tests;

#[cfg(test)]
#[path = "../tests/rel_tests.rs"]
mod rel_tests;
