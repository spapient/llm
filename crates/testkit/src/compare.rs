//! 数值对拍的判据：逐元素 / 方向档与整批相对误差档同住一处。
//!
//! 两档共用同一条双项容差的标量判定 close_scalar：逐元素档答「这一格差多少」，
//! 整批档答「整批里最差的那格相对多大」。分成两个文件只会让同一条对拍口径长出两份分母，
//! 所以这里只留一份实现与一份阈值投影。
//!
//! 逐元素档（[`approx_eq`] / [`close`] / [`assert_close`] / [`all_finite`]）按 `T: Float` 泛型：
//! f32 与 f64 共用一条实现，各成员不必再按精度各抄一份 `close`。

use num_traits::Float;

/// 量化 logits 与 f32 参考的余弦回归下界（方向一致性，尺度不变）。
pub const COSINE_GUARD: f32 = 0.999;

/// 逐个元素**绝对误差**比较：两套同尺度算法变体对拍走这一档。`NaN` 与 `NaN` 视为相等
/// （对拍时它两边同源，不该因为 `NaN != NaN` 失败）。
pub fn approx_eq<T: Float>(left: &[T], right: &[T], tolerance: T) -> bool {
    left.len() == right.len()
        && first_failure(left, right, |a, b| {
            a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= tolerance
        })
        .is_none()
}

/// 返回首个不满足判据的位置；长度由调用方校验，扫描只做一次顺序遍历。
fn first_failure<T: Float>(
    left: &[T],
    right: &[T],
    accept: impl Fn(T, T) -> bool,
) -> Option<usize> {
    left.iter().zip(right).position(|(&a, &b)| !accept(a, b))
}

/// 双项容差标量判定：`|a-b| ≤ abs + rel·max(|a|,|b|)`。
pub(crate) fn close_scalar<T: Float>(a: T, b: T, abs: T, rel: T) -> bool {
    if a == b {
        return true;
    }
    if a.is_nan() {
        return b.is_nan();
    }
    if a.is_infinite() || b.is_infinite() {
        return false;
    }
    let tolerance = abs + rel * a.abs().max(b.abs());
    (a - b).abs() <= tolerance
}

/// 绝对 + 相对双项容差逐元素对拍：随量级缩放的比对（解析梯度 vs 数值梯度）走这一档。
pub fn close<T: Float>(left: &[T], right: &[T], abs: T, rel: T) -> bool {
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
pub fn assert_close<T: Float + std::fmt::Display>(left: &[T], right: &[T], abs: T, rel: T) {
    assert_eq!(
        left.len(),
        right.len(),
        "对拍两侧长度不同（got {}，want {}）",
        left.len(),
        right.len()
    );
    fail_first(
        left,
        right,
        |idx, got, want, tolerance| {
            assert!(
                idx >= left.len(),
                "第 {idx} 个元素 {got} vs {want}（容差 {tolerance}）"
            )
        },
        abs,
        rel,
    );
}

/// 单侧相对容差共用的定位内核；回调只在首个违约元素执行。
fn fail_first<T: Float>(
    left: &[T],
    right: &[T],
    fail: impl FnOnce(usize, T, T, T),
    abs: T,
    rel: T,
) {
    if let Some(idx) = first_failure(left, right, |got, want| {
        (got - want).abs() <= abs + rel * want.abs()
    }) {
        fail(idx, left[idx], right[idx], abs + rel * right[idx].abs());
    }
}

/// 切片内全部为有限值（非 NaN、非无穷）；数值对拍前的前置谓词。
#[must_use]
pub fn all_finite<T: Float>(values: &[T]) -> bool {
    values.iter().all(|value| value.is_finite())
}

/// 标量对拍：`|got − want| ≤ tol · max(1, |want|)`。
///
/// 相对项以参考侧 `want` 为分母、并留绝对地板 1（零附近的量按绝对容差判），`what` 只进失败消息。
/// 与 [`assert_close`] 的双项形式不同：这里单参数 `tol` 同时承担相对与绝对两档，适合性质测试的
/// 「不变量应在这条等式附近」断言。
///
/// # Panics
///
/// 判据不成立时 panic，消息带 `what` 与分母读数。
#[track_caller]
pub fn assert_near<T: Float + std::fmt::Display>(got: T, want: T, tol: T, what: &str) {
    let scale = want.abs().max(T::one());
    assert!(
        (got - want).abs() <= tol * scale,
        "{what}: got {got}, want {want}（tol = {tol}，scale = {scale}）"
    );
}

/// 逐元素对拍，判据同 [`assert_near`]：`|got − want| ≤ tol · max(1, |want|)`。
///
/// 长度不等即失败（形状不对就不是同一批数）；失败消息带首个越界下标与 `what`。
///
/// # Panics
///
/// 长度不同或任一元素越界时 panic。
#[track_caller]
pub fn assert_vec_near<T: Float + std::fmt::Display>(got: &[T], want: &[T], tol: T, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: 长度不等");
    let Some(idx) = first_failure(got, want, |g, w| {
        (g - w).abs() <= tol * w.abs().max(T::one())
    }) else {
        return;
    };
    assert!(
        idx >= got.len(),
        "{what}: 第 {idx} 项 got {}, want {}（tol = {tol}）",
        got[idx],
        want[idx]
    );
}

/// 两条同长向量的余弦相似度；长度不同、或任一向量范数为零时返回 [`None`]。
#[must_use]
pub fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() {
        return None;
    }
    let (dot, left_sq, right_sq) = left.iter().zip(right).fold(
        (0.0f64, 0.0f64, 0.0f64),
        |(dot, left_sq, right_sq), (a, b)| {
            let a = f64::from(*a);
            let b = f64::from(*b);
            (dot + a * b, left_sq + a * a, right_sq + b * b)
        },
    );
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
    /// 整批平均相对误差 `mean-RE`：每元素同口径相对误差的算术平均，控整体精度；与 [`Self::ratio`]
    /// 的离群口径并列双门。空批为 `0.0`，长度不符或非有限不一致时为无穷大。
    pub mean: f64,
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
        mean: f64::INFINITY,
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

/// 非有限值是否配对：两边都有限时定性通过；否则要求与零容差的 close_scalar 一致
/// （NaN 对 NaN、同号 Inf 算对上，其余组合一律判不一致）。
fn pairing_agrees(got: f32, want: f32) -> bool {
    if got.is_finite() && want.is_finite() {
        return true;
    }
    close_scalar(got, want, 0.0, 0.0)
}

/// 单元素读数 `(判据值, 绝对误差, 分母, 该元素)`；逐元素相等时为 [`None`]（不占最差点）。
fn element_reading(idx: usize, got: f32, want: f32, denom: f64) -> Option<(f64, f64, f64, Worst)> {
    if !got.is_finite() || !want.is_finite() {
        return None;
    }
    let diff = (f64::from(got) - f64::from(want)).abs();
    if diff == 0.0 {
        return None;
    }
    let value = if denom == 0.0 { diff } else { diff / denom };
    Some((value, diff, denom, Worst { idx, got, want }))
}

/// 长度已校验后的累加：第一处非有限值不一致立刻定论，否则取判据值最大的元素，并累加整批平均。
fn tally(got: &[f32], want: &[f32], scale: Scale, batch: f64) -> RelErr {
    let mut report = RelErr {
        err: 0.0,
        scale: denominator(scale, batch, 0.0),
        mean: 0.0,
        worst: None,
        mismatched: false,
    };
    let mut best = f64::NEG_INFINITY;
    let mut sum = 0.0f64;
    for (idx, (&got, &want)) in got.iter().zip(want).enumerate() {
        if !pairing_agrees(got, want) {
            report.err = f64::INFINITY;
            report.mean = f64::INFINITY;
            report.worst = Some(Worst { idx, got, want });
            report.mismatched = true;
            return report;
        }
        let Some((value, diff, denom, worst)) =
            element_reading(idx, got, want, denominator(scale, batch, want))
        else {
            continue;
        };
        sum += value;
        if value > best {
            best = value;
            report.err = diff;
            report.scale = denom;
            report.worst = Some(worst);
        }
    }
    report.mean = mean_per_element(sum, got.len());
    report
}

/// 整批平均相对误差：空批记 0，否则按元素数把累加值归一。
fn mean_per_element(sum: f64, len: usize) -> f64 {
    if len == 0 { 0.0 } else { sum / len as f64 }
}

/// 全局相对误差：`max|got − want| / max|want|`，与 `cutile::reference::max_rel_err` 同一口径。
#[must_use]
pub fn max_rel_err(got: &[f32], want: &[f32]) -> f64 {
    rel_err(got, want).ratio()
}

/// 整批平均相对误差（`mean-RE`）：每元素同口径相对误差的算术平均，控整体精度；与 [`max_rel_err`]
/// 的离群口径并列双门。分母策略与 [`rel_err_with`] 一致；长度不符或非有限不一致时为无穷大。
#[must_use]
pub fn mean_rel_err(got: &[f32], want: &[f32]) -> f64 {
    rel_err(got, want).mean
}

/// 一次对拍的双门读数：整体平均误差与离群最大误差，同一次遍历得出，不重复扫整批。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelGate {
    /// `mean-RE`：整批平均相对误差，控整体精度。
    pub mean: f64,
    /// `max-RE`：整批最大相对误差，控孤立离群。
    pub max: f64,
}

impl RelGate {
    /// 两门是否都在各自阈值内：`mean ≤ mean_tol` 且 `max ≤ max_tol`。
    #[must_use]
    pub fn passes(self, mean_tol: f64, max_tol: f64) -> bool {
        self.mean <= mean_tol && self.max <= max_tol
    }
}

/// 一次遍历同时出双门读数（[`Scale::Batch`] 口径）；长度不符或非有限不一致时两门均为无穷大。
#[must_use]
pub fn rel_gate(got: &[f32], want: &[f32]) -> RelGate {
    let report = rel_err(got, want);
    RelGate {
        mean: report.mean,
        max: report.ratio(),
    }
}

/// 双门断言：`mean-RE ≤ mean_tol` 且 `max-RE ≤ max_tol`。
///
/// max 门可按约定倍数放宽（`max_tol` 由调用方显式传入），避免单个孤立离群点拖垮整张量；
/// 两门阈值分别对照，失败消息写清是哪一门超了多少。长度不等或非有限值不一致时读数为无穷大，
/// 必然判失败。
#[track_caller]
pub fn assert_rel_gate(got: &[f32], want: &[f32], mean_tol: f64, max_tol: f64, what: &str) {
    let gate = rel_gate(got, want);
    assert!(
        gate.passes(mean_tol, max_tol),
        "{what} 双门对拍失败：mean-RE={:.3e}（门 {mean_tol:.3e}），max-RE={:.3e}（门 {max_tol:.3e}）",
        gate.mean,
        gate.max
    );
}

/// 指定分母策略的平均相对误差；与 [`mean_rel_err`] 的关系同 [`rel_err_with`] 之于 [`rel_err`]。
#[must_use]
pub fn mean_rel_err_with(got: &[f32], want: &[f32], scale: Scale) -> f64 {
    rel_err_with(got, want, scale).mean
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
mod compare_tests {
    //! 对拍判据自测：逐元素容差、方向、整批/单位分母与生产参考实现交叉对拍。

    use std::sync::LazyLock;

    use super::*;
    use cutile::atom::DType;
    use cutile::reference;

    fn check_approx(got: &[f32], want: &[f32], tolerance: f32, expected: bool) {
        assert_eq!(approx_eq(got, want, tolerance), expected);
    }

    fn check_close(got: &[f32], want: &[f32], abs: f32, rel: f32, expected: bool) {
        assert_eq!(close(got, want, abs, rel), expected);
    }

    fn check_cosine(got: &[f32], want: &[f32], expected: Option<f32>) {
        assert_eq!(cosine_similarity(got, want), expected);
    }

    crate::test_cases!(predicates {
        comparison_is_elementwise_and_length_sensitive => || {
            check_approx(&[1.0, 2.0], &[1.0, 2.0], 0.0, true);
            check_approx(&[1.0, 2.0], &[1.0, 2.0004], 1e-3, true);
            check_approx(&[1.0, 2.0], &[1.0, 2.01], 1e-3, false);
            check_approx(&[1.0], &[1.0, 1.0], 1.0, false);
        },
        nan_matches_nan => || check_approx(&[f32::NAN], &[f32::NAN], 0.0, true),
        equal_infinities_match => || {
            for value in [f32::INFINITY, f32::NEG_INFINITY] {
                check_approx(&[value], &[value], 0.0, true);
            }
            check_approx(&[f32::INFINITY], &[f32::NEG_INFINITY], 1.0, false);
            check_approx(&[f32::INFINITY], &[1.0], 1.0, false);
        },
        close_allows_relative_slack_for_large_values => || {
            check_close(&[1000.0], &[1000.5], 1e-3, 1e-3, true);
            check_close(&[1000.0], &[1002.0], 1e-3, 1e-3, false);
            check_close(&[1.0], &[1.0, 1.0], 0.0, 0.0, false);
        },
        close_nan_and_equal_infinites_match => || {
            check_close(&[f32::NAN], &[f32::NAN], 0.0, 0.0, true);
            check_close(&[f32::INFINITY], &[f32::INFINITY], 0.0, 0.0, true);
            check_close(&[f32::INFINITY], &[f32::NEG_INFINITY], 1.0, 1.0, false);
        },
        finite_predicate_rejects_nan_and_infinity => || {
            assert!(all_finite(&[1.0, -2.0, 0.0]));
            assert!(!all_finite(&[f32::NAN]));
            assert!(!all_finite(&[f32::INFINITY]));
        },
        cosine_ranges_from_minus_one_to_one => || {
            check_cosine(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], Some(1.0));
            check_cosine(&[1.0, 2.0, 3.0], &[-1.0, -2.0, -3.0], Some(-1.0));
            check_cosine(&[1.0, 0.0], &[0.0, 1.0], Some(0.0));
        },
        cosine_is_none_on_mismatched_or_zero_vectors => || {
            check_cosine(&[1.0], &[1.0, 1.0], None);
            check_cosine(&[0.0, 0.0], &[1.0, 2.0], None);
            check_cosine(&[1.0, 2.0], &[0.0, 0.0], None);
        },
        cosine_is_scale_invariant => || {
            check_cosine(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0], Some(1.0));
        },
    }, |case: fn()| case());

    /// 对称分母与参考侧分母不可互换；这条钉住二者的语义差异。
    #[test]
    #[should_panic(expected = "容差")]
    fn the_asymmetric_relative_budget_is_stricter_than_the_symmetric_one() {
        let (got, want, abs, rel) = (1.5f32, 1.0f32, 0.0f32, 0.4f32);
        assert!(close(&[got], &[want], abs, rel), "对称分母放行");
        assert_close(&[got], &[want], abs, rel);
    }

    #[test]
    fn max_rel_err_uses_a_single_global_denominator() {
        assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        let err = max_rel_err(&[1.0, 2.1], &[1.0, 2.0]);
        assert!((err - 0.05).abs() < 1e-6, "{err}");
        assert!(within_rel(&[1.0, 2.0], &[1.0, 2.0], 1e-5));
        assert!(!within_rel(&[1.0, 2.5], &[1.0, 2.0], 1e-5));
    }

    #[test]
    fn max_rel_err_is_infinity_on_mismatched_lengths_or_nonfinite_mismatch() {
        assert_eq!(max_rel_err(&[1.0], &[1.0, 1.0]), f64::INFINITY);
        assert_eq!(max_rel_err(&[f32::NAN], &[0.0]), f64::INFINITY);
        assert_eq!(max_rel_err(&[f32::INFINITY], &[1.0]), f64::INFINITY);
        assert_eq!(
            max_rel_err(&[f32::INFINITY], &[f32::NEG_INFINITY]),
            f64::INFINITY
        );
    }

    #[test]
    fn max_rel_err_matches_nan_and_same_infinity() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let pair = [value];
            assert_eq!(max_rel_err(&pair, &pair), 0.0);
            assert_eq!(mean_rel_err(&pair, &pair), 0.0);
            assert_eq!(
                rel_gate(&pair, &pair),
                RelGate {
                    mean: 0.0,
                    max: 0.0
                }
            );
        }
    }

    #[test]
    fn max_rel_err_falls_back_to_absolute_when_reference_is_zero() {
        assert_eq!(max_rel_err(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
        assert_eq!(max_rel_err(&[0.0, 2.0], &[0.0, 0.0]), 2.0);
    }

    #[test]
    fn rel_err_locates_the_worst_element() {
        let report = rel_err(&[1.0, 2.1, 3.0], &[1.0, 2.0, 3.0]);
        assert!(!report.mismatched);
        assert_eq!(report.scale, 3.0);
        let worst = report.worst.expect("0.1 的偏差落在第二个元素");
        assert_eq!((worst.idx, worst.got, worst.want), (1, 2.1, 2.0));
        assert_eq!(
            report.ratio(),
            max_rel_err(&[1.0, 2.1, 3.0], &[1.0, 2.0, 3.0])
        );
        assert!(
            (report.ratio() - 0.1 / 3.0).abs() < 1e-6,
            "{}",
            report.ratio()
        );
    }

    #[test]
    fn rel_err_is_zero_and_unlocated_when_the_batches_match() {
        let report = rel_err(&[1.0, 2.0], &[1.0, 2.0]);
        assert_eq!((report.err, report.mismatched), (0.0, false));
        assert_eq!(report.worst, None);
        assert_eq!(rel_err(&[], &[]).ratio(), 0.0, "空批视为对上");
    }

    #[test]
    fn rel_err_flags_length_and_nonfinite_breaks() {
        let short = rel_err(&[1.0], &[1.0, 1.0]);
        assert!(short.mismatched);
        assert_eq!(short.ratio(), f64::INFINITY);
        assert_eq!(short.worst, None);
        let nan = rel_err(&[1.0, f32::NAN], &[1.0, 2.0]);
        assert!(nan.mismatched);
        let at = nan.worst.expect("NaN 与有限值不一致落在第二个元素");
        assert_eq!((at.idx, at.want), (1, 2.0));
        assert!(at.got.is_nan());
        assert!(!rel_err(&[f32::INFINITY], &[f32::INFINITY]).mismatched);
    }

    #[test]
    fn assert_within_rel_passes_inside_the_tolerance() {
        assert_within_rel(&[1.0, 2.0], &[1.0, 2.0], 0.0, "同一批数");
        assert_within_rel(&[1.0, 2.0], &[1.0, 2.000_004], 1e-5, "误差在阈值内");
    }

    #[test]
    #[should_panic(expected = "第 1 个元素 got=2.5 want=2")]
    fn assert_within_rel_names_the_offending_element() {
        assert_within_rel(&[1.0, 2.5], &[1.0, 2.0], 1e-3, "GEMM");
    }

    #[test]
    #[should_panic(expected = "长度不同")]
    fn assert_within_rel_reports_a_length_mismatch() {
        assert_within_rel(&[1.0], &[1.0, 2.0], 1e-3, "行批");
    }

    #[test]
    #[should_panic(expected = "第 0 个元素非有限值不一致")]
    fn assert_within_rel_reports_a_nonfinite_break() {
        assert_within_rel(&[f32::NAN], &[1.0], 1e-3, "注意力");
    }

    /// 同一份静态样本矩阵驱动所有交叉对拍，避免每条判据各建一份语料。
    static SAMPLES: LazyLock<[Vec<f32>; 10]> = LazyLock::new(|| {
        [
            vec![],
            vec![0.0],
            vec![0.0, 0.0],
            vec![1.0, -2.0, 3.0],
            vec![1e-30, 3.0],
            vec![f32::MAX, 1.0],
            vec![f32::NAN, 1.0],
            vec![f32::INFINITY, 1.0],
            vec![f32::NEG_INFINITY, f32::INFINITY],
            (0..9).map(|i| i as f32 * 0.25 - 1.0).collect(),
        ]
    });

    fn for_each_sample(mut check: impl FnMut(&[f32], &[f32], usize, usize)) {
        for (left, got) in SAMPLES.iter().enumerate() {
            for (right, want) in SAMPLES.iter().enumerate() {
                check(got, want, left, right);
            }
        }
    }

    #[test]
    fn the_two_oracles_agree_bit_for_bit() {
        for_each_sample(|got, want, left, right| {
            assert_eq!(
                max_rel_err(got, want).to_bits(),
                reference::max_rel_err(got, want).to_bits(),
                "样本 {left}×{right} 的判据读数不同"
            );
            assert_eq!(
                rel_err(got, want).ratio().to_bits(),
                max_rel_err(got, want).to_bits()
            );
        });
    }

    #[test]
    fn the_tolerance_ladder_is_shared_not_copied() {
        for_each_sample(|got, want, left, right| {
            for dtype in [DType::F32, DType::F64, DType::I8, DType::I4] {
                assert_eq!(
                    within_ratio(got, want, reference::tolerance(dtype)),
                    reference::within(got, want, dtype),
                    "{dtype:?} 分档下样本 {left}×{right} 的判定不同"
                );
            }
        });
    }

    #[test]
    fn the_worst_element_attains_the_reported_error() {
        let report = rel_err(&[1.0, 2.5, -3.0, 4.0], &[1.0, 2.0, 3.0, 4.0]);
        let at = report.worst.expect("有偏差就有最差点");
        assert_eq!(at.idx, 2);
        assert_eq!((at.got, at.want), (-3.0, 3.0));
        assert_eq!(report.err, f64::from((at.got - at.want).abs()));
        assert_eq!(report.scale, 4.0);
    }

    #[test]
    fn the_two_denominators_differ_only_in_the_floor() {
        let (got, want) = ([0.00011, 0.0002], [0.0001, 0.0002]);
        let batch = rel_err(&got, &want);
        let unit = rel_err_with(&got, &want, Scale::Unit);
        assert_eq!(batch.worst.map(|worst| worst.idx), Some(0));
        assert_eq!(unit.worst.map(|worst| worst.idx), Some(0));
        assert_eq!(batch.scale, f64::from(0.0002f32));
        assert_eq!(unit.scale, 1.0);
        assert_eq!(unit.ratio(), f64::from((0.00011f32 - 0.0001f32).abs()));
        assert!(batch.ratio() > unit.ratio() * 100.0);
        let (got, want) = ([1.5, 100.0], [1.0, 100.0]);
        assert!(
            rel_err_with(&got, &want, Scale::Unit).ratio() > rel_err(&got, &want).ratio() * 10.0
        );
    }

    crate::test_cases!(gates {
        mean_rel_err_is_zero_on_matching_batches => || {
            assert_eq!(mean_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
            assert_eq!(mean_rel_err(&[], &[]), 0.0, "空批视为对上");
        },
        mean_rel_err_averages_per_element_on_the_same_denominator => || {
            let err = mean_rel_err(&[1.0, 2.1], &[1.0, 2.0]);
            assert!((err - 0.025).abs() < 1e-6, "{err}");
            let max = max_rel_err(&[1.0, 2.1], &[1.0, 2.0]);
            assert!((max - 0.05).abs() < 1e-6, "max 门仍是最差元素：{max}");
        },
        mean_and_max_separate_whole_batch_drift_from_an_isolated_outlier => || {
            let mean = mean_rel_err(&[1.0, 2.5], &[1.0, 2.0]);
            let max = max_rel_err(&[1.0, 2.5], &[1.0, 2.0]);
            assert!((mean - 0.125).abs() < 1e-6, "{mean}");
            assert!((max - 0.25).abs() < 1e-6, "{max}");
            assert!(max > mean);
        },
        mean_rel_err_follows_the_unit_denominator_floor => || {
            let mean = mean_rel_err_with(&[0.00011, 0.0002], &[0.0001, 0.0002], Scale::Unit);
            assert!((mean - 0.000005).abs() < 1e-9, "{mean}");
        },
        mean_rel_err_is_infinity_on_mismatch_and_falls_back_to_absolute_on_zero_reference => || {
            assert_eq!(mean_rel_err(&[1.0], &[1.0, 1.0]), f64::INFINITY);
            assert_eq!(mean_rel_err(&[f32::NAN], &[0.0]), f64::INFINITY);
            assert_eq!(mean_rel_err(&[0.0, 2.0], &[0.0, 0.0]), 1.0);
        },
        rel_gate_reports_mean_and_max_from_one_pass => || {
            let gate = rel_gate(&[1.0, 2.5, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.0]);
            assert!((gate.mean - 0.03125).abs() < 1e-9, "{}", gate.mean);
            assert!((gate.max - 0.125).abs() < 1e-9, "{}", gate.max);
        },
        rel_gate_separates_a_low_mean_from_a_loose_max_gate => || {
            let gate = rel_gate(&[1.0, 2.5, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.0]);
            assert!(gate.passes(0.05, 0.2));
            assert!(!gate.passes(0.05, 0.1));
        },
        rel_gate_fails_when_mean_drifts_across_the_strict_gate => || {
            let gate = rel_gate(&[1.1, 2.1, 3.1, 4.1], &[1.0, 2.0, 3.0, 4.0]);
            assert!(gate.max <= 0.03);
            assert!(!gate.passes(0.02, 0.03));
        },
        rel_gate_is_infinity_on_mismatched_lengths => || {
            let gate = rel_gate(&[1.0], &[1.0, 1.0]);
            assert_eq!((gate.mean, gate.max), (f64::INFINITY, f64::INFINITY));
            assert!(!gate.passes(1.0, 1.0));
        },
    }, |case: fn()| case());

    #[test]
    #[should_panic(expected = "双门对拍失败")]
    fn assert_rel_gate_panics_when_either_gate_fails() {
        assert_rel_gate(
            &[1.0, 2.5, 3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
            0.05,
            0.1,
            "GEMM 双门",
        );
    }
}

// ---- 标量参考数学 -----------------------------------------------------------

/// 两条同长向量的点积：`Σ aᵢ·bᵢ`。
///
/// # Panics
///
/// `left.len() != right.len()` 时 panic；长度不符是测试用例自身的错误。
#[must_use]
pub fn dot<T: Float>(left: &[T], right: &[T]) -> T {
    assert_eq!(
        left.len(),
        right.len(),
        "点积两侧长度不同（left {}，right {}）",
        left.len(),
        right.len()
    );
    left.iter()
        .zip(right)
        .fold(T::zero(), |acc, (a, b)| acc + *a * *b)
}

/// 一维信号的均值；空切片按参考约定返回零。
#[must_use]
pub fn mean<T: Float>(values: &[T]) -> T {
    if values.is_empty() {
        return T::zero();
    }
    values.iter().fold(T::zero(), |acc, v| acc + *v) / len_scalar(values.len())
}

/// 一维信号相对给定均值的方差；空切片按参考约定返回零。
#[must_use]
pub fn variance<T: Float>(values: &[T], mean: T) -> T {
    if values.is_empty() {
        return T::zero();
    }
    values.iter().fold(T::zero(), |acc, v| {
        let d = *v - mean;
        acc + d * d
    }) / len_scalar(values.len())
}

/// 浮点类型无法表示长度时以 NaN 暴露异常，而不是 panic 或静默给出零分母。
fn len_scalar<T: Float>(len: usize) -> T {
    T::from(len).unwrap_or_else(T::nan)
}

#[cfg(test)]
mod reference_math_tests {
    use super::{dot, mean, variance};

    #[test]
    fn dot_accumulates_pairwise_products() {
        assert_eq!(dot(&[1.0f32, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        assert_eq!(dot(&[0.5f64, -1.0], &[2.0, 3.0]), -2.0);
        assert_eq!(dot::<f32>(&[], &[]), 0.0, "空向量点积为零");
    }

    #[test]
    #[should_panic(expected = "长度不同")]
    fn dot_rejects_mismatched_lengths() {
        let _ = dot(&[1.0f32], &[1.0, 2.0]);
    }

    #[test]
    fn mean_and_variance_match_the_textbook_formulas() {
        let values = [1.0f64, 2.0, 3.0, 4.0];
        let mu = mean(&values);
        assert_eq!(mu, 2.5);
        assert_eq!(variance(&values, mu), 1.25);
        assert_eq!(mean::<f64>(&[]), 0.0);
        assert_eq!(variance::<f64>(&[], 0.0), 0.0);
    }
}
