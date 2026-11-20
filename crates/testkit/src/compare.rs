//! 数值对拍口径：逐元素绝对误差、绝对+相对双项容差、全局相对误差、有限值谓词与余弦方向一致性。
//!
//! 判据分四档，不要混用：同尺度逐元素对拍用 [`approx_eq`]；量化 logits 的方向一致性用
//! [`cosine_similarity`] + [`COSINE_GUARD`]；解析梯度 vs 数值梯度这类随量级缩放的比对，
//! 走 [`close`] 的「绝对 + 相对」双项预算，不套绝对容差；内核 vs 标量参考的**整批**对拍走
//! [`max_rel_err`]（与 `cutile::reference` 同一口径）——它只有一个全局分母，答的是「整批数
//! 合在一起差多少」，不是逐元素各算各的。
//!
//! 整批对拍失败时不要停在布尔值上：[`rel_err`] 回一份读数（误差、分母、最差元素），
//! [`assert_within_rel`] 直接把这三样写进 panic 消息，门禁用例应当用它。

/// 量化 logits 与 f32 参考的余弦回归下界（方向一致性，尺度不变）。
///
/// 合成基准与未来的真实样本集共用这一条阈值，避免各用例各抄一个数字、判据悄悄漂移。
pub const COSINE_GUARD: f32 = 0.999;

/// 逐个元素**绝对误差**比较：两套同尺度算法变体对拍走这一档。`NaN` 与 `NaN` 视为相等
/// （对拍时它两边同源，不该因为 `NaN != NaN` 失败）。
///
/// 两个**相同的无限大**也算相等：`inf - inf` 是 `NaN`，只看差值的判定会把「两边同为无穷」
/// 判成不等，而它与 `NaN` 对 `NaN` 是同一个理由。
///
/// 长度不同直接返回 `false`：对拍的首要条件是形状一致。
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
///
/// 两侧同为 NaN、同为同号无穷算相等；异号无穷不算相等。
fn close_scalar(a: f32, b: f32, abs: f32, rel: f32) -> bool {
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
///
/// 判据分三档：同尺度逐元素用 [`approx_eq`]；方向一致性用 [`cosine_similarity`]；
/// 量级相关的误差预算用本函数。长度不同直接返回 `false`。
pub fn close(left: &[f32], right: &[f32], abs: f32, rel: f32) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .all(|(a, b)| close_scalar(*a, *b, abs, rel))
}

/// 误差最大的那个元素：`index` 是它在切片里的下标，`got` / `want` 是两边的原值。
///
/// 对拍失败时先看这一个点：整批的判据是「一个分母下的最大误差」，它一定落在某个元素上，
/// 报出下标与两侧取值才能区分「某一路写错了」与「整体精度差一点」。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Worst {
    /// 元素在切片里的下标。
    pub index: usize,
    /// 被测侧的值。
    pub got: f32,
    /// 参考侧的值。
    pub want: f32,
}

/// 一次整批对拍的读数：[`RelErr::ratio`] 是判据本体，其余字段只为失败时定位。
///
/// 与 [`close`] 的逐元素预算不同，这里只有**一个**分母（`want` 的最大绝对值），所以它答的是
/// 「整批数合在一起差多少」这一个数，而不是逐元素各算各的——这正是内核对拍要的那条判据。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelErr {
    /// 最大绝对误差 `max|got − want|`（分母另存 [`RelErr::scale`]）。
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
    ///
    /// 长度不等或非有限值分类不一致时为 [`f64::INFINITY`]：那是「不是同一批数」，不是
    /// 「误差很大」，两者在报告里必须能分开。`want` 全零时退化为绝对误差：分母为零的相对
    /// 误差没有定义，报无穷会把「参考恰为零」误读成「对不上」。
    #[must_use]
    pub fn ratio(&self) -> f64 {
        if self.mismatched || self.scale == 0.0 {
            self.err
        } else {
            self.err / self.scale
        }
    }

    /// 判据是否在阈值内：`ratio() <= tolerance`。
    #[must_use]
    pub fn within(&self, tolerance: f32) -> bool {
        self.ratio() <= f64::from(tolerance)
    }
}

/// 分母策略：判据 `|got − want| / 分母` 里的「分母」按哪条规矩取。
///
/// 两条口径相差的**只有分母**，其余（长度、非有限值定性、最差元素定位）逐字相同：
///
/// - [`Scale::Batch`]：整批最大参考幅度 `max|want|`，与 `cutile::reference` 同一口径。
///   量级本身有意义时用它，但参考整体远小于 1 时会把绝对值上的小差放大成大相对误差。
/// - [`Scale::Unit`]：每元素 `max(|want|, 1)`，给「参考自然小于 1」的输出留 1 的绝对地板
///   （累加出来的、量级极小的注意力输出就是这一类）。它**不是**更松的那一版：参考幅度大于 1
///   时它对小元素反而更严。选哪条由被测输出的量级决定，不是由「想不想过」决定。
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
///
/// [`max_rel_err`] 只回一个数，是这里 [`RelErr::ratio`] 的投影；需要报「差在哪」时用本函数。
/// 分母按 [`Scale::Batch`] 取——这与 `cutile::reference` 是同一条判据。
#[must_use]
pub fn rel_err(got: &[f32], want: &[f32]) -> RelErr {
    rel_err_with(got, want, Scale::Batch)
}

/// 指定分母策略的整批读数；最差元素按该策略自己的判据取（即 `|got − want| / 分母` 最大者）。
#[must_use]
pub fn rel_err_with(got: &[f32], want: &[f32], scale: Scale) -> RelErr {
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
///
/// 分母为零（整批参考全零）时判据退化为绝对误差——与 [`RelErr::ratio`] 是同一处兜底。
fn element_reading(
    index: usize,
    got: f32,
    want: f32,
    denom: f64,
) -> Option<(f64, f64, f64, Worst)> {
    let diff = (f64::from(got) - f64::from(want)).abs();
    if diff == 0.0 {
        return None;
    }
    let value = if denom == 0.0 { diff } else { diff / denom };
    Some((value, diff, denom, Worst { index, got, want }))
}

/// 长度已校验后的累加：第一处非有限值不一致立刻定论，否则取判据值最大的元素。
///
/// 非有限值不能直接进累加：NaN 会被 `max` 当缺席，漏报成「对上了」，所以逐对先定性。
/// 对上号的两侧同为 NaN / 同号 Inf 时差值是 NaN，天然不计入误差也不占最差点。
fn tally(got: &[f32], want: &[f32], scale: Scale, batch: f64) -> RelErr {
    let mut report = RelErr {
        err: 0.0,
        scale: denominator(scale, batch, 0.0),
        worst: None,
        mismatched: false,
    };
    let mut best = f64::NEG_INFINITY;
    for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
        if !pairing_agrees(got, want) {
            report.err = f64::INFINITY;
            report.worst = Some(Worst { index, got, want });
            report.mismatched = true;
            return report;
        }
        let Some((value, diff, denom, worst)) =
            element_reading(index, got, want, denominator(scale, batch, want))
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
///
/// 放在夹具层是为了让跨层测试不必为了这一个数字去依赖 `cutile`；两条实现由
/// `tests/oracle.rs` 逐位钉死，改任一侧的口径都会在那里红。
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
///
/// 门禁用例应当用它而不是 `assert!(within_rel(..))`：断言只留下一句「false」，而数值对拍
/// 的失败几乎总要追问「第几个元素、差多少、是不是整体精度不够」。
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
            at.index, at.got, at.want
        );
    }
    format!(
        "{what} 对拍失败：max_rel_err={:.3e} > {tolerance:.3e}（第 {} 个元素 got={} want={}，\
         分母 {label}={:.3e}）",
        report.ratio(),
        at.index,
        at.got,
        at.want,
        report.scale,
        label = scale.label()
    )
}

/// 全局相对误差是否在阈值内：`max_rel_err(got, want) <= tolerance`。
///
/// [`max_rel_err`] 的布尔投影，与 `cutile::reference::within` 对应，但不绑定具体数据类型——
/// 阈值由调用方按精度自定。
#[must_use]
pub fn within_rel(got: &[f32], want: &[f32], tolerance: f32) -> bool {
    within_ratio(got, want, f64::from(tolerance))
}

/// 同一条判定的 `f64` 口径：精度分档表（`cutile::reference::tolerance`）以 `f64` 给出阈值，
/// 走这里就不必先降成 `f32` 再比。`within_rel` 是它的窄投影，两者不各算一遍。
#[must_use]
pub fn within_ratio(got: &[f32], want: &[f32], tolerance: f64) -> bool {
    rel_err(got, want).ratio() <= tolerance
}

/// 切片内全部为有限值（非 NaN、非无穷）；数值对拍前的前置谓词。
#[must_use]
pub fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

/// 两条同长向量的余弦相似度；长度不同、或任一向量范数为零时返回 [`None`]。
///
/// 回归判据用：量化前向的 logits 与 f32 参考比对方向一致性（不要求逐元素相等）。累加在 `f64`
/// 完成，避免长向量上 `f32` 平方和与点积的累加误差把相似度压低；返回值落在 `[-1, 1]`。
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
mod tests {
    use super::*;

    #[test]
    fn comparison_is_elementwise_and_length_sensitive() {
        assert!(approx_eq(&[1.0, 2.0], &[1.0, 2.0], 0.0));
        assert!(approx_eq(&[1.0, 2.0], &[1.0, 2.0004], 1e-3));
        assert!(!approx_eq(&[1.0, 2.0], &[1.0, 2.01], 1e-3));
        assert!(!approx_eq(&[1.0], &[1.0, 1.0], 1.0), "长度不同必须为假");
    }

    #[test]
    fn nan_matches_nan() {
        assert!(approx_eq(&[f32::NAN], &[f32::NAN], 0.0));
    }

    /// `inf - inf` 是 `NaN`：差值判定会把「两边同为无穷」判成不等，而它与 NaN 对 NaN 同理。
    #[test]
    fn equal_infinities_match() {
        for value in [f32::INFINITY, f32::NEG_INFINITY] {
            assert!(approx_eq(&[value], &[value], 0.0), "{value}");
        }
        assert!(!approx_eq(&[f32::INFINITY], &[f32::NEG_INFINITY], 1.0));
        assert!(!approx_eq(&[f32::INFINITY], &[1.0], 1.0));
    }

    #[test]
    fn close_allows_relative_slack_for_large_values() {
        assert!(close(&[1000.0], &[1000.5], 1e-3, 1e-3));
        assert!(!close(&[1000.0], &[1002.0], 1e-3, 1e-3));
        assert!(!close(&[1.0], &[1.0, 1.0], 0.0, 0.0), "长度不同必须为假");
    }

    #[test]
    fn close_nan_and_equal_infinites_match() {
        assert!(close(&[f32::NAN], &[f32::NAN], 0.0, 0.0));
        assert!(close(&[f32::INFINITY], &[f32::INFINITY], 0.0, 0.0));
        assert!(!close(&[f32::INFINITY], &[f32::NEG_INFINITY], 1.0, 1.0));
    }

    #[test]
    fn max_rel_err_uses_a_single_global_denominator() {
        assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        // 分母是 max|want| = 2.0，不是逐元素各自归一。
        let err = max_rel_err(&[1.0, 2.1], &[1.0, 2.0]);
        assert!((err - 0.05).abs() < 1e-6, "{err}");
        assert!(within_rel(&[1.0, 2.0], &[1.0, 2.0], 1e-5));
        assert!(!within_rel(&[1.0, 2.5], &[1.0, 2.0], 1e-5));
    }

    #[test]
    fn max_rel_err_is_infinity_on_mismatched_lengths_or_nonfinite_mismatch() {
        assert_eq!(max_rel_err(&[1.0], &[1.0, 1.0]), f64::INFINITY, "长度不同");
        assert_eq!(
            max_rel_err(&[f32::NAN], &[0.0]),
            f64::INFINITY,
            "NaN 对有限"
        );
        assert_eq!(
            max_rel_err(&[f32::INFINITY], &[1.0]),
            f64::INFINITY,
            "无穷对有限"
        );
        assert_eq!(
            max_rel_err(&[f32::INFINITY], &[f32::NEG_INFINITY]),
            f64::INFINITY,
            "异号无穷"
        );
    }

    #[test]
    fn max_rel_err_matches_nan_and_same_infinity() {
        assert_eq!(max_rel_err(&[f32::NAN], &[f32::NAN]), 0.0);
        assert_eq!(max_rel_err(&[f32::INFINITY], &[f32::INFINITY]), 0.0);
        assert_eq!(max_rel_err(&[f32::NEG_INFINITY], &[f32::NEG_INFINITY]), 0.0);
    }

    #[test]
    fn max_rel_err_falls_back_to_absolute_when_reference_is_zero() {
        // 参考全零：相对误差分母为零，退化为绝对误差，而不是误报「对不上」。
        assert_eq!(max_rel_err(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
        assert_eq!(max_rel_err(&[0.0, 2.0], &[0.0, 0.0]), 2.0);
    }

    #[test]
    fn rel_err_locates_the_worst_element() {
        let report = rel_err(&[1.0, 2.1, 3.0], &[1.0, 2.0, 3.0]);
        assert!(!report.mismatched);
        assert_eq!(report.scale, 3.0);
        let worst = report.worst.expect("0.1 的偏差落在第二个元素");
        assert_eq!((worst.index, worst.got, worst.want), (1, 2.1, 2.0));
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
        assert_eq!(report.worst, None, "没有偏差就没有最差点");
        assert_eq!(rel_err(&[], &[]).ratio(), 0.0, "空批视为对上");
    }

    #[test]
    fn rel_err_flags_length_and_nonfinite_breaks() {
        let short = rel_err(&[1.0], &[1.0, 1.0]);
        assert!(short.mismatched);
        assert_eq!(short.ratio(), f64::INFINITY);
        assert_eq!(short.worst, None, "长度不同没有可比的下标");

        let nan = rel_err(&[1.0, f32::NAN], &[1.0, 2.0]);
        assert!(nan.mismatched);
        let at = nan.worst.expect("NaN 与有限值不一致落在第二个元素");
        assert_eq!((at.index, at.want), (1, 2.0));
        assert!(at.got.is_nan(), "{}", at.got);

        assert!(
            !rel_err(&[f32::INFINITY], &[f32::INFINITY]).mismatched,
            "同号无穷算对上"
        );
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

    #[test]
    fn finite_predicate_rejects_nan_and_infinity() {
        assert!(all_finite(&[1.0, -2.0, 0.0]));
        assert!(!all_finite(&[f32::NAN]));
        assert!(!all_finite(&[f32::INFINITY]));
    }

    #[test]
    fn cosine_ranges_from_minus_one_to_one() {
        let same = [1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&same, &same), Some(1.0));
        assert_eq!(cosine_similarity(&same, &[-1.0, -2.0, -3.0]), Some(-1.0));
        let orthogonal = [1.0, 0.0];
        let other = [0.0, 1.0];
        assert_eq!(cosine_similarity(&orthogonal, &other), Some(0.0));
    }

    #[test]
    fn cosine_is_none_on_mismatched_or_zero_vectors() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 1.0]), None, "长度不同");
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]), None, "零向量");
        assert_eq!(
            cosine_similarity(&[1.0, 2.0], &[0.0, 0.0]),
            None,
            "另一向量为零"
        );
    }

    #[test]
    fn cosine_is_scale_invariant() {
        let left = [1.0, 2.0, 3.0];
        let right = [2.0, 4.0, 6.0];
        assert_eq!(
            cosine_similarity(&left, &right),
            Some(1.0),
            "同向缩放不改方向"
        );
    }
}
