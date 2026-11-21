//! 整批相对误差口径：一个全局分母下的最大误差、失败读数与阈值投影。
//!
//! 内核 vs 标量参考的整批对拍走这里——[`max_rel_err`] 与 `cutile::reference` 同一口径，只有
//! 一个全局分母，答的是「整批数合在一起差多少」，不是逐元素各算各的；逐元素与方向判据在
//! [`crate::compare`]，两者不可混用。
//!
//! 对拍失败时不要停在布尔值上：[`rel_err`] 回一份读数（误差、分母、最差元素），
//! [`assert_within_rel`] 直接把这三样写进 panic 消息，门禁用例应当用它。

use crate::compare::close_scalar;

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

#[cfg(test)]
#[path = "../tests/rel_tests.rs"]
mod rel_tests;
