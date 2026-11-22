//! 整批相对误差口径：一个全局分母下的最大误差、失败读数与阈值投影。

use crate::compare::close_scalar;

/// 误差最大的那个元素：`idx` 是它在切片里的下标，`got` / `want` 是两边的原值。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Worst {
    /// 元素在切片里的下标。
    pub idx: usize,
    /// 被测侧的值。
    pub got: f32,
    /// 参考侧的值。
    pub want: f32,
}

/// 一次整批对拍的读数：[`RelErr::ratio`] 是判据本体，其余字段只为失败时定位。
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

/// 全局相对误差是否在阈值内：`max_rel_err(got, want) <= tolerance`。
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
