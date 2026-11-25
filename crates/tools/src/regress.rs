//! 回退判定：把"实测值 vs 基线"变成一条可执行的结论。
//!
//! **拥有**：`Case`、`Direction`、`Baseline`、`Judgement`。**不拥有**：测量本身（谁跑、
//! 跑多久、取中位数由 `llm-benchmarks` 的 bench target 与 CI 的 bench job 决定）、性能目标的
//! 数值（写在 `Case` 里）。
//!
//! 为什么把它做成类型而不是脚本里的几个 `if`：性能判据有三个容易搞错的地方——**方向**
//! （带宽越高越好，延迟越低越好）、**相对预算**（不同机器的绝对值不可比）、**区分"回退"
//! 与"失败"**（回退要拦，改进不该报警）。把这三件事写进类型，CI 就不可能各写一套。
//!
//! 与 [`crate::Gate`] 同属"判定"这一族：`Gate` 判"代码指标有没有变差"，本模块判"实测性能
//! 有没有变差"；两者都只产出数据，失败由调用方决定如何报告。
//!
//! # 例子
//!
//! ```
//! use llm_tools::regress::{Baseline, Case, Direction, Judgement};
//!
//! let case = Case::new("gemv_i4/cpu", 10.0, 5.0, Direction::Higher).expect("场景");
//! let baseline = Baseline::new(vec![case]).expect("基线");
//! assert!(baseline.judge("gemv_i4/cpu", 10.2).expect("判定").is_pass());
//! assert!(matches!(
//!     baseline.judge("gemv_i4/cpu", 9.0).expect("判定"),
//!     Judgement::Regressed { .. }
//! ));
//! ```

use diagnostic::{Code, Error, Result};

/// 配置错误：上下文统一带 `tools::regress::` 前缀。
fn invalid(ctx: &str, message: impl Into<String>) -> Error {
    crate::error::invalid("regress", ctx, message)
}

/// 指标方向：越大越好，还是越小越好。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// 越大越好（吞吐、带宽、达成率）。
    Higher,
    /// 越小越好（延迟、指令数、显存峰值）。
    Lower,
}

impl Direction {
    /// 稳定的短名，写进输出。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Higher => "higher-is-better",
            Self::Lower => "lower-is-better",
        }
    }
}

/// 一次测量：点估计与其置信区间。
///
/// 区间来自 criterion mean 的 `confidence_interval`，本模块不做第二套 bootstrap 重采样
/// （重采样区间归 [`crate::card::Tally`]）。三点必须满足 `lower ≤ point ≤ upper` 且均为有限数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    /// 点估计（mean.point_estimate）。
    point: f64,
    /// 置信区间下界。
    lower: f64,
    /// 置信区间上界。
    upper: f64,
}

impl Estimate {
    /// 单点测量：区间退化为零宽 `[x, x]`；用于旧单点入口与尚无区间的采集路径。
    pub fn point(value: f64) -> Self {
        Self {
            point: value,
            lower: value,
            upper: value,
        }
    }

    /// 带区间的测量；非有限数或区间逆序返回 `invalid`（采集坏了，不是没测到）。
    pub fn interval(lower: f64, point: f64, upper: f64) -> Result<Self> {
        let finite = [lower, point, upper].iter().all(|v| v.is_finite());
        if !finite || lower > point || point > upper {
            return Err(invalid(
                "Estimate::interval",
                format!("区间必须有限且有序：lower {lower}, point {point}, upper {upper}"),
            ));
        }
        Ok(Self {
            point,
            lower,
            upper,
        })
    }

    /// 点估计。
    pub fn point_estimate(&self) -> f64 {
        self.point
    }

    /// 区间下界。
    pub fn lower(&self) -> f64 {
        self.lower
    }

    /// 区间上界。
    pub fn upper(&self) -> f64 {
        self.upper
    }
}

/// 未判定理由：区间跨着预算线，样本还不够把「过」与「不过」分开。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenReason {
    /// 区间与预算线相交，方向上不能整段归内侧或外侧。
    Straddle,
}

impl OpenReason {
    /// 稳定的短名，写进归档。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Straddle => "straddle",
        }
    }
}

/// 一个基准场景：名字、基线值、允许的相对偏差、不可察觉带与方向。
#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    name: String,
    baseline: f64,
    budget_percent: f64,
    epsilon_percent: f64,
    direction: Direction,
}

impl Case {
    /// `baseline` 必须为正的有限数，`budget_percent` 落在 `0..=100`。
    pub fn new(
        name: impl Into<String>,
        baseline: f64,
        budget_percent: f64,
        direction: Direction,
    ) -> Result<Self> {
        Self::with_epsilon(name, baseline, budget_percent, 0.0, direction)
    }

    /// 同 [`new`](Self::new)，另设不可察觉带 `epsilon_percent`：相对基线在该带内的漂移记持平，
    /// 不报改进。`epsilon_percent` 落在 `0..=100`。
    pub fn with_epsilon(
        name: impl Into<String>,
        baseline: f64,
        budget_percent: f64,
        epsilon_percent: f64,
        direction: Direction,
    ) -> Result<Self> {
        if !baseline.is_finite() || baseline <= 0.0 {
            return Err(invalid(
                "Case::new",
                format!("基线必须是正的有限数，得到 {baseline}"),
            ));
        }
        if !(0.0..=100.0).contains(&budget_percent) {
            return Err(invalid(
                "new",
                format!("预算必须落在 0..=100，得到 {budget_percent}"),
            ));
        }
        if !(0.0..=100.0).contains(&epsilon_percent) {
            return Err(invalid(
                "with_epsilon",
                format!("不可察觉带必须落在 0..=100，得到 {epsilon_percent}"),
            ));
        }
        Ok(Self {
            name: name.into(),
            baseline,
            budget_percent,
            epsilon_percent,
            direction,
        })
    }

    /// 场景名（与 `doctor` / `bench` 输出的场景名一致）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 基线值。
    pub fn baseline(&self) -> f64 {
        self.baseline
    }

    /// 允许的相对偏差（百分比）。
    pub fn budget_percent(&self) -> f64 {
        self.budget_percent
    }

    /// 指标方向。
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// 判定阈值：比基线放宽 `budget_percent` 后的那条线。
    pub fn limit(&self) -> f64 {
        let factor = self.budget_percent / 100.0;
        match self.direction {
            Direction::Higher => self.baseline * (1.0 - factor),
            Direction::Lower => self.baseline * (1.0 + factor),
        }
    }

    /// 改进门槛：整段优于这条线才算稳健改进；落在不可察觉带内记持平。
    fn improved_line(&self) -> f64 {
        let eps = self.epsilon_percent / 100.0;
        match self.direction {
            Direction::Higher => self.baseline * (1.0 + eps),
            Direction::Lower => self.baseline * (1.0 - eps),
        }
    }

    /// 效应规模：相对基线的变化比例，归一为「改进为正」；回退为负。
    fn effect(&self, point: f64) -> f64 {
        match self.direction {
            Direction::Higher => (point - self.baseline) / self.baseline,
            Direction::Lower => (self.baseline - point) / self.baseline,
        }
    }

    /// 判定一次单点实测（区间退化为零宽，不会跨门槛）。
    pub fn judge(&self, observed: f64) -> Judgement {
        self.judge_interval(Estimate::point(observed))
    }

    /// 判定一次带置信区间的实测：整段内侧 Pass、整段外侧 Regressed、跨门槛 Open。
    pub fn judge_interval(&self, estimate: Estimate) -> Judgement {
        if !estimate.point_estimate().is_finite() {
            return self.regressed(estimate.point_estimate());
        }
        match self.direction {
            Direction::Higher => self.judge_higher(estimate),
            Direction::Lower => self.judge_lower(estimate),
        }
    }

    /// 越大越好：预算线在下侧，改进门槛在上侧。
    fn judge_higher(&self, est: Estimate) -> Judgement {
        if est.upper < self.limit() {
            return self.regressed(est.point_estimate());
        }
        if est.lower > self.improved_line() {
            return self.improved(est.point_estimate());
        }
        if est.lower >= self.limit() {
            return Judgement::Pass;
        }
        self.straddle(est)
    }

    /// 越小越好：预算线在上侧，改进门槛在下侧。
    fn judge_lower(&self, est: Estimate) -> Judgement {
        if est.lower > self.limit() {
            return self.regressed(est.point_estimate());
        }
        if est.upper < self.improved_line() {
            return self.improved(est.point_estimate());
        }
        if est.upper <= self.limit() {
            return Judgement::Pass;
        }
        self.straddle(est)
    }

    /// 回退结论：携带实测、基线与效应规模。
    fn regressed(&self, observed: f64) -> Judgement {
        Judgement::Regressed {
            observed,
            baseline: self.baseline,
            effect: self.effect(observed),
        }
    }

    /// 改进结论：携带实测、基线与效应规模。
    fn improved(&self, observed: f64) -> Judgement {
        Judgement::Improved {
            observed,
            baseline: self.baseline,
            effect: self.effect(observed),
        }
    }

    /// 跨门槛结论：携带区间两端、基线与理由。
    fn straddle(&self, est: Estimate) -> Judgement {
        Judgement::Open {
            lower: est.lower,
            upper: est.upper,
            baseline: self.baseline,
            reason: OpenReason::Straddle,
        }
    }
}

/// 判定结论：通过、回退、改进，或跨门槛未判定。
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub enum Judgement {
    /// 在预算内（含与基线持平）。
    Pass,
    /// 超出预算：CI 必须失败。
    Regressed {
        /// 实测点估计。
        observed: f64,
        /// 基线值。
        baseline: f64,
        /// 效应规模（改进为正，回退为负，相对基线）。
        effect: f64,
    },
    /// 整段优于基线：不报警，但在报告里单独列出。
    Improved {
        /// 实测点估计。
        observed: f64,
        /// 基线值。
        baseline: f64,
        /// 效应规模（改进为正）。
        effect: f64,
    },
    /// 区间跨着预算线：样本还不够把「过」与「不过」分开，不等于失败。
    Open {
        /// 区间下界。
        lower: f64,
        /// 区间上界。
        upper: f64,
        /// 基线值。
        baseline: f64,
        /// 未判定理由。
        reason: OpenReason,
    },
}

impl Judgement {
    /// 是否回退（唯一的失败态；未判定与改进都不是失败）。
    pub fn is_regressed(&self) -> bool {
        matches!(self, Self::Regressed { .. })
    }

    /// 是否未判定（跨门槛，需补样本或在报告里标 Open）。
    pub fn is_open(&self) -> bool {
        matches!(self, Self::Open { .. })
    }

    /// 是否不算失败：回退才失败，通过 / 改进 / 未判定均不拦 CI。
    pub fn is_pass(&self) -> bool {
        !self.is_regressed()
    }

    /// 效应规模；持平与未判定没有可报的效应，返回 `None`。
    pub fn effect(&self) -> Option<f64> {
        match self {
            Self::Regressed { effect, .. } | Self::Improved { effect, .. } => Some(*effect),
            Self::Pass | Self::Open { .. } => None,
        }
    }

    /// 稳定的短名。
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Regressed { .. } => "regressed",
            Self::Improved { .. } => "improved",
            Self::Open { .. } => "open",
        }
    }
}

/// 一组场景：名字唯一，按名字查。
#[derive(Debug, Clone, PartialEq)]
pub struct Baseline {
    cases: Vec<Case>,
}

impl Baseline {
    /// 校验并构造：至少一个场景，名字不得重复。
    pub fn new(cases: Vec<Case>) -> Result<Self> {
        if cases.is_empty() {
            return Err(invalid("Baseline::new", "基线不能为空"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for case in &cases {
            if !seen.insert(case.name()) {
                return Err(invalid(
                    "Baseline::new",
                    format!("场景 `{}` 重复", case.name()),
                ));
            }
        }
        Ok(Self { cases })
    }

    /// 全部场景。
    pub fn cases(&self) -> &[Case] {
        &self.cases
    }

    /// 按名字查场景。
    pub fn get(&self, name: &str) -> Option<&Case> {
        self.cases.iter().find(|case| case.name() == name)
    }

    /// 判定某个场景的实测值；场景不存在是配置错误，不是性能回退。
    pub fn judge(&self, name: &str, observed: f64) -> Result<Judgement> {
        let case = self.get(name).ok_or_else(|| {
            Error::new(
                Code::Missing,
                format!(
                    "场景 `{name}` 不在基线里（已登记：{}）",
                    self.cases
                        .iter()
                        .map(Case::name)
                        .collect::<Vec<_>>()
                        .join("、")
                ),
            )
            .with("tools::regress::Baseline::judge")
        })?;
        Ok(case.judge(observed))
    }
}
