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

    /// 折算到「越大越好」的可比坐标：`Lower` 取相反数（取负对浮点是精确的）。
    fn polar(self, value: f64) -> f64 {
        match self {
            Self::Higher => value,
            Self::Lower => -value,
        }
    }
}

/// 一个基准场景：名字、基线值、允许的相对偏差与方向。
#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    name: String,
    baseline: f64,
    budget_percent: f64,
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
        if !baseline.is_finite() || baseline <= 0.0 {
            return Err(invalid(
                "Case::new",
                format!("基线必须是正的有限数，得到 {baseline}"),
            ));
        }
        if !(0.0..=100.0).contains(&budget_percent) {
            return Err(invalid(
                "Case::new",
                format!("预算必须落在 0..=100，得到 {budget_percent}"),
            ));
        }
        Ok(Self {
            name: name.into(),
            baseline,
            budget_percent,
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

    /// 判定一次实测。
    pub fn judge(&self, observed: f64) -> Judgement {
        if !observed.is_finite() {
            return self.regressed(observed);
        }
        let seen = self.direction.polar(observed);
        if seen > self.direction.polar(self.baseline) {
            return Judgement::Improved {
                observed,
                baseline: self.baseline,
            };
        }
        if seen >= self.direction.polar(self.limit()) {
            Judgement::Pass
        } else {
            self.regressed(observed)
        }
    }

    /// 回退结论：携带实测与基线。
    fn regressed(&self, observed: f64) -> Judgement {
        Judgement::Regressed {
            observed,
            baseline: self.baseline,
        }
    }
}

/// 判定结论：通过、回退、改进。
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub enum Judgement {
    /// 在预算内（含与基线持平）。
    Pass,
    /// 超出预算：CI 必须失败。
    Regressed {
        /// 实测值。
        observed: f64,
        /// 基线值。
        baseline: f64,
    },
    /// 好于基线：不报警，但在报告里单独列出。
    Improved {
        /// 实测值。
        observed: f64,
        /// 基线值。
        baseline: f64,
    },
}

impl Judgement {
    /// 是否算通过（回退是唯一的失败态）。
    pub fn is_pass(&self) -> bool {
        !matches!(self, Self::Regressed { .. })
    }

    /// 稳定的短名。
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Regressed { .. } => "regressed",
            Self::Improved { .. } => "improved",
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
