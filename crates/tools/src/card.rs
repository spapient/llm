//! 实测卡：把 `docs/lld.md` §10.10 的协议卡落成一次可复核的三态结论（PASS / FAIL / OPEN）。
//!
//! 拥有 `Protocol`（判哪个量、怎么聚、门槛、样本下限、主机闭集）、`Statistic`（均值 / 变异
//! 系数）、`Tally`（样本摘要 + 统计量的重采样区间）、`Shortfall`（为什么 OPEN）、`Standing`
//! （三态）、`Card`（协议 + 装置身份 + 摘要 + 结论）。不拥有样本本身、时钟、温度轨迹。
//!
//! 与 [`crate::regress::Judgement`] 分工：`Judgement` 判「相对基线回退没有」；本模块判「这份
//! 协议卡这一格成立了没有」，门槛是绝对的。`lld` 规则 0.3：OPEN ≠ 降 Must——「没测出来」既
//! 不许写 PASS，也不许写 FAIL。

use diagnostic::{Code, Error, Result};

use crate::bench::Harness;
use crate::escape::quote;
pub use crate::regress::Direction;
use crate::stats;

/// 协议卡默认样本下限（`lld` §10.10 的预注册窗口，几十个起步）。
const LEAST: usize = 30;
/// 协议卡置信水平（`lld` 一律 95% CI，不开放配置）。
const CONFIDENCE: f64 = 0.95;

/// 配置错误：上下文统一带 `tools::card::` 前缀。
fn invalid(context: &str, message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(format!("tools::card::{context}"))
}

/// 卡片上那个数怎么从样本聚出来。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Statistic {
    /// 均值：延迟、吞吐、达成率这类「一个窗口一个读数」的量。
    Mean,
    /// 变异系数 `σ/μ`（`lld` E10）：均值 0 时无定义。
    Cv,
}

impl Statistic {
    /// 稳定的短名，写进归档。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Cv => "cv",
        }
    }

    /// 这一列上的统计量；无定义时 `None`（由调用方记 OPEN，而不是填假数）。
    fn value(self, samples: &[f64]) -> Option<f64> {
        match self {
            Self::Mean => (!samples.is_empty()).then(|| stats::mean(samples)),
            Self::Cv => {
                let center = stats::mean(samples);
                (samples.len() >= 2 && center != 0.0).then(|| stats::sd(samples) / center)
            }
        }
    }
}

/// 一份协议卡：判哪个量、怎么聚、往哪个方向过门、要几个样本、认哪些主机。门槛是绝对的。
#[derive(Debug, Clone)]
pub struct Protocol {
    name: &'static str,
    metric: &'static str,
    statistic: Statistic,
    direction: Direction,
    bound: f64,
    least: usize,
    hosts: &'static [&'static str],
}

impl Protocol {
    /// 按「名字 / 被测量 / 聚合方式 / 方向 / 门槛」构造；样本下限取默认，主机闭集为空。
    ///
    /// # Errors
    ///
    /// 名字或量名为空、门槛非有限数时返回 `invalid`。
    pub fn of(
        name: &'static str,
        metric: &'static str,
        statistic: Statistic,
        direction: Direction,
        bound: f64,
    ) -> Result<Self> {
        if name.is_empty() || metric.is_empty() {
            return Err(invalid("of", "协议名与被测量名不能为空"));
        }
        if !bound.is_finite() {
            return Err(invalid("of", format!("门槛必须是有限数，得到 {bound}")));
        }
        Ok(Self {
            name,
            metric,
            statistic,
            direction,
            bound,
            least: LEAST,
            hosts: &[],
        })
    }

    /// 改样本下限：少于这个数就记 OPEN。
    ///
    /// # Errors
    ///
    /// 小于 2 时返回 `invalid`：一个样本既算不出标准差，也算不出区间。
    pub fn with_least(self, least: usize) -> Result<Self> {
        if least < 2 {
            return Err(invalid("least", format!("样本下限至少 2，得到 {least}")));
        }
        Ok(Self { least, ..self })
    }

    /// 改主机闭集：非空时闭集外的主机一律记 OPEN（`lld` §10.2「闭集外 SKU → OPEN」）。
    pub fn with_hosts(self, hosts: &'static [&'static str]) -> Self {
        Self { hosts, ..self }
    }

    /// 协议名（`lld` 的维度编号，如 `E10`）。
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// 这台装置在不在闭集里；闭集为空时恒真。
    fn admits(&self, host: &str) -> bool {
        self.hosts.is_empty() || self.hosts.contains(&host)
    }
}

/// 这一格为什么是 OPEN：四种「不成立」措辞分开，避免把「样本不够」读成「没达标」。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Shortfall {
    /// 装置不在闭集里：换一台机器，不是重跑。
    Host,
    /// 样本数不到下限：补窗口。
    Samples,
    /// 统计量在这列上无定义（变异系数的均值为 0）。
    Degenerate,
    /// 区间跨着门槛：样本还不够把「过」与「不过」分开。
    Straddle,
}

impl Shortfall {
    /// 稳定的短名，写进归档。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Samples => "samples",
            Self::Degenerate => "degenerate",
            Self::Straddle => "straddle",
        }
    }
}

/// 一张卡的三态结论。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Standing {
    /// 区间整段落在门槛内侧。
    Pass,
    /// 区间整段落在门槛外侧。
    Fail,
    /// 判不了，理由在 [`Shortfall`]；**不等于**降 Must，也不等于失败。
    Open(Shortfall),
}

impl Standing {
    /// 稳定的短名，写进归档。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Open(_) => "open",
        }
    }

    /// 未判定时的理由。
    pub fn shortfall(self) -> Option<Shortfall> {
        match self {
            Self::Open(shortfall) => Some(shortfall),
            Self::Pass | Self::Fail => None,
        }
    }
}

/// 一列样本的摘要：计数、离散，以及**统计量自己**的重采样区间（与统计量同坐标，不是样本分位）。
#[derive(Debug, Clone, Copy)]
pub struct Tally {
    count: usize,
    mean: f64,
    sd: f64,
    cv: Option<f64>,
    p50: f64,
    p95: f64,
    p99: f64,
    lower: f64,
    upper: f64,
}

impl Tally {
    /// 摘要一列样本；统计量无定义（空白列、变异系数的均值为 0）时返回 `Ok(None)`。
    ///
    /// # Errors
    ///
    /// 样本含非有限数时返回 `invalid`：那是采集坏了，静默跳过会把时间序列断裂藏进一张正常的卡。
    pub fn of(samples: &[f64], statistic: Statistic) -> Result<Option<Self>> {
        finite(samples)?;
        let Some(draws) = stats::interval(samples, |drawn| statistic.value(drawn)) else {
            return Ok(None);
        };
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let tail = (1.0 - CONFIDENCE) / 2.0;
        let center = stats::mean(samples);
        let sd = stats::sd(samples);
        Ok(Some(Self {
            count: samples.len(),
            mean: center,
            sd,
            cv: (samples.len() >= 2 && center != 0.0).then(|| sd / center),
            p50: stats::quantile(&sorted, 0.50),
            p95: stats::quantile(&sorted, 0.95),
            p99: stats::quantile(&sorted, 0.99),
            lower: stats::quantile(&draws, tail),
            upper: stats::quantile(&draws, 1.0 - tail),
        }))
    }

    /// 样本数。
    pub fn count(&self) -> usize {
        self.count
    }

    /// 样本标准差（`n = 1` 时为 0）。
    pub fn sd(&self) -> f64 {
        self.sd
    }

    /// 统计量区间的下界。
    pub fn lower(&self) -> f64 {
        self.lower
    }

    /// 统计量区间的上界。
    pub fn upper(&self) -> f64 {
        self.upper
    }
}

/// 一张实测卡：协议 + 装置身份 + 摘要 + 三态结论。摘要可缺（统计量无定义），结论不会因此变 PASS。
#[derive(Debug, Clone)]
pub struct Card {
    protocol: Protocol,
    harness: Harness,
    tally: Option<Tally>,
    standing: Standing,
}

impl Card {
    /// 判一张卡：先聚摘要，再按门槛落成三态。判定次序：装置闭集 → 样本量 → 统计量定义 → 区间与门槛。
    ///
    /// # Errors
    ///
    /// 样本含非有限数时返回 `invalid`（见 [`Tally::of`]）。
    pub fn judge(protocol: Protocol, harness: Harness, samples: &[f64]) -> Result<Self> {
        let tally = Tally::of(samples, protocol.statistic)?;
        let standing = standing_of(&protocol, &harness, samples.len(), tally.as_ref());
        Ok(Self {
            protocol,
            harness,
            tally,
            standing,
        })
    }

    /// 协议。
    pub fn protocol(&self) -> &Protocol {
        &self.protocol
    }

    /// 样本摘要；统计量无定义时为 `None`。
    pub fn tally(&self) -> Option<&Tally> {
        self.tally.as_ref()
    }

    /// 三态结论。
    pub fn standing(&self) -> Standing {
        self.standing
    }

    /// 归档 JSON：键序固定，缺测写 `null`，结论旁带理由（`shortfall`）。
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");
        string_field(&mut out, "protocol", self.protocol.name);
        string_field(&mut out, "metric", self.protocol.metric);
        string_field(&mut out, "statistic", self.protocol.statistic.as_str());
        string_field(&mut out, "direction", self.protocol.direction.as_str());
        number_field(&mut out, "bound", self.protocol.bound);
        number_field(&mut out, "least", self.protocol.least);
        number_field(&mut out, "confidence", CONFIDENCE);
        harness_fields(&mut out, &self.harness);
        tally_field(&mut out, self.tally.as_ref());
        standing_fields(&mut out, self.standing);
        out.push_str("}\n");
        out
    }
}

use std::fmt::Write as _;

/// 顶层 `"键": "值",`。
fn string_field(out: &mut String, key: &str, value: &str) {
    let _ = write!(out, "  \"{key}\": ");
    quote(out, value);
    out.push_str(",\n");
}

/// 顶层 `"键": <数>,`。
fn number_field(out: &mut String, key: &str, value: impl std::fmt::Display) {
    let _ = write!(out, "  \"{key}\": {value},\n");
}

/// 装置段：`lld` §10.2 要求「同装置对照」，所以身份与数一起归档。
fn harness_fields(out: &mut String, harness: &Harness) {
    out.push_str("  \"harness\": {\n");
    for (position, (key, value)) in [
        ("host", harness.host()),
        ("cpu", harness.cpu()),
        ("os", harness.os()),
        ("arch", harness.arch()),
        ("toolchain", harness.toolchain()),
        ("driver", harness.driver()),
    ]
    .into_iter()
    .enumerate()
    {
        if position > 0 {
            out.push_str(",\n");
        }
        out.push_str("    ");
        quote(out, key);
        out.push_str(": ");
        quote(out, value);
    }
    out.push_str("\n  },\n");
}

/// 摘要段：统计量无定义时整段写 `null`，不拿 0 冒充。
fn tally_field(out: &mut String, tally: Option<&Tally>) {
    out.push_str("  \"tally\": ");
    let Some(tally) = tally else {
        out.push_str("null,\n");
        return;
    };
    out.push_str("{\n    \"count\": ");
    let _ = write!(out, "{},\n", tally.count);
    real(out, "mean", tally.mean);
    real(out, "sd", tally.sd);
    match tally.cv {
        None => out.push_str(",\n    \"cv\": null"),
        Some(cv) => real(out, "cv", cv),
    }
    real(out, "p50", tally.p50);
    real(out, "p95", tally.p95);
    real(out, "p99", tally.p99);
    real(out, "lower", tally.lower);
    real(out, "upper", tally.upper);
    out.push_str("\n  },\n");
}

/// 一个实数域：前置逗号与换行，键名与值都按 JSON 写。
fn real(out: &mut String, key: &str, value: f64) {
    let _ = write!(out, ",\n    \"{key}\": {value}");
}

/// 结论段：三态名 + 理由；PASS / FAIL 的理由写 `null`。
fn standing_fields(out: &mut String, standing: Standing) {
    out.push_str("  \"standing\": ");
    quote(out, standing.as_str());
    out.push_str(",\n  \"shortfall\": ");
    match standing.shortfall() {
        None => out.push_str("null"),
        Some(shortfall) => quote(out, shortfall.as_str()),
    }
    out.push('\n');
}

/// 三态判定的全部规则，集中在这一处。
fn standing_of(
    protocol: &Protocol,
    harness: &Harness,
    count: usize,
    tally: Option<&Tally>,
) -> Standing {
    if !protocol.admits(harness.host()) {
        return Standing::Open(Shortfall::Host);
    }
    if count < protocol.least {
        return Standing::Open(Shortfall::Samples);
    }
    let Some(tally) = tally else {
        return Standing::Open(Shortfall::Degenerate);
    };
    against(protocol.direction, tally, protocol.bound)
}

/// 区间与门槛的关系：整段在内侧 PASS、整段在外侧 FAIL、跨着门槛只能 OPEN。
fn against(direction: Direction, tally: &Tally, bound: f64) -> Standing {
    let (inner, outer) = match direction {
        Direction::Higher => (tally.lower >= bound, tally.upper < bound),
        Direction::Lower => (tally.upper < bound, tally.lower >= bound),
    };
    if inner {
        Standing::Pass
    } else if outer {
        Standing::Fail
    } else {
        Standing::Open(Shortfall::Straddle)
    }
}

/// 样本必须是有限数：混进 `NaN` 说明采集坏了，不是「这一格没测出来」。
fn finite(samples: &[f64]) -> Result<()> {
    match samples.iter().find(|value| !value.is_finite()) {
        None => Ok(()),
        Some(bad) => Err(invalid(
            "of",
            format!("样本必须是有限数，得到 {bad}（采集坏了，不是没测到）"),
        )),
    }
}
