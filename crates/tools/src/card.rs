//! 实测卡：把 `docs/lld.md` §10.10 的协议卡做成一次可复核的三态结论。
//!
//! **拥有**：`Protocol`（判哪个量、怎么聚、门槛在哪、要几个样本、认哪些主机）、`Statistic`
//! （聚合方式：均值 / 变异系数）、`Tally`（样本摘要与统计量的重采样区间）、`Shortfall`
//! （这一格为什么是 OPEN）、`Standing`（PASS / FAIL / OPEN 三态）、`Card`（协议 + 装置身份 +
//! 摘要 + 结论）。**不拥有**：样本本身（预热、稳态窗口与时长由 `llm-benchmarks` 与 CI 决定）、
//! 时钟来源、温度 / 频率 / 功耗轨迹（E10 卡要求分账，那是采集侧的事）。
//!
//! 三态不是三档颜色，而是三种**不同的知识状态**：样本够、区间整段落在门槛内侧是 PASS；
//! 区间整段落在门槛外侧是 FAIL；区间跨着门槛、样本不够、统计量在数学上无定义、或者主机不在
//! 协议认的闭集里，都只能记 OPEN。`lld` 的规则 0.3 写死了这条：**OPEN ≠ 降 Must**，所以
//! 「没测出来」既不许写成 PASS，也不许写成 FAIL——两种写法都会让下一轮拿不到该补的那一格。
//!
//! 与 [`crate::regress::Judgement`] 分工：`Judgement` 判「与基线相比回退了没有」，基线是
//! 相对的；本模块判「这份协议卡这一格成立了没有」，门槛是绝对的。三态名刻意用 `Standing`
//! 而不是 `Verdict`——`cutile::bench::Verdict` 已经在用那个名字判「这次测量能不能与档案峰值
//! 比较」。
//!
//! # 例子
//!
//! ```
//! use llm_tools::bench::Harness;
//! use llm_tools::card::{Card, Direction, Protocol, Standing, Statistic};
//!
//! let protocol = Protocol::of("E10", "decode_step_ns", Statistic::Cv, Direction::Lower, 0.05)
//!     .expect("协议")
//!     .with_least(8)
//!     .expect("样本下限");
//! let harness = Harness::new("macos-aarch64", "cpu", "os", "aarch64", "rustc", "none");
//!
//! // 稳的一列：均值 10、标准差 0，变异系数 0，区间上界必然 < 5%。
//! let steady = Card::judge(protocol.clone(), harness.clone(), &[10.0; 8]).expect("判");
//! assert_eq!(steady.standing(), Standing::Pass);
//!
//! // 抖的一列：同一协议下区间整段越过门槛，只能记 FAIL。
//! let noisy = Card::judge(protocol, harness, &[1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0])
//!     .expect("判");
//! assert_eq!(noisy.standing(), Standing::Fail);
//! ```

use diagnostic::{Code, Error, Result};

use crate::bench::Harness;
use crate::escape::quote;
pub use crate::regress::Direction;
use crate::stats;

/// 协议卡默认的样本下限：`lld` §10.10 各卡要求的「预注册窗口」在机器上通常有几十个以上，
/// 少于这个数谈不上区间，直接 OPEN。
const LEAST: usize = 30;

/// 协议卡默认的置信水平（`lld` 的 95% CI 口径）。
const CONFIDENCE: f64 = 0.95;

/// 配置错误：上下文统一带 `tools::card::` 前缀。
fn invalid(context: &str, message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(format!("tools::card::{context}"))
}

/// 卡片上那个数是怎么从样本聚出来的。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Statistic {
    /// 均值：延迟、吞吐、达成率这类「一个窗口一个读数」的量。
    Mean,
    /// 变异系数 `σ/μ`：计算可预测性（`lld` E10）用的就是它，均值 0 时无定义。
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

    /// 这一列样本上的统计量；无定义时 `None`（由调用方记 OPEN，而不是填一个假数）。
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

/// 一份协议卡：判哪个量、怎么聚、往哪个方向过门、要几个样本、认哪些主机。
///
/// 门槛是**绝对**的（`lld` 的 Must 就是绝对值），方向决定区间怎样算「整段在内侧」：
/// 「越大越好」要求区间下界 ≥ 门槛（`lld` E6 的「只有下界 ≥0.9 才 PASS」），「越小越好」
/// 要求区间上界 < 门槛（E10 的「只有 CV 的 95% CI 上界 <5% 才 PASS」）。
#[derive(Debug, Clone)]
pub struct Protocol {
    name: &'static str,
    metric: &'static str,
    statistic: Statistic,
    direction: Direction,
    bound: f64,
    least: usize,
    confidence: f64,
    hosts: &'static [&'static str],
}

impl Protocol {
    /// 按「名字 / 被测量 / 聚合方式 / 方向 / 门槛」构造，样本下限与置信水平取默认值，
    /// 主机闭集为空（任何装置都认）。
    ///
    /// # Errors
    ///
    /// 名字或量名为空、门槛非有限数时返回 `invalid`；门槛与方向是否自相矛盾**不**在这里判
    /// ——「越低越好且门槛为负」在某些量上是有意义的，判据由协议作者负责。
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
            confidence: CONFIDENCE,
            hosts: &[],
        })
    }

    /// 改样本下限：少于这个数就记 OPEN，不判 PASS 也不判 FAIL。
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

    /// 改置信水平（`lld` 一律 95%）。
    ///
    /// # Errors
    ///
    /// 不在 `(0, 1)` 开区间内时返回 `invalid`。
    pub fn with_confidence(self, confidence: f64) -> Result<Self> {
        if !(confidence > 0.0 && confidence < 1.0) {
            return Err(invalid(
                "confidence",
                format!("置信水平必须在 (0,1) 内，得到 {confidence}"),
            ));
        }
        Ok(Self { confidence, ..self })
    }

    /// 改主机闭集：非空时，闭集外的主机一律记 OPEN（`lld` §10.2「闭集外 SKU → OPEN」）。
    pub fn with_hosts(self, hosts: &'static [&'static str]) -> Self {
        Self { hosts, ..self }
    }

    /// 协议名（`lld` 的维度编号，如 `E10`）。
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// 被测量的名字（与 [`crate::bench::BenchRecord`] 的 `name` 同口径）。
    pub fn metric(&self) -> &'static str {
        self.metric
    }

    /// 聚合方式。
    pub fn statistic(&self) -> Statistic {
        self.statistic
    }

    /// 门槛方向。
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// 门槛。
    pub fn bound(&self) -> f64 {
        self.bound
    }

    /// 样本下限。
    pub fn least(&self) -> usize {
        self.least
    }

    /// 置信水平。
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// 主机闭集；空表示不设闭集。
    pub fn hosts(&self) -> &'static [&'static str] {
        self.hosts
    }

    /// 这台装置在不在协议认的闭集里；闭集为空时恒真。
    fn admits(&self, host: &str) -> bool {
        self.hosts.is_empty() || self.hosts.contains(&host)
    }
}

/// 这一格为什么是 OPEN：四种「不成立」，措辞分开，避免读者把「样本不够」读成「没达标」。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Shortfall {
    /// 装置不在协议认的闭集里：换一台机器再测，不是重跑。
    Host,
    /// 样本数不到协议下限：补窗口。
    Samples,
    /// 统计量在这列样本上无定义（变异系数的均值为 0）。
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

/// 一张卡的三态结论：通过 / 未通过 / 未判定。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Standing {
    /// 区间整段落在门槛内侧。
    Pass,
    /// 区间整段落在门槛外侧：这一格确实没达标。
    Fail,
    /// 判不了，理由在 [`Shortfall`] 里；**不等于**降 Must，也不等于失败。
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

    /// 是否通过。
    pub fn is_pass(self) -> bool {
        matches!(self, Self::Pass)
    }

    /// 是否未判定（含理由）。
    pub fn shortfall(self) -> Option<Shortfall> {
        match self {
            Self::Open(shortfall) => Some(shortfall),
            Self::Pass | Self::Fail => None,
        }
    }
}

/// 一列样本的摘要：计数、中心与离散、三个分位，以及**统计量自己**的重采样区间。
///
/// 区间是对统计量（均值或变异系数）重采样得到的，不是对样本本身取分位——卡片要判的是
/// 「这个统计量过不过门」，所以区间必须与统计量同坐标。
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
    /// 样本里混进非有限数时返回 `invalid`：那是采集坏了，不是「这一格没测出来」，静默跳过会
    /// 把时间序列的断裂藏进一张看起来正常的卡里。
    pub fn of(samples: &[f64], statistic: Statistic, confidence: f64) -> Result<Option<Self>> {
        finite(samples)?;
        let Some(draws) = stats::interval(samples, |drawn| statistic.value(drawn)) else {
            return Ok(None);
        };
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let tail = (1.0 - confidence) / 2.0;
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

    /// 均值。
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// 样本标准差（`n = 1` 时为 0）。
    pub fn sd(&self) -> f64 {
        self.sd
    }

    /// 变异系数 `σ/μ`；均值为 0 或样本不足 2 个时 `None`。
    pub fn cv(&self) -> Option<f64> {
        self.cv
    }

    /// 样本中位数。
    pub fn p50(&self) -> f64 {
        self.p50
    }

    /// 样本 95 分位。
    pub fn p95(&self) -> f64 {
        self.p95
    }

    /// 样本 99 分位。
    pub fn p99(&self) -> f64 {
        self.p99
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

/// 一张实测卡：协议 + 装置身份 + 摘要 + 三态结论。
///
/// 摘要可以缺（统计量无定义时），结论不会因此变成 PASS；装置身份一定在，因为「同装置对照」
/// 是 `lld` §10.2 的前提，缺了身份的实测数不可比。
#[derive(Debug, Clone)]
pub struct Card {
    protocol: Protocol,
    harness: Harness,
    tally: Option<Tally>,
    standing: Standing,
}

impl Card {
    /// 判一张卡：先把样本聚成摘要，再按协议门槛落成三态。
    ///
    /// 判定次序是有意的：装置闭集 → 样本量 → 统计量定义 → 区间与门槛的关系。闭集排在最前，
    /// 因为一台不在闭集里的机器上跑出来的数，样本再多也**不是**这份协议的证据。
    ///
    /// # Errors
    ///
    /// 样本含非有限数时返回 `invalid`（见 [`Tally::of`]）。
    pub fn judge(protocol: Protocol, harness: Harness, samples: &[f64]) -> Result<Self> {
        let tally = Tally::of(samples, protocol.statistic, protocol.confidence)?;
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

    /// 装置身份。
    pub fn harness(&self) -> &Harness {
        &self.harness
    }

    /// 样本摘要；统计量无定义时为 `None`。
    pub fn tally(&self) -> Option<&Tally> {
        self.tally.as_ref()
    }

    /// 三态结论。
    pub fn standing(&self) -> Standing {
        self.standing
    }

    /// 归档 JSON：键序固定，缺测写 `null`，结论旁边一定带着理由（`shortfall`）。
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");
        protocol_fields(&mut out, &self.protocol);
        harness_fields(&mut out, &self.harness);
        tally_field(&mut out, self.tally.as_ref());
        standing_fields(&mut out, self.standing);
        out.push_str("}\n");
        out
    }
}

/// 协议段：判什么、怎么聚、往哪边、门槛多少、要几个样本。
fn protocol_fields(out: &mut String, protocol: &Protocol) {
    out.push_str("  \"protocol\": ");
    quote(out, protocol.name);
    out.push_str(",\n  \"metric\": ");
    quote(out, protocol.metric);
    out.push_str(",\n  \"statistic\": ");
    quote(out, protocol.statistic.as_str());
    out.push_str(",\n  \"direction\": ");
    quote(out, protocol.direction.as_str());
    out.push_str(",\n  \"bound\": ");
    out.push_str(&protocol.bound.to_string());
    out.push_str(",\n  \"least\": ");
    out.push_str(&protocol.least.to_string());
    out.push_str(",\n  \"confidence\": ");
    out.push_str(&protocol.confidence.to_string());
    out.push_str(",\n");
}

/// 装置段：`lld` §10.2 要求「同装置对照」，所以身份与数一起归档。
fn harness_fields(out: &mut String, harness: &Harness) {
    out.push_str("  \"harness\": {\n");
    let parts = [
        ("host", harness.host()),
        ("cpu", harness.cpu()),
        ("os", harness.os()),
        ("arch", harness.arch()),
        ("toolchain", harness.toolchain()),
        ("driver", harness.driver()),
    ];
    for (position, (key, value)) in parts.iter().enumerate() {
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
    match tally {
        None => out.push_str("null,\n"),
        Some(tally) => tally_fields(out, tally),
    }
}

/// 摘要是整数 + 实数混合，计数单独写，其余走 [`real`]。
fn tally_fields(out: &mut String, tally: &Tally) {
    out.push_str("{\n    \"count\": ");
    out.push_str(&tally.count.to_string());
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
    out.push_str(",\n    ");
    quote(out, key);
    out.push_str(": ");
    out.push_str(&value.to_string());
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
///
/// 两方向的差别只有「哪一端算内侧」与「等号归谁」：越大越好时下界含等号即达标，
/// 越小越好时上界必须严格小于门槛（`lld` E10 的「CI 上界 <5% 才 PASS」）。
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
