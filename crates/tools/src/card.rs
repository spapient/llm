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
//! use llm_tool::bench::Harness;
//! use llm_tool::card::{Card, Direction, Protocol, Standing, Statistic};
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
pub use crate::regress::Direction;

/// 重采样次数：协议卡报的是 95% 区间，两千次抽样的分位点已经稳定到卡片精度之内。
const RESAMPLES: usize = 2000;

/// 重采样种子固定：**同一份样本只会得到一份卡片**，复核时能逐位对拍，报告也不随机漂移。
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

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
            Self::Mean => (!samples.is_empty()).then(|| mean(samples)),
            Self::Cv => {
                let center = mean(samples);
                (samples.len() >= 2 && center != 0.0).then(|| std_dev(samples) / center)
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
        let Some(draws) = interval(samples, statistic) else {
            return Ok(None);
        };
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let tail = (1.0 - confidence) / 2.0;
        let center = mean(samples);
        let sd = std_dev(samples);
        Ok(Some(Self {
            count: samples.len(),
            mean: center,
            sd,
            cv: (samples.len() >= 2 && center != 0.0).then(|| sd / center),
            p50: quantile(&sorted, 0.50),
            p95: quantile(&sorted, 0.95),
            p99: quantile(&sorted, 0.99),
            lower: quantile(&draws, tail),
            upper: quantile(&draws, 1.0 - tail),
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
    write_string(out, protocol.name);
    out.push_str(",\n  \"metric\": ");
    write_string(out, protocol.metric);
    out.push_str(",\n  \"statistic\": ");
    write_string(out, protocol.statistic.as_str());
    out.push_str(",\n  \"direction\": ");
    write_string(out, protocol.direction.as_str());
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
        write_string(out, key);
        out.push_str(": ");
        write_string(out, value);
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
    write_string(out, key);
    out.push_str(": ");
    out.push_str(&value.to_string());
}

/// 结论段：三态名 + 理由；PASS / FAIL 的理由写 `null`。
fn standing_fields(out: &mut String, standing: Standing) {
    out.push_str("  \"standing\": ");
    write_string(out, standing.as_str());
    out.push_str(",\n  \"shortfall\": ");
    match standing.shortfall() {
        None => out.push_str("null"),
        Some(shortfall) => write_string(out, shortfall.as_str()),
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

/// 统计量的重采样区间（已排序）；统计量无定义或有效抽样太少时 `None`。
fn interval(samples: &[f64], statistic: Statistic) -> Option<Vec<f64>> {
    statistic.value(samples)?;
    let mut draws = resample(samples, statistic);
    if draws.len() < RESAMPLES / 2 {
        return None;
    }
    draws.sort_by(f64::total_cmp);
    Some(draws)
}

/// 对统计量做有放回重采样，返回每次抽样上的取值；无定义的抽样直接丢弃。
fn resample(samples: &[f64], statistic: Statistic) -> Vec<f64> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut state = SEED;
    let mut drawn = Vec::with_capacity(samples.len());
    let mut values = Vec::with_capacity(RESAMPLES);
    for _ in 0..RESAMPLES {
        drawn.clear();
        drawn.extend(
            samples
                .iter()
                .map(|_| samples[index(&mut state, samples.len())]),
        );
        if let Some(value) = statistic.value(&drawn) {
            values.push(value);
        }
    }
    values
}

/// splitmix64：状态只有 64 位，重采样可复现，不引入随机依赖。
fn index(state: &mut u64, len: usize) -> usize {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let z = z ^ (z >> 31);
    (z % len as u64) as usize
}

/// 样本均值；空列返回 0（调用方在此之前已排除空列）。
fn mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// 样本标准差（`n − 1` 分母）；少于 2 个样本时返回 0。
fn std_dev(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let center = mean(samples);
    let sum: f64 = samples.iter().map(|value| (value - center).powi(2)).sum();
    (sum / (samples.len() - 1) as f64).sqrt()
}

/// 最近秩分位：`q` 落进哪个秩就取哪个秩，不做插值（样本量小的时候插值会让分位看起来更稳）。
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let last = sorted.len() as f64;
    let rank = (q * last).ceil().clamp(1.0, last);
    sorted[rank as usize - 1]
}

/// JSON 字符串里必须换写法的五个字符。
const ESCAPES: [(char, &str); 5] = [
    ('"', "\\\""),
    ('\\', "\\\\"),
    ('\n', "\\n"),
    ('\r', "\\r"),
    ('\t', "\\t"),
];

/// 写一个 JSON 字符串：常见转义查表，其余控制字符按 `\u00XX`，其他原样。
fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        push_char(out, ch);
    }
    out.push('"');
}

/// 一个字符的 JSON 写法。
fn push_char(out: &mut String, ch: char) {
    if let Some((_, code)) = ESCAPES.iter().find(|(from, _)| *from == ch) {
        out.push_str(code);
        return;
    }
    if ch < ' ' {
        out.push_str(&format!("\\u{:04x}", ch as u32));
        return;
    }
    out.push(ch);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一台在闭集里的 A800：多数用例只关心门槛，不关心装置。
    fn harness(host: &str) -> Harness {
        Harness::new(host, "cpu", "os", "aarch64", "rustc", "driver")
    }

    fn jitter() -> Protocol {
        Protocol::of(
            "E10",
            "decode_step_ns",
            Statistic::Cv,
            Direction::Lower,
            0.05,
        )
        .expect("协议")
    }

    /// 同一协议下抖动为 0 的一列：区间退化成一个点，整段落在门槛内侧。
    #[test]
    fn a_steady_series_passes_the_jitter_gate() {
        let card = Card::judge(jitter(), harness("A800"), &[10.0; 40]).expect("判");
        assert_eq!(card.standing(), Standing::Pass);
        let tally = card.tally().expect("有摘要");
        assert_eq!(tally.cv(), Some(0.0));
        assert_eq!((tally.lower(), tally.upper()), (0.0, 0.0));
        assert_eq!(tally.p95(), 10.0, "常数列每个分位都是它自己");
    }

    /// 抖动远超门槛的一列：区间整段越界，这一格记 FAIL 而不是 OPEN。
    #[test]
    fn a_series_beyond_the_gate_fails() {
        let mut samples = Vec::new();
        for _ in 0..20 {
            samples.extend([1.0, 100.0]);
        }
        let card = Card::judge(jitter(), harness("A800"), &samples).expect("判");
        assert_eq!(card.standing(), Standing::Fail);
        assert!(card.tally().expect("有摘要").cv().expect("有定义") > 0.05);
    }

    /// 区间跨着门槛：样本还不足以把「过」与「不过」分开，只能记 OPEN。
    #[test]
    fn an_interval_across_the_gate_stays_open() {
        let protocol = Protocol::of(
            "E1",
            "tokens_per_s",
            Statistic::Mean,
            Direction::Higher,
            2.0,
        )
        .expect("协议")
        .with_least(3)
        .expect("下限");
        let card = Card::judge(protocol, harness("A800"), &[1.0, 2.0, 3.0]).expect("判");
        assert_eq!(card.standing(), Standing::Open(Shortfall::Straddle));
        assert!(card.tally().expect("有摘要").lower() < 2.0);
        assert!(card.tally().expect("有摘要").upper() > 2.0);
    }

    /// 样本不够时不判：摘要照记，结论只有一个 OPEN 理由，不写「达标」也不写「未达标」。
    #[test]
    fn too_few_samples_are_not_judged_at_all() {
        let card = Card::judge(jitter(), harness("A800"), &[10.0, 12.0, 11.0]).expect("判");
        assert_eq!(card.standing(), Standing::Open(Shortfall::Samples));
        assert_eq!(card.tally().expect("摘要照记").count(), 3);
        assert_eq!(card.standing().shortfall(), Some(Shortfall::Samples));
        assert!(!card.standing().is_pass());
    }

    /// 闭集外的主机：样本再多也不是这份协议的证据。
    #[test]
    fn a_host_outside_the_closure_cannot_pass() {
        let protocol = jitter().with_hosts(&["A800", "RTX_3090", "T600"]);
        let card = Card::judge(protocol, harness("macos-aarch64"), &[10.0; 40]).expect("判");
        assert_eq!(card.standing(), Standing::Open(Shortfall::Host));
    }

    /// 闭集为空表示不设闭集：任何装置都按门槛判。
    #[test]
    fn an_empty_closure_admits_every_host() {
        let card = Card::judge(jitter(), harness("macos-aarch64"), &[10.0; 40]).expect("判");
        assert_eq!(card.standing(), Standing::Pass);
        assert!(card.protocol().hosts().is_empty());
    }

    /// 均值为 0 的列上变异系数没有定义：填 0 会让「完全抖动」看起来像「完全稳定」。
    #[test]
    fn a_zero_mean_series_has_no_jitter_to_report() {
        let mut samples = Vec::new();
        for _ in 0..20 {
            samples.extend([-1.0, 1.0]);
        }
        let card = Card::judge(jitter(), harness("A800"), &samples).expect("判");
        assert_eq!(card.standing(), Standing::Open(Shortfall::Degenerate));
        assert!(card.tally().is_none(), "无定义就不写摘要");
    }

    /// 空白列：一个样本都没有，既不摘要也不判。
    #[test]
    fn no_samples_report_no_summary() {
        let card = Card::judge(jitter(), harness("A800"), &[]).expect("判");
        assert_eq!(card.standing(), Standing::Open(Shortfall::Samples));
        assert!(card.tally().is_none());
    }

    /// 非有限样本是采集坏了：报错，不许静默跳过。
    #[test]
    fn a_broken_sample_is_reported_as_such() {
        let err = Card::judge(jitter(), harness("A800"), &[1.0, f64::NAN]).unwrap_err();
        assert_eq!(err.code(), Code::Invalid);
        assert!(err.message().contains("采集坏了"), "{}", err.message());
    }

    /// 重采样种子固定：同一份样本只会得到一份卡片，复核时能逐位对拍。
    #[test]
    fn the_same_samples_produce_the_same_interval() {
        let samples = [9.0, 11.0, 10.5, 9.4, 12.0, 10.1, 9.9, 10.8];
        let first = Card::judge(jitter(), harness("A800"), &samples).expect("判");
        let second = Card::judge(jitter(), harness("A800"), &samples).expect("判");
        assert_eq!(
            first.tally().map(|tally| (tally.lower(), tally.upper())),
            second.tally().map(|tally| (tally.lower(), tally.upper()))
        );
    }

    /// 分位不许交叉：中位 ≤ 95 ≤ 99，区间下界 ≤ 上界。
    #[test]
    fn the_summary_never_contradicts_itself() {
        let samples: Vec<f64> = (1..=100).map(f64::from).collect();
        let tally = Tally::of(&samples, Statistic::Mean, 0.95)
            .expect("摘要")
            .expect("有定义");
        assert!(tally.p50() <= tally.p95());
        assert!(tally.p95() <= tally.p99());
        assert!(tally.lower() <= tally.upper());
        assert!(tally.lower() <= tally.mean() && tally.mean() <= tally.upper());
        assert_eq!(tally.count(), 100);
    }

    /// 归档里结论与理由必须相邻：读者不该靠推断知道这次是 OPEN 还是 FAIL。
    #[test]
    fn the_archive_keeps_the_reason_next_to_the_verdict() {
        let open = Card::judge(jitter(), harness("A800"), &[10.0])
            .expect("判")
            .to_json();
        assert!(open.contains("\"standing\": \"open\""), "{open}");
        assert!(open.contains("\"shortfall\": \"samples\""), "{open}");
        assert!(open.contains("\"statistic\": \"cv\""), "{open}");
        assert!(open.contains("\"host\": \"A800\""), "{open}");

        let mut zero_mean = Vec::new();
        for _ in 0..15 {
            zero_mean.extend([-1.0, 1.0]);
        }
        let degenerate = Card::judge(jitter(), harness("A800"), &zero_mean)
            .expect("判")
            .to_json();
        assert!(degenerate.contains("\"tally\": null"), "{degenerate}");
        assert!(
            degenerate.contains("\"shortfall\": \"degenerate\""),
            "{degenerate}"
        );
    }

    /// 短名是归档口径：改措辞属于破坏性改动。
    #[test]
    fn the_short_names_are_stable() {
        assert_eq!(Statistic::Mean.as_str(), "mean");
        assert_eq!(Statistic::Cv.as_str(), "cv");
        assert_eq!(Shortfall::Host.as_str(), "host");
        assert_eq!(Shortfall::Samples.as_str(), "samples");
        assert_eq!(Shortfall::Degenerate.as_str(), "degenerate");
        assert_eq!(Shortfall::Straddle.as_str(), "straddle");
        assert_eq!(Standing::Pass.as_str(), "pass");
        assert_eq!(Standing::Fail.as_str(), "fail");
        assert_eq!(Standing::Open(Shortfall::Host).as_str(), "open");
        assert_eq!(
            Standing::Open(Shortfall::Host).shortfall(),
            Some(Shortfall::Host)
        );
    }

    /// 构造期拒绝不可能成立的协议：空名、非有限门槛、下限 <2、置信水平越界。
    #[test]
    fn a_protocol_rejects_impossible_settings() {
        let empty = Protocol::of("", "ns", Statistic::Mean, Direction::Lower, 1.0).unwrap_err();
        assert_eq!(empty.code(), Code::Invalid);
        let nan =
            Protocol::of("E1", "ns", Statistic::Mean, Direction::Lower, f64::NAN).unwrap_err();
        assert_eq!(nan.code(), Code::Invalid);
        assert!(nan.message().contains("有限数"), "{}", nan.message());
        assert_eq!(jitter().with_least(1).unwrap_err().code(), Code::Invalid);
        assert_eq!(
            jitter().with_confidence(1.0).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(
            jitter().with_confidence(0.0).unwrap_err().code(),
            Code::Invalid
        );
        assert!(jitter().with_confidence(0.99).is_ok());
    }
}
