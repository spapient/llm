//! 统一基准记录结构（M1-C2）：一条算子 / 融合 / 层 / 整模型 / 服务负载基准的身份、形状、
//! 批次、精度、后端与硬件峰值上界。

use diagnostic::{Error, Result};

use crate::stats::mean;

/// 配置错误：上下文统一带 `tools::bench::` 前缀。
fn invalid(ctx: &str, message: impl Into<String>) -> Error {
    crate::error::invalid("bench", ctx, message)
}

/// 条目层级：算子内核、融合复合、层、整模型、服务负载。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntryKind {
    /// 单条算子内核基准。
    Op,
    /// 多算子融合 / 复合路径基准（一条调用链内含多个算子）。
    Fused,
    /// 一层组合基准。
    Layer,
    /// 整模型一步基准。
    Model,
    /// 服务化负载基准（请求流 / 并发；类型位先占位，测量落地归 P4）。
    Serving,
}

impl EntryKind {
    /// 稳定短名，写进归档报告。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Op => "op",
            Self::Fused => "fused",
            Self::Layer => "layer",
            Self::Model => "model",
            Self::Serving => "serving",
        }
    }
}

/// 硬件与工具链身份。跨机绝对值不可比，基线按这套身份登记；字段全部由调用方从运行环境
/// 采集后填入，本类型不读环境、不猜平台。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Harness {
    host: String,
    cpu: String,
    os: String,
    arch: String,
    toolchain: String,
    driver: String,
}

impl Harness {
    /// 逐条身份字段构造；空串由调用方显式给 `"unknown"`，不在这里代填。
    pub fn new(
        host: impl Into<String>,
        cpu: impl Into<String>,
        os: impl Into<String>,
        arch: impl Into<String>,
        toolchain: impl Into<String>,
        driver: impl Into<String>,
    ) -> Self {
        Self {
            host: host.into(),
            cpu: cpu.into(),
            os: os.into(),
            arch: arch.into(),
            toolchain: toolchain.into(),
            driver: driver.into(),
        }
    }

    /// 主机标识（`target/bench-report/<host>/` 的那一级）。
    pub fn host(&self) -> &str {
        &self.host
    }

    /// CPU / 设备型号。
    pub fn cpu(&self) -> &str {
        &self.cpu
    }

    /// 操作系统名。
    pub fn os(&self) -> &str {
        &self.os
    }

    /// 指令集架构。
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// 工具链版本。
    pub fn toolchain(&self) -> &str {
        &self.toolchain
    }

    /// 后端驱动 / 运行时标识（CPU 参考填 `cpu-reference`）。
    pub fn driver(&self) -> &str {
        &self.driver
    }
}

/// 硬件峰值画像：同一硬件的峰值带宽与峰值算力上界。任一项未测得即留 [`None`]，
/// 不拿标称值或他机数字顶替。与报告侧的达成率口径（`Attainment`）区分：本类型只持有硬件峰值。
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PeakProfile {
    peak_bw_gb_s: Option<f64>,
    peak_gflops: Option<f64>,
}

impl PeakProfile {
    /// 构造上界；给出的值必须是有限且非负的。
    pub fn new(peak_bw_gb_s: Option<f64>, peak_gflops: Option<f64>) -> Result<Self> {
        for (label, value) in [("peak_bw_gb_s", peak_bw_gb_s), ("peak_gflops", peak_gflops)] {
            match value {
                Some(number) if !number.is_finite() || number < 0.0 => {
                    return Err(invalid(
                        "PeakProfile::new",
                        format!("{label} 必须是非负有限数，得到 {number}"),
                    ));
                }
                _ => {}
            }
        }
        Ok(Self {
            peak_bw_gb_s,
            peak_gflops,
        })
    }

    /// 峰值带宽（GB/s）；未登记为 `None`。
    pub fn peak_bw_gb_s(&self) -> Option<f64> {
        self.peak_bw_gb_s
    }

    /// 峰值算力（GFLOP/s）；未登记为 `None`。
    pub fn peak_gflops(&self) -> Option<f64> {
        self.peak_gflops
    }

    /// 打分：由算术强度（FLOP/byte）与实测吞吐（GFLOP/s）折成占端口峰值的百分比。
    ///
    /// 上界是带宽侧与算力侧两条 roofline 的较慢者：`min(峰值算力, 峰值带宽 × 算术强度)`。
    /// 两个峰值都未登记时无法打分，返回错误而不是拿标称值顶替。
    pub fn score(&self, arithmetic_intensity: f64, achieved_gflops: f64) -> Result<Score> {
        check_ratio(arithmetic_intensity, "arithmetic_intensity")?;
        check_ratio(achieved_gflops, "achieved_gflops")?;
        let (peak_bw, peak_flops) = self.peaks()?;
        let bandwidth_ceiling = peak_bw * arithmetic_intensity;
        let (ceiling, side) = if bandwidth_ceiling < peak_flops {
            (bandwidth_ceiling, Bottleneck::Bandwidth)
        } else {
            (peak_flops, Bottleneck::Compute)
        };
        let percent = achieved_gflops / ceiling * 100.0;
        Ok(Score {
            ceiling_gflops: ceiling,
            percent,
            side,
        })
    }

    /// 两个峰值都要登记，否则无法折成百分比。
    fn peaks(&self) -> Result<(f64, f64)> {
        let Some(peak_bw) = self.peak_bw_gb_s else {
            return Err(invalid("PeakProfile::score", "带宽峰值未登记，无法打分"));
        };
        let Some(peak_flops) = self.peak_gflops else {
            return Err(invalid("PeakProfile::score", "算力峰值未登记，无法打分"));
        };
        Ok((peak_bw, peak_flops))
    }
}

/// 瓶颈侧：roofline 里绑定上界的那一侧。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    /// 带宽侧：`峰值带宽 × 算术强度` 低于峰值算力，访存是上界。
    Bandwidth,
    /// 算力侧：峰值算力更低，计算是上界。
    Compute,
}

impl Bottleneck {
    /// 稳定短名，写进报告。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bandwidth => "bandwidth",
            Self::Compute => "compute",
        }
    }
}

/// 一次 roofline 打分结果：上界吞吐、占上界百分比与瓶颈侧。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    ceiling_gflops: f64,
    percent: f64,
    side: Bottleneck,
}

impl Score {
    /// 上界吞吐（GFLOP/s）。
    pub fn ceiling_gflops(&self) -> f64 {
        self.ceiling_gflops
    }

    /// 占上界百分比（%）。
    pub fn percent(&self) -> f64 {
        self.percent
    }

    /// 瓶颈侧。
    pub fn side(&self) -> Bottleneck {
        self.side
    }
}

/// 被测问题的几何：形状、批次与元素精度。
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    shape: Vec<usize>,
    batch: u32,
    dtype: String,
}

impl Geometry {
    /// 构造几何：形状至少一维、批次至少为 1、精度标识非空。
    pub fn new(shape: Vec<usize>, batch: u32, dtype: impl Into<String>) -> Result<Self> {
        let dtype = dtype.into();
        check_shape_batch(&shape, batch)?;
        if dtype.is_empty() {
            return Err(invalid("Geometry::new", "精度标识不能为空"));
        }
        Ok(Self {
            shape,
            batch,
            dtype,
        })
    }

    /// 形状维度。
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// 批大小。
    pub fn batch(&self) -> u32 {
        self.batch
    }

    /// 元素精度标识。
    pub fn dtype(&self) -> &str {
        &self.dtype
    }
}

/// 一条统一基准记录：身份 + 几何 + 后端 + 硬件峰值画像 + 可选实测。
#[derive(Debug, Clone, PartialEq)]
pub struct BenchRecord {
    kind: EntryKind,
    name: String,
    backend: String,
    geometry: Geometry,
    harness: Harness,
    peak_profile: PeakProfile,
    observed: Option<f64>,
}

impl BenchRecord {
    /// 构造一条尚未测量的记录。
    pub fn new(
        kind: EntryKind,
        name: impl Into<String>,
        backend: impl Into<String>,
        geometry: Geometry,
        harness: Harness,
        peak_profile: PeakProfile,
    ) -> Result<Self> {
        let name = name.into();
        let backend = backend.into();
        check_strings(&name, &backend)?;
        Ok(Self {
            kind,
            name,
            backend,
            geometry,
            harness,
            peak_profile,
            observed: None,
        })
    }

    /// 回填一次实测值；值必须是有限数。未跑测量时保持 `None`，不补零。
    pub fn with_observed(mut self, observed: f64) -> Result<Self> {
        if !observed.is_finite() {
            return Err(invalid(
                "BenchRecord::with_observed",
                format!("实测值必须是有限数，得到 {observed}"),
            ));
        }
        self.observed = Some(observed);
        Ok(self)
    }

    /// 条目层级。
    pub fn kind(&self) -> EntryKind {
        self.kind
    }

    /// 条目名（算子名 / 层名 / 模型名）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 后端标识。
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// 被测几何（形状 / 批次 / 精度）。
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// 硬件与工具链身份。
    pub fn harness(&self) -> &Harness {
        &self.harness
    }

    /// 硬件峰值画像（带宽 / 算力上界）。
    pub fn peak_profile(&self) -> &PeakProfile {
        &self.peak_profile
    }

    /// 实测值；未测量为 `None`，不伪造。
    pub fn observed(&self) -> Option<f64> {
        self.observed
    }
}

/// 名字与后端标识不能为空；逐条拒绝，消息带字段名。
fn check_strings(name: &str, backend: &str) -> Result<()> {
    if name.is_empty() {
        return Err(invalid("BenchRecord::new", "条目名不能为空"));
    }
    if backend.is_empty() {
        return Err(invalid("BenchRecord::new", "后端标识不能为空"));
    }
    Ok(())
}

/// 形状至少一维、批次至少为 1。
fn check_shape_batch(shape: &[usize], batch: u32) -> Result<()> {
    if shape.is_empty() {
        return Err(invalid("Geometry::new", "形状至少要有一维"));
    }
    if batch == 0 {
        return Err(invalid("Geometry::new", "批次不能为 0"));
    }
    Ok(())
}

/// 算术强度与实测吞吐必须是非负有限数。
fn check_ratio(value: f64, label: &str) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(invalid(
            "PeakProfile::score",
            format!("{label} 必须是非负有限数，得到 {value}"),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 形状网格点：tile 整除对齐点与满块减一碎片边界点。每形状 FLOPs / 字节仍由 `nn::Bounds`
// 语义表推出，这里不写第二份流量常数。
// ---------------------------------------------------------------------------

/// tile 档点生成器：`tile·q + offset`（对齐点 offset 为 0，碎片边界点为 −1），不超过 `max`；
/// `tile` 为 0 时为空。`i64` 中转是为了让 `−1` 偏移不在无符号域下溢。
fn tile_points(tile: usize, max: usize, offset: i64) -> Vec<usize> {
    if tile == 0 {
        return Vec::new();
    }
    (1..)
        .map(|q| (q as i64 * tile as i64 + offset) as usize)
        .take_while(|&point| point <= max)
        .collect()
}

/// 整除对齐点：`tile` 的整数倍，不超过 `max`；`tile` 为 0 或 `max` 小于 `tile` 时为空。
#[must_use]
pub fn aligned_points(tile: usize, max: usize) -> Vec<usize> {
    tile_points(tile, max, 0)
}

/// 非整除边界点：每档 `tile·q − 1`（满块减一的碎片形状），不超过 `max`。
///
/// 覆盖 tile 量化余量：分块内核在 `tile·q − 1` 这种最后一块不满的形状上最容易出边界错误，
/// 对齐点测不到这条路径。
#[must_use]
pub fn boundary_points(tile: usize, max: usize) -> Vec<usize> {
    tile_points(tile, max, -1)
}

// ---------------------------------------------------------------------------
// 分配计数值对象：一次基准运行的堆分配读数。计数分配器挂载与高水位采样属运行时接入位。
// ---------------------------------------------------------------------------

/// 一次运行的堆分配读数：次数、累计分配字节、驻留峰值字节。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocProfile {
    count: u64,
    bytes: u64,
    peak_bytes: u64,
}

impl AllocProfile {
    /// 登记一份分配读数；三项均为计数口径，由计数分配器在采集侧填充。
    pub fn new(count: u64, bytes: u64, peak_bytes: u64) -> Self {
        Self {
            count,
            bytes,
            peak_bytes,
        }
    }

    /// 分配次数。
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// 本次运行累计分配字节（含已释放段，是分配总量而非驻留量）。
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// 驻留高水位字节。
    #[must_use]
    pub fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }
}

// ---------------------------------------------------------------------------
// 服务化基准：场景 / 到达过程闭集与分相时延 / goodput 口径。只做确定性统计，不读时钟、
// 不起服务、不做调度；逐 token 产出时刻由采集层喂入。
// ---------------------------------------------------------------------------

/// 服务负载场景：对齐 MLPerf 四档，LLM 在此之上拆 TTFT / TPOT / ITL / E2EL。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ServingScene {
    /// 单序列串行：一次一个请求，测首 token 与单步时延。
    SingleStream,
    /// 多序列同时在途，按批次完成度报吞吐。
    MultiStream,
    /// 开环服务：Poisson 到达，暴露排队与尾延迟，是 goodput 与 SLO 测量前提。
    Server,
    /// 离线批量：无到达约束，只测吞吐天花板。
    Offline,
}

impl ServingScene {
    /// 稳定短名，写进归档报告。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SingleStream => "single-stream",
            Self::MultiStream => "multi-stream",
            Self::Server => "server",
            Self::Offline => "offline",
        }
    }
}

/// 请求到达过程：开环与闭环是两条不同的问题，不可互换。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arrival {
    /// 开环：到达率独立于响应，暴露排队与尾延迟。
    OpenPoisson,
    /// 闭环：固定并发、完成即补，测吞吐天花板，但 back-pressure 会掩盖变慢。
    ClosedConcurrency,
}

impl Arrival {
    /// 稳定短名，写进归档报告。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenPoisson => "open-poisson",
            Self::ClosedConcurrency => "closed-concurrency",
        }
    }
}

/// 一条请求的分相时延（单位 ns）。
///
/// `ttft` 是首 token 时刻（相对计时起点）；`itl` 是相邻 token 间隔列；`tpot` 是后续间隔的均值；
/// `e2e` 是末 token 时刻（端到端）。单 token 请求 `itl` 为空、`tpot` 为 0。
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseLatency {
    ttft_ns: f64,
    tpot_ns: f64,
    e2e_ns: f64,
    itl: Vec<f64>,
}

impl PhaseLatency {
    /// 首 token 时延（ns）。
    #[must_use]
    pub fn ttft_ns(&self) -> f64 {
        self.ttft_ns
    }

    /// 后续 token 平均步长（ns）。
    #[must_use]
    pub fn tpot_ns(&self) -> f64 {
        self.tpot_ns
    }

    /// 端到端时延（ns）。
    #[must_use]
    pub fn e2e_ns(&self) -> f64 {
        self.e2e_ns
    }

    /// 相邻 token 间隔列（ns）。
    #[must_use]
    pub fn itl(&self) -> &[f64] {
        &self.itl
    }
}

/// 由 token 产出时刻（ns，相对计时起点，单调不减）拆分相时延。
///
/// 时刻为空、非有限或回退（后一个早于前一个）时返回 [`None`]：口径不猜，也不把乱序时序
/// 折成看似平稳的间隔。
#[must_use]
pub fn phase_latency(marks: &[f64]) -> Option<PhaseLatency> {
    if marks.is_empty() || !marks[0].is_finite() {
        return None;
    }
    let itl: Vec<f64> = marks.windows(2).map(|pair| pair[1] - pair[0]).collect();
    if itl.iter().any(|gap| !gap.is_finite() || *gap < 0.0) {
        return None;
    }
    let tpot = mean(&itl);
    Some(PhaseLatency {
        ttft_ns: marks[0],
        tpot_ns: tpot,
        e2e_ns: marks[marks.len() - 1],
        itl,
    })
}

/// SLO 达成率：满足 SLO 的请求占比，钳制在 `[0, 1]`；总数为 0 时返回 0，不除零。
#[must_use]
pub fn slo_hit_rate(met: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        met.min(total) as f64 / total as f64
    }
}

/// 归一吞吐：每纳秒 token 数折成 tokens/s；时长非正返回 0，不拿零分母出 inf。
#[must_use]
pub fn tokens_per_s(tokens: u64, elapsed_ns: f64) -> f64 {
    if elapsed_ns > 0.0 {
        tokens as f64 / elapsed_ns * 1e9
    } else {
        0.0
    }
}

/// goodput：满足 SLO 的有效吞吐（tokens/s），= 归一吞吐 × SLO 达成率。
///
/// 总吞吐里不满足 SLO 的部分不计入 goodput；达成率由 [`slo_hit_rate`] 单独给出，便于报告
/// 同时列出原始吞吐与有效吞吐。
#[must_use]
pub fn goodput_tokens_per_s(tokens: u64, elapsed_ns: f64, met: usize, total: usize) -> f64 {
    tokens_per_s(tokens, elapsed_ns) * slo_hit_rate(met, total)
}
