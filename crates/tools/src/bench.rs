//! 统一基准记录结构（M1-C2）：一条算子 / 融合 / 层 / 整模型 / 服务负载基准的身份、形状、
//! 批次、精度、后端与硬件峰值上界。

use diagnostic::{Error, Result};

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
