//! 统一基准记录结构（M1-C2）：一条算子 / 层 / 整模型基准的身份、形状、批次、精度、后端与
//! roofline 上界。
//!
//! 本模块只持有**记录**，不测量、不写性能结论：实测值由调用方在跑完后填入，基线是否登记
//! 由 [`regress::Baseline`](crate::regress::Baseline) 判定。没有登记基线的条目跳过回退判定，
//! 不在此伪造基线数字；未测得的硬件峰值上界留 [`None`](Option::None)，不预填。
//!
//! **拥有**：[`EntryKind`]、[`Harness`]、[`Roofline`]、[`Geometry`]、[`BenchRecord`]。
//! **不拥有**：测量采样、基线台账、跨机对照结论。

use diagnostic::{Code, Error, Result};

/// 配置错误：上下文统一带 `tool::bench::` 前缀。
fn invalid(context: &str, message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(format!("tool::bench::{context}"))
}

/// 条目层级：算子内核、层、整模型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntryKind {
    /// 单条算子基准。
    Op,
    /// 一层组合基准。
    Layer,
    /// 整模型一步基准。
    Model,
}

impl EntryKind {
    /// 稳定短名，写进归档报告。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Op => "op",
            Self::Layer => "layer",
            Self::Model => "model",
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

/// roofline 上界：同一硬件的峰值带宽与峰值算力。任一项未测得即留 [`None`](Option::None)，
/// 不拿标称值或他机数字顶替。
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Roofline {
    peak_bw_gb_s: Option<f64>,
    peak_gflops: Option<f64>,
}

impl Roofline {
    /// 构造上界；给出的值必须是有限且非负的。
    pub fn new(peak_bw_gb_s: Option<f64>, peak_gflops: Option<f64>) -> Result<Self> {
        for (label, value) in [("peak_bw_gb_s", peak_bw_gb_s), ("peak_gflops", peak_gflops)] {
            match value {
                Some(number) if !number.is_finite() || number < 0.0 => {
                    return Err(invalid(
                        "Roofline::new",
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

/// 一条统一基准记录：身份 + 几何 + 后端 + roofline 上界 + 可选实测。
#[derive(Debug, Clone, PartialEq)]
pub struct BenchRecord {
    kind: EntryKind,
    name: String,
    backend: String,
    geometry: Geometry,
    harness: Harness,
    roofline: Roofline,
    observed: Option<f64>,
}

impl BenchRecord {
    /// 构造一条尚未测量的记录。
    ///
    /// 校验：名字 / 后端非空；几何由 [`Geometry`] 自检。形状为整数维度，不存在负维度，
    /// `usize` 已兜住越界方向。
    pub fn new(
        kind: EntryKind,
        name: impl Into<String>,
        backend: impl Into<String>,
        geometry: Geometry,
        harness: Harness,
        roofline: Roofline,
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
            roofline,
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

    /// roofline 上界。
    pub fn roofline(&self) -> &Roofline {
        &self.roofline
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

#[cfg(test)]
#[path = "../tests/bench_tests.rs"]
mod tests;
