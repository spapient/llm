//! 统一基准记录结构（M1-C2）：一条算子 / 层 / 整模型基准的身份、形状、批次、精度、后端与
//! roofline 上界。
//!
//! 本模块只持有**记录**，不测量、不写性能结论：实测值由调用方在跑完后填入，基线是否登记
//! 由 [`regress::Baseline`](crate::regress::Baseline) 判定。没有登记基线的条目跳过回退判定，
//! 不在此伪造基线数字；未测得的硬件峰值上界留 [`None`]，不预填。
//!
//! **拥有**：[`EntryKind`]、[`Harness`]、[`Roofline`]、[`Geometry`]、[`BenchRecord`]。
//! **不拥有**：测量采样、基线台账、跨机对照结论。

use diagnostic::{Code, Error, Result};

/// 配置错误：上下文统一带 `tools::bench::` 前缀。
fn invalid(context: &str, message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(format!("tools::bench::{context}"))
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

/// roofline 上界：同一硬件的峰值带宽与峰值算力。任一项未测得即留 [`None`]，
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
mod tests {
    use super::*;
    use crate::bench::json::{from_json, to_json};
    use diagnostic::Code;

    fn harness() -> Harness {
        Harness::new(
            "host-a",
            "cpu-x",
            "macos",
            "aarch64",
            "rustc 1.82",
            "cpu-reference",
        )
    }

    fn geometry(shape: Vec<usize>, batch: u32, dtype: &str) -> Result<Geometry> {
        Geometry::new(shape, batch, dtype)
    }

    fn record(name: &str, backend: &str, geom: Geometry) -> Result<BenchRecord> {
        BenchRecord::new(
            EntryKind::Op,
            name,
            backend,
            geom,
            harness(),
            Roofline::default(),
        )
    }

    /// 负例夹具共用的档案与 roofline 片段（身份值在各用例里保持一致）。
    const HARNESS_JSON: &str =
        r#"{"host": "h", "cpu": "c", "os": "o", "arch": "a", "toolchain": "t", "driver": "d"}"#;
    const ROOFLINE_JSON: &str = r#"{"peak_bw_gb_s": null, "peak_gflops": null}"#;

    /// 拼几何对象片段：`shape` / `batch` 已是 JSON 词法，`dtype` 作为字符串转义。
    fn geometry_json(shape: &str, batch: &str, dtype: &str) -> String {
        format!(r#"{{"shape": {shape}, "batch": {batch}, "dtype": {dtype:?}}}"#)
    }

    /// 一条基准记录夹具：身份与档案片段固定，仅 kind / 几何 / 实测三档可覆盖。
    fn op_record(geometry: &str, observed: &str, kind: &str) -> String {
        record_json(&[
            ("kind", kind),
            ("name", r#""x""#),
            ("backend", r#""b""#),
            ("geometry", geometry),
            ("harness", HARNESS_JSON),
            ("roofline", ROOFLINE_JSON),
            ("observed", observed),
        ])
    }

    /// 按 `(键, JSON 值片段)` 拼一条记录对象；缺字段的用例只省略对应条目。
    fn record_json(fields: &[(&str, &str)]) -> String {
        let inner = fields
            .iter()
            .map(|(key, value)| format!(r#""{key}": {value}"#))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{{{inner}}}")
    }

    #[test]
    fn a_minimal_record_keeps_observed_and_roofline_optional() {
        let geom = geometry(vec![4096, 4096], 1, "f32").expect("几何合法");
        let record = record("gemv_f32", "cpu-reference", geom).expect("记录合法");
        assert_eq!(record.kind(), EntryKind::Op);
        assert_eq!(record.name(), "gemv_f32");
        assert_eq!(record.backend(), "cpu-reference");
        assert_eq!(record.geometry().shape(), &[4096, 4096]);
        assert_eq!(record.geometry().batch(), 1);
        assert_eq!(record.geometry().dtype(), "f32");
        assert!(record.observed().is_none(), "未测量不预填实测");
        assert!(record.roofline().peak_bw_gb_s().is_none());
        assert!(record.roofline().peak_gflops().is_none());
    }

    #[test]
    fn observed_is_filled_only_when_measured() {
        let geom = geometry(vec![4096], 1, "f32").expect("几何合法");
        let filled = record("rope_f32", "cpu-reference", geom)
            .expect("记录合法")
            .with_observed(12.5)
            .expect("实测有限");
        assert_eq!(filled.observed(), Some(12.5));

        let geom = geometry(vec![4096], 1, "f32").expect("几何合法");
        let nan = record("rope_f32", "cpu-reference", geom)
            .expect("记录合法")
            .with_observed(f64::NAN);
        assert!(nan.is_err(), "非有限实测不能回填");
        assert_eq!(nan.unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn empty_identity_and_invalid_geometry_are_rejected() {
        let cases: [(&str, &str, Result<Geometry>); 5] = [
            ("", "cpu-reference", geometry(vec![4], 1, "f32")),
            ("gemv", "", geometry(vec![4], 1, "f32")),
            ("gemv", "cpu-reference", geometry(vec![], 1, "f32")),
            ("gemv", "cpu-reference", geometry(vec![4], 0, "f32")),
            ("gemv", "cpu-reference", geometry(vec![4], 1, "")),
        ];
        for (name, backend, geom) in cases {
            let geom = match geom {
                Ok(geom) => geom,
                Err(_) => continue,
            };
            let err = record(name, backend, geom).expect_err("必须拒绝");
            assert_eq!(err.code(), Code::Invalid, "输入：{name}/{backend}");
        }
    }

    #[test]
    fn invalid_geometry_itself_is_an_error() {
        assert!(geometry(vec![], 1, "f32").is_err());
        assert!(geometry(vec![4], 0, "f32").is_err());
        assert!(geometry(vec![4], 1, "").is_err());
    }

    #[test]
    fn roofline_rejects_negative_or_nonfinite_peaks() {
        assert!(Roofline::new(Some(-1.0), None).is_err());
        assert!(Roofline::new(None, Some(f64::INFINITY)).is_err());
        let ok = Roofline::new(Some(1024.0), Some(512.0)).expect("上界合法");
        assert_eq!(ok.peak_bw_gb_s(), Some(1024.0));
        assert_eq!(ok.peak_gflops(), Some(512.0));
    }

    #[test]
    fn entry_kind_short_names_are_stable() {
        assert_eq!(EntryKind::Op.as_str(), "op");
        assert_eq!(EntryKind::Layer.as_str(), "layer");
        assert_eq!(EntryKind::Model.as_str(), "model");
    }

    fn filled_record() -> BenchRecord {
        let geom = geometry(vec![4096, 4096], 4, "f32").expect("几何合法");
        let roofline = Roofline::new(Some(1024.0), Some(512.0)).expect("上界合法");
        BenchRecord::new(
            EntryKind::Model,
            "batch4_step",
            "cpu-reference",
            geom,
            harness(),
            roofline,
        )
        .expect("记录合法")
        .with_observed(1_500_000.0)
        .expect("实测有限")
    }

    fn assert_round_trip(record: &BenchRecord) {
        let text = to_json(record);
        let parsed = from_json(&text).expect("回读合法");
        assert_eq!(&parsed, record);
    }

    #[test]
    fn json_round_trips_record_without_observed() {
        let geom = geometry(vec![4096], 1, "f32").expect("几何合法");
        let record = record("gemv \"v1\" / \\x", "cpu-reference", geom).expect("记录合法");
        assert_round_trip(&record);

        let parsed = from_json(&to_json(&record)).expect("回读合法");
        assert!(parsed.observed().is_none());
        assert!(parsed.roofline().peak_bw_gb_s().is_none());
        assert!(parsed.roofline().peak_gflops().is_none());
    }

    #[test]
    fn json_round_trips_record_with_observed_and_roofline() {
        let record = filled_record();
        assert_round_trip(&record);

        let parsed = from_json(&to_json(&record)).expect("回读合法");
        assert_eq!(parsed.kind(), EntryKind::Model);
        assert_eq!(parsed.observed(), Some(1_500_000.0));
        assert_eq!(parsed.roofline().peak_bw_gb_s(), Some(1024.0));
        assert_eq!(parsed.roofline().peak_gflops(), Some(512.0));
    }

    #[test]
    fn json_rejects_malformed_and_truncated_input() {
        let cases = [
            "{",
            "{\"kind\": \"op\"",
            "{\"geometry\": {\"shape\": [4",
            "{\"name\": \"x\"",
            "[1, 2]",
            "{} trailing",
            "not json",
        ];
        for text in cases {
            let err = from_json(text).expect_err("截断 / 非法输入必须拒绝");
            assert_eq!(err.code(), Code::Invalid, "输入：{text}");
        }
    }

    #[test]
    fn json_rejects_duplicate_keys() {
        let top_level = format!(
            r#"{{"kind": "op", "kind": "layer", "name": "x", "backend": "b",
           "geometry": {}, "harness": {}, "roofline": {}, "observed": null}}"#,
            geometry_json("[1]", "1", "f32"),
            HARNESS_JSON,
            ROOFLINE_JSON,
        );
        let err = from_json(&top_level).expect_err("顶层重复键必须拒绝");
        assert_eq!(err.code(), Code::Invalid);

        let nested = format!(
            r#"{{"kind": "op", "name": "x", "backend": "b",
           "geometry": {{"shape": [1], "shape": [2], "batch": 1, "dtype": "f32"}},
           "harness": {}, "roofline": {}, "observed": null}}"#,
            HARNESS_JSON, ROOFLINE_JSON,
        );
        let err = from_json(&nested).expect_err("几何对象内重复键必须拒绝");
        assert_eq!(err.code(), Code::Invalid);
    }

    #[test]
    fn json_rejects_non_finite_and_wrong_typed_values() {
        let op_geometry = geometry_json("[1]", "1", "f32");
        let base = |observed: &str| -> String { op_record(&op_geometry, observed, r#""op""#) };
        let dense = op_record(&op_geometry, "null", r#""dense""#);
        let batch_fraction = op_record(&geometry_json("[1]", "1.5", "f32"), "null", r#""op""#);
        let shape_text = op_record(&geometry_json(r#"["x"]"#, "1", "f32"), "null", r#""op""#);
        let cases = [
            base("NaN"),
            base("Infinity"),
            base("-Infinity"),
            base(r#""12""#),
            dense,
            batch_fraction,
            shape_text,
        ];
        for text in cases {
            let err = from_json(&text).expect_err("坏数值 / 类型不符必须拒绝");
            assert_eq!(err.code(), Code::Invalid, "输入：{text}");
        }
    }

    #[test]
    fn json_rejects_missing_required_fields() {
        let cases = ["kind", "name", "backend", "geometry", "harness", "roofline"]
            .iter()
            .map(|omit| record_without(omit))
            .collect::<Vec<_>>();
        for text in cases {
            let err = from_json(&text).expect_err("缺必填字段必须拒绝");
            assert_eq!(err.code(), Code::Invalid, "输入：{text}");
        }
    }

    /// 全字段记录里删掉 `omitted` 这一键，其余片段与正例保持一致。
    fn record_without(omitted: &str) -> String {
        let mut fields: Vec<(String, String)> = Vec::new();
        let mut add = |key: &str, value: String| {
            if omitted != key {
                fields.push((key.to_string(), value));
            }
        };
        add("kind", "\"op\"".to_string());
        add("name", "\"x\"".to_string());
        add("backend", "\"b\"".to_string());
        add("geometry", geometry_json("[1]", "1", "f32"));
        add("harness", HARNESS_JSON.to_string());
        add("roofline", ROOFLINE_JSON.to_string());
        add("observed", "null".to_string());
        let refs: Vec<(&str, &str)> = fields
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        record_json(&refs)
    }
}
