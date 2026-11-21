//! 统一记录构造：几何、上界、负载与模型记录。

use super::*;

/// 由场景描述符构造几何（形状 / 批次 / 精度），构造自检失败属于登记数据错误，直接 panic。
pub(crate) fn geometry_of(scenario: &Scenario) -> Geometry {
    Geometry::new(scenario.shape.to_vec(), scenario.batch, scenario.dtype)
        .expect("场景几何合法：形状 / 批次 / 精度通过构造自检")
}

/// 构造一条算子基准的统一记录，并回填实测 ns/iter。
///
/// 整机峰值带宽 / 算力没有在本机测得：上界字段留 [`None`]，不拿标称值或他机数字顶替。分档
/// 搬运输送率属于 Report 侧的 roofline 分析件，不重复塞进记录。
pub(crate) fn record_for(host: &HostProfile, scenario: &Scenario, observed: f64) -> BenchRecord {
    let record = BenchRecord::new(
        EntryKind::Op,
        scenario.scenario,
        scenario.backend,
        geometry_of(scenario),
        harness_of(host, scenario.backend),
        RecordRoofline::default(),
    )
    .expect("算子记录合法：身份 / 几何 / 后端通过构造自检");
    record.with_observed(observed).expect("实测有限")
}

/// 构造一条上界基准的统一记录：实测搬运速率登记进峰值带宽字段，峰值算力未测得留 [`None`]。
pub(crate) fn ceiling_record(host: &HostProfile, ceiling: &Ceiling, observed: f64) -> BenchRecord {
    let geometry = Geometry::new(vec![ceiling.bytes as usize], 1, "u8")
        .expect("上界几何合法：每迭代字节数作为一维");
    // MB/s（十进制）→ GB/s：记录上界字段按 GB/s 口径。
    let peak_bw_gb_s = mbps(ceiling) * BYTES_PER_MB / BYTES_PER_GB;
    let roofline = RecordRoofline::new(Some(peak_bw_gb_s), None).expect("上界带宽非负有限");
    let name = format!("mem_stream/{}_{}", ceiling.tier, ceiling.traffic);
    let record = BenchRecord::new(
        EntryKind::Op,
        name,
        "cpu-reference",
        geometry,
        harness_of(host, "cpu-reference"),
        roofline,
    )
    .expect("上界记录合法：身份 / 几何 / 后端通过构造自检");
    record.with_observed(observed).expect("实测有限")
}

/// 构造一条装载基准的统一记录（算子级）。
///
/// 装载是算子级 IO 路径，记 [`EntryKind::Op`]；整机峰值带宽 / 算力未在本机测得，上界留
/// [`None`]，不拿标称值顶替。字节吞吐是 Report 侧由字节账目与实测 ns 派生的分析件，
/// 不塞进记录。
pub(crate) fn load_record(
    host: &HostProfile,
    scenario: &LoadScenario,
    observed: f64,
) -> BenchRecord {
    let geometry = Geometry::new(scenario.shape.to_vec(), scenario.batch, scenario.dtype)
        .expect("装载几何合法：形状 / 批次 / 精度通过构造自检");
    let record = BenchRecord::new(
        EntryKind::Op,
        scenario.scenario,
        scenario.backend,
        geometry,
        harness_of(host, scenario.backend),
        RecordRoofline::default(),
    )
    .expect("装载记录合法：身份 / 几何 / 后端通过构造自检");
    record.with_observed(observed).expect("实测有限")
}

/// 构造一条整模型一步基准的统一记录（模型级）。
///
/// 整模型一步记 [`EntryKind::Model`]；整机峰值 BW / GFLOPS 未测得，上界留 [`None`]。
/// tokens/s 与页数是 Report 侧账目 / 派生件，不塞进记录。
pub(crate) fn model_record(
    host: &HostProfile,
    scenario: &DecodeScenario,
    observed: f64,
) -> BenchRecord {
    let geometry = Geometry::new(vec![MODEL_DIM, MODEL_OUT], scenario.batch, "f32")
        .expect("模型几何合法：形状 / 批次 / 精度通过构造自检");
    let record = BenchRecord::new(
        EntryKind::Model,
        scenario.scenario,
        "cpu-reference",
        geometry,
        harness_of(host, "cpu-reference"),
        RecordRoofline::default(),
    )
    .expect("模型记录合法：身份 / 几何 / 后端通过构造自检");
    record.with_observed(observed).expect("实测有限")
}

/// 一条记录对拍所期望的身份 / 几何 / 实测；主机档案取自 [`HostProfile`]。
pub(crate) struct RecordExpectation<'a> {
    pub(crate) name: &'a str,
    pub(crate) backend: &'a str,
    pub(crate) shape: &'a [usize],
    pub(crate) batch: u32,
    pub(crate) dtype: &'a str,
    pub(crate) observed: f64,
}

impl<'a> RecordExpectation<'a> {
    pub(crate) fn new(
        name: &'a str,
        backend: &'a str,
        shape: &'a [usize],
        batch: u32,
        dtype: &'a str,
        observed: f64,
    ) -> Self {
        Self {
            name,
            backend,
            shape,
            batch,
            dtype,
            observed,
        }
    }
}

/// 逐字段核对一条记录的身份 / 几何 / 主机档案与实测；屋顶上界按未测得留 [`None`]。
///
/// 族别（Op / Model）由调用方单独断言，其余字段三条归档口径一致；driver 与后端同值。
pub(crate) fn assert_record_fields(
    record: &BenchRecord,
    host: &HostProfile,
    expect: &RecordExpectation<'_>,
) {
    assert_eq!(record.name(), expect.name);
    assert_eq!(record.backend(), expect.backend);
    assert_eq!(record.geometry().shape(), expect.shape);
    assert_eq!(record.geometry().batch(), expect.batch);
    assert_eq!(record.geometry().dtype(), expect.dtype);
    assert_eq!(record.harness().host(), host.host);
    assert_eq!(record.harness().cpu(), host.cpu);
    assert_eq!(record.harness().os(), host.os);
    assert_eq!(record.harness().toolchain(), host.toolchain);
    assert_eq!(record.harness().driver(), expect.backend);
    assert!(record.roofline().peak_bw_gb_s().is_none());
    assert!(record.roofline().peak_gflops().is_none());
    assert_eq!(record.observed(), Some(expect.observed));
}

/// 核对一条算子级归档记录：族别 Op，身份 / 几何取自台账描述符，主机档案逐字段对上。
pub(crate) fn assert_op_record(
    host: &HostProfile,
    record: &BenchRecord,
    fields: (&str, &str, &[usize], u32, &str),
    observed: f64,
) {
    let (name, backend, shape, batch, dtype) = fields;
    assert_eq!(record.kind(), EntryKind::Op);
    assert_record_fields(
        record,
        host,
        &RecordExpectation::new(name, backend, shape, batch, dtype, observed),
    );
}

/// 断言归档 JSON 含全部读者必需字段；`what` 是报告类别前缀，进失败文案。
pub(crate) fn assert_present(rendered: &str, fields: &[&str], what: &str) {
    for field in fields {
        assert!(
            rendered.contains(field),
            "{what}报告缺少 {field}：读者无法自己判断结论"
        );
    }
}

/// 读回归档 JSON：必须能被仓库唯一解析器读懂（整数约定见模块文档）。
pub(crate) fn parsed_report(rendered: &str) -> weights::json::Value {
    weights::json::parse(rendered).expect("合法 JSON")
}

/// 取归档 JSON 里的批大小整数；键缺失或非整数返回 `None`。
pub(crate) fn parsed_batch(parsed: &weights::json::Value) -> Option<i64> {
    parsed.get("batch").and_then(|value| value.integer().ok())
}
