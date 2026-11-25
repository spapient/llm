//! 归档渲染与对拍：记录 → JSON、结论、上界 / 负载 / 解码报告。

use super::*;

/// 一份归档报告：统一记录（身份 / 几何 / 后端 / 上界 / 实测）+ 本地分析件 + 结论。
pub(crate) struct Report<'a> {
    pub(crate) scenario: &'a Scenario,
    pub(crate) record: BenchRecord,
    pub(crate) roofline: Roofline,
    pub(crate) verdict: &'static str,
    pub(crate) baseline_ns: f64,
    pub(crate) budget_percent: f64,
}

/// 把形状渲染成 JSON 整数数组：`[4096, 4096]` / `[4096]`。
pub(crate) fn shape_json(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(std::string::ToString::to_string).collect();
    format!("[{}]", parts.join(", "))
}

/// 归档 JSON 字段的标量取值：字符串加引号、整数按整数约定直出，`Raw` 是预渲染的 JSON 片段。
#[derive(Clone)]
pub(crate) enum Field {
    Str(String),
    Int(i64),
    Raw(String),
}

impl Field {
    /// 渲染成 JSON 右值（不含键名）。
    pub(crate) fn render(&self) -> String {
        match self {
            Field::Str(value) => format!("\"{value}\""),
            Field::Int(value) => value.to_string(),
            Field::Raw(value) => value.clone(),
        }
    }
}

/// 一份归档报告从记录侧取齐的公共取值：身份 / 几何 / 主机档案与实测。
///
/// 装载与模型两组报告共用这一层；各自独有的账目字段（字节吞吐 / tokens、页数）留在各自
/// 报告里，不进记录也不进这里。
pub(crate) struct ReportRecord<'a> {
    pub(crate) name: &'a str,
    pub(crate) kind: &'a str,
    pub(crate) backend: &'a str,
    pub(crate) shape: String,
    pub(crate) batch: u32,
    pub(crate) dtype: &'a str,
    pub(crate) observed: f64,
    pub(crate) host: &'a str,
    pub(crate) cpu: &'a str,
    pub(crate) os: &'a str,
    pub(crate) toolchain: &'a str,
    pub(crate) arch: &'a str,
}

impl<'a> ReportRecord<'a> {
    /// 从回填过实测的记录上一次取齐公共字段。
    pub(crate) fn from(record: &'a BenchRecord) -> Self {
        let geometry = record.geometry();
        let harness = record.harness();
        Self {
            name: record.name(),
            kind: record.kind().as_str(),
            backend: record.backend(),
            shape: shape_json(geometry.shape()),
            batch: geometry.batch(),
            dtype: geometry.dtype(),
            observed: record.observed().expect("记录已回填实测"),
            host: harness.host(),
            cpu: harness.cpu(),
            os: harness.os(),
            toolchain: harness.toolchain(),
            arch: harness.arch(),
        }
    }
}

/// 渲染一份无基线归档 JSON：公共身份 / 几何 / 主机字段 + 两组各自的账目字段。
///
/// `front` 插在 `dtype` 之后（每迭代账目），`tail` 插在 `observed_ns` 之后（派生速率）；
/// 键序沿用既有报告书写顺序，解析器按键名读回，不依赖顺序。
pub(crate) fn report_json(
    group: &str,
    benchmark: &str,
    meta: &ReportRecord<'_>,
    front: &[(&str, Field)],
    tail: &[(&str, Field)],
) -> String {
    let mut fields: Vec<(&str, Field)> = vec![
        ("surface", Field::Str(group.to_string())),
        ("benchmark", Field::Str(benchmark.to_string())),
        ("scenario", Field::Str(meta.name.to_string())),
        ("kind", Field::Str(meta.kind.to_string())),
        ("backend", Field::Str(meta.backend.to_string())),
        ("shape", Field::Raw(meta.shape.clone())),
        ("batch", Field::Int(i64::from(meta.batch))),
        ("dtype", Field::Str(meta.dtype.to_string())),
    ];
    for (key, value) in front {
        fields.push((key, value.clone()));
    }
    fields.push(("metric", Field::Str("ns_per_iter".to_string())));
    fields.push(("observed_ns", Field::Int(meta.observed.round() as i64)));
    for (key, value) in tail {
        fields.push((key, value.clone()));
    }
    fields.push(("host", Field::Str(meta.host.to_string())));
    fields.push(("cpu", Field::Str(meta.cpu.to_string())));
    fields.push(("os", Field::Str(meta.os.to_string())));
    fields.push(("toolchain", Field::Str(meta.toolchain.to_string())));
    fields.push(("algorithm", Field::Str(meta.arch.to_string())));
    fields.push(("profile", Field::Str("bench".to_string())));

    let inner = fields
        .iter()
        .map(|(key, value)| format!("  \"{key}\": {}", value.render()))
        .collect::<Vec<_>>()
        .join(",\n");
    format!("{{\n{inner}\n}}\n")
}

impl Report<'_> {
    /// 渲染成整数数值的 JSON（见模块文档最后一段）。
    pub(crate) fn render(&self) -> String {
        let scenario = self.scenario;
        let record = &self.record;
        let roofline = &self.roofline;
        let harness = record.harness();
        let geometry = record.geometry();
        let observed = record.observed().expect("记录已回填实测");
        format!(
            "{{\n  \"surface\": \"{}\",\n  \"benchmark\": \"{}\",\n  \
             \"scenario\": \"{}\",\n  \"op\": \"{}\",\n  \"backend\": \"{}\",\n  \
             \"shape\": {},\n  \"batch\": {},\n  \"dtype\": \"{}\",\n  \
             \"read_bytes\": {},\n  \"write_bytes\": {},\n  \"working_set_bytes\": {},\n  \
             \"cache_tier\": \"{}\",\n  \"flops\": {},\n  \
             \"metric\": \"ns_per_iter\",\n  \"direction\": \"{}\",\n             \"budget_percent\": {},\n             \"baseline_ns\": {},\n             \"observed_ns\": {},\n             \"verdict\": \"{}\",\n             \"host\": \"{}\",\n             \"cpu\": \"{}\",\n             \"os\": \"{}\",\n             \"toolchain\": \"{}\",\n             \"algorithm\": \"{}\",\n             \"read_ceiling_mbps\": {},\n             \"write_ceiling_mbps\": {},\n             \"roofline_ns\": {},\n             \"bottleneck_side\": \"{}\",\n             \"achieved_percent\": {},\n             \"declared_ns\": {},\n             \"ceiling_percent\": {},\n             \"s_plus_percent\": {},\n             \"s_plus_line\": {},\n             \"mflops_per_s\": {},\n             \"bottleneck\": \"{}\",\n             \
             \"profile\": \"bench\"\n}}\n",
            scenario.group,
            scenario.benchmark,
            record.name(),
            scenario.op,
            record.backend(),
            shape_json(geometry.shape()),
            geometry.batch(),
            geometry.dtype(),
            roofline.read,
            roofline.write,
            roofline.read + roofline.write,
            roofline.tier,
            roofline.flops.round() as i64,
            Direction::Lower.as_str(),
            self.budget_percent.round() as i64,
            self.baseline_ns.round() as i64,
            observed.round() as i64,
            self.verdict,
            harness.host(),
            harness.cpu(),
            harness.os(),
            harness.toolchain(),
            harness.arch(),
            roofline.read_ceiling_mbps.round() as i64,
            roofline.write_ceiling_mbps.round() as i64,
            roofline.ns.round() as i64,
            roofline.side,
            roofline.achieved_percent.round() as i64,
            roofline.declared_ns.round() as i64,
            roofline.ceiling_percent.round() as i64,
            roofline.s_plus_percent().round() as i64,
            roofline.s_plus_line().round() as i64,
            roofline.mflops_per_s.round() as i64,
            roofline.bottleneck(),
        )
    }
}

/// 判定单个场景：建 `Case`、比对实测、归档报告；回退则 panic。
pub(crate) fn judge_one(manifest: &Path, host: &HostProfile, scenario: &Scenario) -> Option<()> {
    let baseline = baseline_for(host.host, scenario.backend, scenario.scenario)?;
    let observed = observed_ns(&estimates(manifest, scenario.group, scenario.benchmark))?;

    let case = Case::new(
        scenario.scenario,
        baseline.ns,
        baseline.budget_percent,
        Direction::Lower,
    )
    .expect("基线合法");
    let judgement = case.judge(observed);

    let dir = manifest
        .join("../..")
        .join("target/bench-report")
        .join(host.host);
    std::fs::create_dir_all(&dir).expect("归档目录可建");
    let report = Report {
        scenario,
        record: record_for(host, scenario, observed),
        roofline: Roofline::of(host, scenario, baseline.ns, observed),
        verdict: judgement.as_str(),
        baseline_ns: baseline.ns,
        budget_percent: baseline.budget_percent,
    };
    let rendered = report.render();
    // 自检：归档必须是能被仓库唯一 JSON 解析器读懂的文档（整数约定见模块文档）。
    weights::json::parse(&rendered).expect("归档必须是合法 JSON");
    std::fs::write(dir.join(format!("{}.json", scenario.group)), &rendered).expect("归档可写");

    if !report.roofline.meets_s_plus() {
        panic!(
            "{} 没到 S++ 线：{} 侧达成率 {:.0}% < {:.0}%（实测 {observed:.0} ns，\
             声明上界 {:.0} ns，带宽上界 {:.0} ns）。\
             慢在内核就优化内核；慢在上界就说明上界登记过期，按新测量重登",
            scenario.scenario,
            report.roofline.side,
            report.roofline.s_plus_percent(),
            report.roofline.s_plus_line(),
            baseline.ns,
            report.roofline.ns,
        );
    }

    if judgement.is_pass() {
        return Some(());
    }
    panic!(
        "{} 回退：实测 {observed:.0} ns，基线 {:.0} ns，预算 {:.0}%（上限 {:.0} ns）。\
         先确认不是这台机器在忙；同一口径下确实回退了，要么修，要么按新测量重新登记基线",
        scenario.scenario,
        baseline.ns,
        baseline.budget_percent,
        case.limit()
    );
}

/// 一份上界归档：统一记录（含实测带宽上界）+ 结论。
pub(crate) struct CeilingReport<'a> {
    pub(crate) ceiling: &'a Ceiling,
    pub(crate) record: BenchRecord,
    pub(crate) verdict: &'static str,
}

impl CeilingReport<'_> {
    /// 渲染成整数数值的 JSON。
    pub(crate) fn render(&self) -> String {
        let ceiling = self.ceiling;
        let harness = self.record.harness();
        let observed = self.record.observed().expect("记录已回填实测");
        format!(
            "{{\n  \"surface\": \"{}\",\n  \"benchmark\": \"{}\",\n  \
             \"scenario\": \"mem_stream/{}_{}\",\n  \"kind\": \"ceiling\",\n  \
             \"tier\": \"{}\",\n  \"traffic\": \"{}\",\n             \"bytes_per_iter\": {},\n  \
             \"metric\": \"ns_per_iter\",\n  \"direction\": \"{}\",\n             \"budget_percent\": {},\n             \"baseline_ns\": {},\n             \"observed_ns\": {},\n             \"verdict\": \"{}\",\n             \"host\": \"{}\",\n             \"cpu\": \"{}\",\n             \"os\": \"{}\",\n             \"toolchain\": \"{}\",\n             \"mb_per_s\": {},\n             \"profile\": \"bench\"\n}}\n",
            ceiling.group,
            ceiling.benchmark,
            ceiling.tier,
            ceiling.traffic,
            ceiling.tier,
            ceiling.traffic,
            ceiling.bytes,
            Direction::Lower.as_str(),
            ceiling.budget_percent.round() as i64,
            ceiling.ns.round() as i64,
            observed.round() as i64,
            self.verdict,
            harness.host(),
            harness.cpu(),
            harness.os(),
            harness.toolchain(),
            mbps(ceiling).round() as i64,
        )
    }
}

/// 判定单个上界：搬运速率掉了，说明机器或工具链变了，roofline 的分母必须重新登记。
pub(crate) fn judge_ceiling(manifest: &Path, host: &HostProfile, ceiling: &Ceiling) -> Option<()> {
    let observed = observed_ns(&estimates(manifest, ceiling.group, ceiling.benchmark))?;
    let case = Case::new(
        ceiling.benchmark,
        ceiling.ns,
        ceiling.budget_percent,
        Direction::Lower,
    )
    .expect("上界合法");
    let judgement = case.judge(observed);

    let dir = manifest
        .join("../..")
        .join("target/bench-report")
        .join(host.host);
    std::fs::create_dir_all(&dir).expect("归档目录可建");
    let rendered = CeilingReport {
        ceiling,
        record: ceiling_record(host, ceiling, observed),
        verdict: judgement.as_str(),
    }
    .render();
    weights::json::parse(&rendered).expect("归档必须是合法 JSON");
    let path = dir.join(format!(
        "{}_{}_{}.json",
        ceiling.group, ceiling.tier, ceiling.traffic
    ));
    std::fs::write(&path, &rendered).expect("归档可写");

    if judgement.is_pass() {
        return Some(());
    }
    panic!(
        "上界 {} 回退：实测 {observed:.0} ns，基线 {:.0} ns（上限 {:.0} ns）。\
         分母变了，所有达成率都要按新测量重算并重新登记",
        ceiling.benchmark,
        ceiling.ns,
        case.limit()
    );
}

/// 一份装载归档：统一记录 + 字节账目；无基线、无判定结论（跨机绝对值不可比）。
pub(crate) struct LoadReport<'a> {
    pub(crate) scenario: &'a LoadScenario,
    pub(crate) record: BenchRecord,
}

impl LoadReport<'_> {
    /// 渲染成整数数值的 JSON（整数约定见 [`Report`]）。
    pub(crate) fn render(&self) -> String {
        let scenario = self.scenario;
        let meta = ReportRecord::from(&self.record);
        let bytes = scenario.bytes;
        report_json(
            scenario.group,
            scenario.benchmark,
            &meta,
            &[("bytes_per_iter", Field::Int(bytes as i64))],
            &[(
                "gb_per_s",
                Field::Int(gb_per_s(bytes, meta.observed).round() as i64),
            )],
        )
    }
}

/// 一份整模型一步归档：统一记录 + token 与页数账目；无基线、无判定结论。
pub(crate) struct DecodeReport<'a> {
    pub(crate) scenario: &'a DecodeScenario,
    pub(crate) record: BenchRecord,
}

impl DecodeReport<'_> {
    /// 渲染成整数数值的 JSON（整数约定见 [`Report`]）。
    pub(crate) fn render(&self) -> String {
        let scenario = self.scenario;
        let meta = ReportRecord::from(&self.record);
        let tokens = scenario.tokens;
        let pages = scenario.pages;
        report_json(
            scenario.group,
            scenario.benchmark,
            &meta,
            &[
                ("tokens_per_iter", Field::Int(tokens as i64)),
                ("pages_in_use", Field::Int(pages as i64)),
            ],
            &[(
                "tokens_per_s",
                Field::Int(tokens_per_s(tokens, meta.observed).round() as i64),
            )],
        )
    }
}

/// 归档一条无基线基准：读本机实测，建记录，写归档 JSON；没有测量则跳过。
///
/// `make` 用实测 ns 构造记录，`render` 把记录渲染成归档报告，`file` 是归档文件名；
/// 装载与模型两组只在记录构造、独有账目字段与文件名上不同。
pub(crate) fn archive_record(
    manifest: &Path,
    host: &HostProfile,
    group: &str,
    benchmark: &str,
    file: String,
    make: impl FnOnce(&HostProfile, f64) -> BenchRecord,
    render: impl FnOnce(BenchRecord) -> String,
) -> Option<()> {
    let observed = observed_ns(&estimates(manifest, group, benchmark))?;
    let record = make(host, observed);
    let dir = manifest
        .join("../..")
        .join("target/bench-report")
        .join(host.host);
    std::fs::create_dir_all(&dir).expect("归档目录可建");
    let rendered = render(record);
    weights::json::parse(&rendered).expect("归档必须是合法 JSON");
    std::fs::write(dir.join(file), rendered).expect("归档可写");
    Some(())
}

/// 归档一条装载基准：读本机实测，建记录，写归档 JSON；没有测量则跳过。
pub(crate) fn archive_load(
    manifest: &Path,
    host: &HostProfile,
    scenario: &LoadScenario,
) -> Option<()> {
    archive_record(
        manifest,
        host,
        scenario.group,
        scenario.benchmark,
        format!("weights_load_{}.json", scenario.benchmark),
        |host, observed| load_record(host, scenario, observed),
        |record| LoadReport { scenario, record }.render(),
    )
}

/// 归档一条整模型一步基准：读本机实测，建记录，写归档 JSON；没有测量则跳过。
pub(crate) fn archive_model(
    manifest: &Path,
    host: &HostProfile,
    scenario: &DecodeScenario,
) -> Option<()> {
    archive_record(
        manifest,
        host,
        scenario.group,
        scenario.benchmark,
        format!("model_decode_{}.json", scenario.benchmark),
        |host, observed| model_record(host, scenario, observed),
        |record| DecodeReport { scenario, record }.render(),
    )
}

/// 本机档案；有基线却没有档案时直接失败，而不是拿一个没有机型的 host 字符串判。
pub(crate) fn local_profile(host: &str) -> Option<&'static HostProfile> {
    let profile = profile_for(host);
    if profile.is_none() && BASELINES.iter().any(|b| b.host == host) {
        panic!(
            "主机 `{host}` 有登记基线却没有档案：补上机型、系统与工具链（见 HOSTS），\
             否则归档里的数字无法被复核"
        );
    }
    profile
}
