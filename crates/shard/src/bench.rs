//! `bench`：量一次，并把它折成"达成率"这个可判定的数（README §12.3）。

use crate::build;
use crate::config::config;
use crate::pipeline::{self, Run, model_for};
use crate::runner;
use crate::toolchain;
use crate::witness::{self, Evidence};
use cutile::builtin::{AUTO_DEVICE, Catalogue};
use cutile::dirs::Dirs;
use cutile::prof::{Emit, Profile};
use diagnostic::{Code, Error};
use prof::decl::OpDecl;
use prof::model::{Cost, Sample, Values};
use std::path::PathBuf;

/// 一次 `bench` 的实测口径（README §12.3 的台账）。
#[derive(Debug, Clone, PartialEq)]
pub struct Bench {
    /// 算子名。
    pub op: String,
    /// 被测设备身份（`vendor:id`）：多设备机器上"测得多少"必须绑在设备上，否则台账无法归属。
    pub device: String,
    /// 形状 `(m, n)`。
    pub shape: (u64, u64),
    /// 重复次数（= 计时样本数；warmup 不计入）。
    pub repeat: usize,
    /// ③ select 的产物（预测值就在这里）。
    pub plan: cutile::plan::Plan,
    /// 每次调用的耗时（纳秒）：全部样本的 min（最好情况）。
    pub per_run_ns: f64,
    /// 每次调用的耗时中位数（纳秒）：达成率按它计算，对单次抖动与频率状态更稳。
    pub p50_ns: f64,
    /// 每次调用的 95 分位耗时（纳秒）：尾部离群上界，与 p50 的差距即抖动幅度；不参与达成率判定。
    pub p95_ns: f64,
    /// 实测吞吐。
    pub measured: Throughput,
    /// 档案声明的带宽峰值与来源。
    pub peak: Option<Peak>,
    /// 输入工作集（字节）：权重 + scales + 激活，不含输出。达成率离开它没有意义——装得进缓存时
    /// 测到的是缓存带宽，不是档案里的主存峰值。
    pub working_set: u64,
    /// 档案声明的 FAST 容量（字节，0 表示未声明）。
    pub fast_bytes: u64,
    /// 缓存目录。
    pub cache_dir: Option<PathBuf>,
    /// 这次跑的那份编译产物：`reused` 为真表示没调用编译器。
    pub artifact: Option<cutile::cache::Artifact>,
    /// 本次运行使用的编译器身份；设备基准不缓存，仍需报告实际工具链。
    pub toolchain: Option<String>,
    /// 设备/host 输出与标量参考的最大相对误差。
    pub max_rel_err: f64,
    /// 这次跑通的见证条数：计时之前必须已有证据，否则测的是一条未经确认的能力。
    pub proven: usize,
    /// 这次编译用的参数（档案的 `[compile].flags`）：计时与产物都属于这一组。
    pub flags: String,
}

impl Bench {
    /// §16 的性能判据：达成率与它的尺子一起给（口径无效时 [`Performance::is_measured`] 为假）。
    pub fn performance(&self) -> Performance {
        Performance::new(self.measured.bandwidth_ratio, self.in_fast())
    }

    /// 工作集是否落在档案声明的 FAST 容量内（重复执行会命中缓存）。
    pub fn in_fast(&self) -> bool {
        self.fast_bytes > 0 && self.working_set <= self.fast_bytes
    }

    /// 用实测 p50 回填这台设备的有效带宽/算力系数（B/s 与 FLOP/s）；耗时为 0 时没有可回填的系数。
    pub fn coefficients(&self) -> Option<CostCoefficients> {
        cost_coefficients(self.plan.bytes, self.plan.flops, self.p50_ns)
    }

    /// 用回填系数对一份代价做 roofline 预测（秒）：`max(flops/有效算力, bytes/有效带宽)`。
    /// 分母是这台设备实测出来的，不是档案峰值——这是"校准"与"预估"的差别。
    pub fn predict_calibrated(&self, cost: &Cost) -> Option<f64> {
        let coefficients = self.coefficients()?;
        Some((cost.flops / coefficients.flops).max(cost.bytes() as f64 / coefficients.bw))
    }
}

/// `bench` 的口径结论（README §12.1/§12.3）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// 档案没有带宽峰值：无法判定。
    Unknown,
    /// 实测带宽超过档案峰值：数据驻留在缓存中，这个达成率不是同一把尺子。
    OverPeak,
    /// 工作集落在 FAST 容量内：测到的是缓存带宽，与档案的主存峰值不可比。
    CacheResident,
    /// 达到 L3 门槛（≥ 70%）。
    Reached,
    /// 低于 L3 门槛。
    Below,
}

/// 判据：这个达成率能不能比 L3 门槛、不能的话为什么；不回答"实现好在哪"——`> 1` 不是"更好"。
fn classify(ratio: Option<f64>, cache_resident: bool) -> Verdict {
    let Some(ratio) = ratio else {
        return Verdict::Unknown;
    };
    if ratio > 1.0 {
        return Verdict::OverPeak;
    }
    if cache_resident {
        return Verdict::CacheResident;
    }
    attainment(ratio)
}

/// 判据：口径有效的达成率过没过 70%；不回答"差的那部分该改哪行代码"。
fn attainment(ratio: f64) -> Verdict {
    if ratio >= config().l3_threshold {
        Verdict::Reached
    } else {
        Verdict::Below
    }
}

/// §16 的性能判据：一个达成率连同它的尺子一起给。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Performance {
    /// 实测 / 档案峰值；档案没声明峰值或耗时为 0 时为 `None`。
    pub ratio: Option<f64>,
    /// 口径与判定（与 [`Bench::performance`] 共用同一个判定）。
    pub verdict: Verdict,
}

impl Performance {
    /// 由达成率与工作集判定。
    pub fn new(ratio: Option<f64>, cache_resident: bool) -> Self {
        Self {
            ratio,
            verdict: classify(ratio, cache_resident),
        }
    }

    /// 这个数是否参与判定：口径无效的达成率按 §12.3 视为未测量。
    pub const fn is_measured(&self) -> bool {
        matches!(self.verdict, Verdict::Reached | Verdict::Below)
    }

    /// §16「某设备 < 70%」的结论：这台设备承认需要特化代码。
    pub const fn needs_specialisation(&self) -> bool {
        matches!(self.verdict, Verdict::Below)
    }
}

/// 一次 bench 的实测口径。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Throughput {
    /// 带宽（GB/s）。
    pub gb_s: f64,
    /// 算力（GFLOP/s）。
    pub gflops: f64,
    /// 带宽达成率（0–1）；档案没有峰值或耗时为零时为 `None`。
    pub bandwidth_ratio: Option<f64>,
}

/// 档案里的一个峰值：取值与人可读的来源（README §12.1 要求每个数都带推导式来源）。
#[derive(Debug, Clone, PartialEq)]
pub struct Peak {
    /// 以 B/s 计的取值。
    pub value: f64,
    /// 来源。
    pub src: String,
}

/// 判据：这台设备的达成率台账（回答"是多少、分母凭什么"；不回答"实现该怎么改"）。
pub fn bench(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    shape: Option<&str>,
    device: Option<&str>,
    repeat: usize,
) -> diagnostic::Result<Bench> {
    let (op, (m, n)) = target(catalogue, op_name, shape)?;
    let (profile, evidence) = verified(catalogue, dirs, device)?;
    let run = Run::new(op, profile, evidence.proof(profile))?;
    let timed = run_sample(&run, dirs, (m as usize, n as usize), repeat.max(1))?;
    Ok(assemble(op_name, timed, profile, &evidence))
}

/// 判据：名字存在吗、形状是多少；不回答"这台设备能不能跑它"——那是 [`verified`] 与协商的事。
fn target<'a>(
    catalogue: &'a Catalogue,
    op_name: &str,
    shape: Option<&str>,
) -> diagnostic::Result<(&'a OpDecl, (u64, u64))> {
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| Error::new(Code::Missing, format!("没有算子 `{op_name}`")))?;
    let model = model_for(&op.name)?;
    Ok((op, pipeline::shape(shape, &model)?))
}

/// 判据：有没有一条刚由本机执行证明的能力；不回答"跑得多快"——计时在 [`run_sample`]。
fn verified<'a>(
    catalogue: &'a Catalogue,
    dirs: &Dirs,
    device: Option<&str>,
) -> diagnostic::Result<(&'a Profile, Evidence)> {
    // `--device` 与 `doctor` 同一套取值（`auto` 或缺省 = 按本机指纹挑），因为"测哪台设备"与
    // "审计哪台设备"必须是同一个选择器，否则台账里的名字会对不上 `doctor` 的首档。
    let profile = catalogue.resolve(device.unwrap_or(AUTO_DEVICE))?;
    require_bench_backend(profile)?;
    // 计时之前必须先有证据：否则测的是"档案说它能跑"，不是"这台机器能跑"。
    let evidence = witness::establish(profile, dirs);
    if let Some(error) = &evidence.failure {
        return Err(error.clone().with("bench 不能给一台没有执行证据的设备计时"));
    }
    if evidence.basis != witness::Basis::Executed {
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "`{}({})` 本机执行不了（emit 是 {}）：bench 没有可测的对象",
                profile.meta.vendor, profile.meta.id, profile.meta.emit
            ),
        ));
    }
    Ok((profile, evidence))
}

/// 当前 harness 只计时串行 C 语义路径；设备生产态计时走 `prof::runtime` FFI 后端岛。
fn require_bench_backend(profile: &Profile) -> diagnostic::Result<()> {
    if profile.meta.emit == Emit::C {
        return Ok(());
    }
    Err(Error::new(
        Code::Unsupported,
        format!(
            "`bench` 的 `{}` 设备生产态计时走 prof::runtime FFI 后端岛，shard 不再生成设备基准程序",
            profile.meta.emit
        ),
    ))
}

/// 判据：测了什么、测到多少（形状、样本数、耗时、工作集与产物）；不回答"离档案峰值差多少"。
struct Timed {
    shape: (u64, u64),
    repeat: usize,
    per_run_ns: f64,
    p50_ns: f64,
    p95_ns: f64,
    working_set: u64,
    plan: cutile::plan::Plan,
    artifact: Option<cutile::cache::Artifact>,
    toolchain: Option<String>,
    max_rel_err: f64,
    cache_dir: Option<PathBuf>,
}

/// 判据：跑了几次、每次多久、工作集多大；不回答"这个数是不是缓存带宽"——工作集随样本一起交回。
fn run_sample(
    run: &Run<'_>,
    dirs: &Dirs,
    (m, n): (usize, usize),
    repeat: usize,
) -> diagnostic::Result<Timed> {
    let (prepared, sample) = prepare_sample(run, dirs, (m, n))?;
    let measured = measure(run, dirs, &prepared, &sample, repeat)?;
    let (per_run_ns, p50_ns, p95_ns, max_rel_err) =
        readings(run, &sample, repeat, &measured.outcome)?;
    Ok(Timed {
        shape: (m as u64, n as u64),
        repeat,
        per_run_ns,
        p50_ns,
        p95_ns,
        working_set: working_set(&sample, run.op.signature.output().unwrap_or_default()),
        cache_dir: measured.cache_dir,
        plan: prepared.plan,
        artifact: measured.artifact,
        toolchain: measured.compiler,
        max_rel_err,
    })
}

/// 装配一段：先确定变体与源文本，再取样本；拼装期签名不符在编译之前就被拒。
fn prepare_sample(
    run: &Run<'_>,
    dirs: &Dirs,
    (m, n): (usize, usize),
) -> diagnostic::Result<(pipeline::Prepared, Sample)> {
    let prepared = prepare(run, dirs, m, n)?;
    let sample = run.sample(m, n);
    run.op.signature.check_sample(&sample)?;
    Ok((prepared, sample))
}

/// 一次测量的读数：计时三档与两处校验（样本数、精度）。任一不成立就不产温度数。
fn readings(
    run: &Run<'_>,
    sample: &Sample,
    repeat: usize,
    outcome: &runner::Outcome,
) -> diagnostic::Result<(f64, f64, f64, f64)> {
    let (per_run_ns, p50_ns, p95_ns) = timings(outcome)?;
    check_samples(repeat, outcome)?;
    let max_rel_err = check_accuracy(run, sample, outcome)?;
    Ok((per_run_ns, p50_ns, p95_ns, max_rel_err))
}

/// 一次测量走哪条通道只由 emit 决定：宿主 C 走本机编译，其余走注册过的设备后端。
fn measure(
    run: &Run<'_>,
    dirs: &Dirs,
    prepared: &pipeline::Prepared,
    sample: &Sample,
    repeat: usize,
) -> diagnostic::Result<Measured> {
    if run.profile.meta.emit == Emit::C {
        host_measure(run, dirs, prepared, sample, repeat)
    } else {
        Err(Error::new(
            Code::Unsupported,
            format!(
                "设备 `{}` 的生产态计时走 prof::runtime FFI 后端岛，shard 不再生成设备基准程序",
                run.profile.meta.id
            ),
        ))
    }
}

/// 一次测量的原始产物：计时读数、可选缓存产物、其目录与编译器标识。
struct Measured {
    outcome: runner::Outcome,
    artifact: Option<cutile::cache::Artifact>,
    cache_dir: Option<PathBuf>,
    compiler: Option<String>,
}

/// 主机 C 基准：重复次数也是产物身份的一部分（换 `--repeat` 就换一份产物），产物可复用缓存。
fn host_measure(
    run: &Run<'_>,
    dirs: &Dirs,
    prepared: &pipeline::Prepared,
    sample: &Sample,
    repeat: usize,
) -> diagnostic::Result<Measured> {
    let harness = prepared.harness(run.op, sample, repeat)?;
    let tool = toolchain::for_profile(run.profile);
    let (outcome, artifact) =
        build::execute(run.profile, tool.as_ref(), &prepared.key, &harness, dirs)?;
    Ok(Measured {
        compiler: Some(artifact.key.tool.clone()),
        cache_dir: build::dir(&prepared.key, dirs),
        outcome,
        artifact: Some(artifact),
    })
}

/// 样本数就是重复次数：少一条就不能拿 p50/p95 说事。
fn check_samples(repeat: usize, outcome: &runner::Outcome) -> diagnostic::Result<()> {
    let got = outcome.timing.samples_ns.len();
    if got != repeat {
        return Err(Error::new(
            Code::Internal,
            format!("计时样本数不匹配：期望 {repeat}，得到 {got}"),
        ));
    }
    Ok(())
}

/// 与标量参考逐项对拍：相对误差超过该 dtype 的容差即失败，返回这次的最大误差。
fn check_accuracy(
    run: &Run<'_>,
    sample: &Sample,
    outcome: &runner::Outcome,
) -> diagnostic::Result<f64> {
    let expected = (run.model.reference)(sample)?;
    let max_rel_err = cutile::reference::max_rel_err(&outcome.bits, &expected);
    let tolerance = cutile::reference::tolerance(run.model.dtype);
    if max_rel_err > tolerance {
        return Err(Error::new(
            Code::Internal,
            format!("基准结果与标量参考不一致：max_rel_err {max_rel_err:.3e} > {tolerance:.0e}"),
        ));
    }
    Ok(max_rel_err)
}

/// 判据：选了哪条变体、按什么形状编哪份源文本；不回答"它编不编得过"——编译在 [`run_sample`]。
fn prepare(
    run: &Run<'_>,
    dirs: &Dirs,
    m: usize,
    n: usize,
) -> diagnostic::Result<pipeline::Prepared> {
    let path = run
        .negotiate()
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))?;
    run.prepare(&path, dirs, m, n)
}

/// 判据：每调用耗时是多少；不回答"样本够不够多"——p50/p95 缺失时退回 min，不把缺项变成第二个失败条件。
fn timings(outcome: &runner::Outcome) -> diagnostic::Result<(f64, f64, f64)> {
    let summary = outcome
        .timing
        .summary()
        .ok_or_else(|| Error::new(Code::Internal, "内核没有报告可用的计时样本"))?;
    Ok((summary.min_ns, summary.p50_ns, summary.p95_ns))
}

/// 判据：达成率与它的口径各是什么；不回答"实现该怎么改"——它只摊开数，不给建议。
fn assemble(op_name: &str, timed: Timed, profile: &Profile, evidence: &Evidence) -> Bench {
    let peak = profile.peak.bw.as_ref().map(|peak| Peak {
        value: peak.value,
        src: peak.src.clone(),
    });
    Bench {
        op: op_name.to_string(),
        device: format!("{}:{}", profile.meta.vendor, profile.meta.id),
        shape: timed.shape,
        repeat: timed.repeat,
        measured: throughput(
            timed.plan.bytes,
            timed.plan.flops,
            timed.p50_ns,
            peak.as_ref().map(|peak| peak.value),
        ),
        plan: timed.plan,
        per_run_ns: timed.per_run_ns,
        p50_ns: timed.p50_ns,
        p95_ns: timed.p95_ns,
        peak,
        working_set: timed.working_set,
        fast_bytes: profile.mem.fast_bytes,
        cache_dir: timed.cache_dir,
        artifact: timed.artifact,
        toolchain: timed.toolchain,
        max_rel_err: timed.max_rel_err,
        proven: evidence.device,
        flags: profile.flags.join(" "),
    }
}

/// 判据：除输出外的全部缓冲占多少字节；不回答"它装不装得进缓存"——那要与 FAST 容量一起看。
fn working_set(sample: &Sample, output: &str) -> u64 {
    sample
        .bufs
        .iter()
        .filter(|buf| buf.name != output)
        .map(|buf| match &buf.values {
            Values::F32(values) => 4 * values.len() as u64,
            Values::U8(values) => values.len() as u64,
            Values::F16(values) => 2 * values.len() as u64,
            Values::Bf16(values) => 2 * values.len() as u64,
        })
        .sum()
}

/// 判据：每秒搬了多少字节 / 算了多少次浮点、占档案峰值的几成；不回答"这个口径是否可比"。
pub fn throughput(bytes: u64, flops: f64, per_run_ns: f64, peak_bw: Option<f64>) -> Throughput {
    if per_run_ns <= 0.0 {
        return Throughput {
            gb_s: 0.0,
            gflops: 0.0,
            bandwidth_ratio: None,
        };
    }
    let gb_s = bytes as f64 / per_run_ns;
    let bandwidth_ratio = peak_bw
        .filter(|peak| *peak > 0.0)
        .map(|peak| gb_s * 1e9 / peak);
    Throughput {
        gb_s,
        gflops: flops / per_run_ns,
        bandwidth_ratio,
    }
}

/// 校准系数：用实测 p50 反推出的有效带宽（B/s）与算力（FLOP/s），回填进 roofline 当分母，
/// 替代档案里的峰值（README §12.3 的「实跑回填代价系数」）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostCoefficients {
    /// 有效带宽（B/s）。
    pub bw: f64,
    /// 有效算力（FLOP/s）。
    pub flops: f64,
}

/// 判据：把一次实测的流量/算力与 p50 折成这台设备能真实达成的系数；耗时为 0 即无法回填。
pub fn cost_coefficients(bytes: u64, flops: f64, p50_ns: f64) -> Option<CostCoefficients> {
    if p50_ns <= 0.0 {
        return None;
    }
    Some(CostCoefficients {
        bw: bytes as f64 / p50_ns * 1e9,
        flops: flops / p50_ns * 1e9,
    })
}

#[cfg(test)]
#[path = "../tests/bench_tests.rs"]
mod tests;
