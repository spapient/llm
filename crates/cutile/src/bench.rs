//! `bench`：跑一次并报告达成率（DESIGN §12.3）。

use crate::backend;
use crate::build;
use crate::builtin::Catalogue;
use crate::pipeline::{self, Run, model_for};
use crate::witness;
use diagnostic::{Code, Error};
use llm_config::Settings;
use op::model::{Sample, Values};

/// L3 达成率门槛（DESIGN §12.3）：低于它就是"没有可用证据"。
pub const L3_THRESHOLD: f64 = 0.7;

/// 一次 `bench` 的实测口径（DESIGN §12.3 的台账）。
#[derive(Debug, Clone, PartialEq)]
pub struct Bench {
    /// 算子名。
    pub op: String,
    /// 形状 `(m, n)`。
    pub shape: (u64, u64),
    /// 重复次数（= 计时样本数；warmup 不计入）。
    pub repeat: usize,
    /// ③ select 的产物（预测值就在这里）。
    pub plan: crate::plan::Plan,
    /// 每次调用的耗时（纳秒）：全部样本的 min（最好情况）。
    pub per_run_ns: f64,
    /// 每次调用的耗时中位数（纳秒）：达成率按它计算，对单次抖动与频率状态更稳。
    pub p50_ns: f64,
    /// 实测吞吐。
    pub measured: Throughput,
    /// 档案声明的带宽峰值与来源。
    pub peak: Option<Peak>,
    /// 输入工作集（字节）：权重 + scales + 激活，不含输出。
    ///
    /// 达成率离开它就没有意义：工作集小于缓存时测到的是缓存带宽，不是档案里的主存峰值。
    pub working_set: u64,
    /// 档案声明的 FAST 容量（字节，0 表示未声明）。
    pub fast_bytes: u64,
    /// 缓存目录。
    pub cache_dir: Option<std::path::PathBuf>,
    /// 这次跑的那份编译产物：`reused` 为真表示没调用编译器。
    pub artifact: crate::cache::Artifact,
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
}

/// `bench` 的口径结论（DESIGN §12.1/§12.3）。
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

/// 由达成率与工作集判定口径。
///
/// `> 1` 不是"更好"，而是说明测量与档案峰值不是同一口径。
fn classify(ratio: Option<f64>, cache_resident: bool) -> Verdict {
    match ratio {
        None => Verdict::Unknown,
        Some(ratio) if ratio > 1.0 => Verdict::OverPeak,
        _ if cache_resident => Verdict::CacheResident,
        Some(ratio) if ratio >= L3_THRESHOLD => Verdict::Reached,
        Some(_) => Verdict::Below,
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

/// 档案里的一个峰值：取值与人可读的来源（DESIGN §12.1 要求每个数都带推导式来源）。
#[derive(Debug, Clone, PartialEq)]
pub struct Peak {
    /// 以 B/s 计的取值。
    pub value: f64,
    /// 来源。
    pub src: String,
}

/// `shape` 是命令行覆盖；`None` 时用算子的生产形状（`Model::shape`），审计不持有一个缺省方阵。
///
/// # Errors
///
/// 形状非法、协商失败、缺编译器、编译/执行失败或内核没有给出计时样本时返回说明。
pub fn bench(
    catalogue: &Catalogue,
    settings: &Settings,
    op_name: &str,
    shape: Option<&str>,
    repeat: usize,
) -> diagnostic::Result<Bench> {
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| Error::new(Code::Missing, format!("没有算子 `{op_name}`")))?;
    let model = model_for(&op.name)?;
    let (m, n) = pipeline::shape(shape, &model)?;
    let profile = catalogue.resolve("auto")?;
    // 计时之前必须先有证据：否则测的是一条“档案说它能跑”的能力，而不是这台机器能跑的能力。
    let evidence = witness::establish(profile, settings);
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
    let run = Run::new(op, profile, evidence.proof(profile))?;
    let path = run
        .negotiate()
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))?;
    let (m, n) = (m as usize, n as usize);
    let prepared = run.prepare(&path, settings, m, n)?;
    let sample = (run.model.sample)(m, n);

    // 命中产物时连编译器都不需要，所以工具链只是“能重新编译”的前提，不是跑 bench 的前提。
    let tool = backend::detect();
    let repeat = repeat.max(1);
    // 重复次数在主函数里，所以它也是产物身份的一部分：换 `--repeat` 就换一份产物。
    let run_harness = prepared.harness(op, &sample, repeat)?;
    let (outcome, artifact) = build::execute(
        profile,
        tool.as_ref(),
        &prepared.key,
        &run_harness,
        settings,
    )?;
    let per_run_ns = outcome
        .timing
        .min_ns()
        .ok_or_else(|| Error::new(Code::Internal, "内核没有报告可用的计时样本"))?;
    // 中位数只用于报告抖动；样本缺失时退回 min，不让"没有 p50"变成第二个失败条件。
    let p50_ns = outcome.timing.p50_ns().unwrap_or(per_run_ns);
    let peak = profile.peak.bw.as_ref().map(|peak| Peak {
        value: peak.value,
        src: peak.src.clone(),
    });
    let cache_dir = build::dir(&prepared.key, settings);
    let plan = prepared.plan;
    Ok(Bench {
        op: op_name.to_string(),
        shape: (m as u64, n as u64),
        repeat,
        measured: throughput(
            plan.bytes,
            plan.flops,
            p50_ns,
            peak.as_ref().map(|peak| peak.value),
        ),
        plan,
        per_run_ns,
        p50_ns,
        peak,
        working_set: working_set(&sample, op.signature.output().unwrap_or_default()),
        fast_bytes: profile.mem.fast_bytes,
        cache_dir,
        artifact,
        proven: evidence.device,
        flags: profile.flags.join(" "),
    })
}

/// 输入工作集：除输出外的全部缓冲（权重、scales、激活）。
fn working_set(sample: &Sample, output: &str) -> u64 {
    sample
        .buffers
        .iter()
        .filter(|buffer| buffer.name != output)
        .map(|buffer| match &buffer.values {
            Values::F32(values) => 4 * values.len() as u64,
            Values::U8(values) => values.len() as u64,
        })
        .sum()
}

/// `bytes / ns` 直接就是 GB/s、`flops / ns` 直接就是 GFLOP/s（两者都以 1e9 为单位），
/// 但达成率必须换算回 B/s 再除以档案峰值——直接用量纲不同的 GB/s 去除会差 1e9 倍。
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

#[cfg(test)]
mod tests {
    use super::*;
    use op::model::Buffer;

    #[test]
    fn an_attainment_above_one_is_an_invalid_measurement_not_a_pass() {
        // 数据驻留在缓存里时 `bytes/ns` 会超过档案的主存峰值：这不是"超过 100% 的实现质量"。
        assert_eq!(classify(Some(1.42), false), Verdict::OverPeak);
        // 装得进缓存时，分母（主存峰值）本来就不是这次测量的口径。
        assert_eq!(classify(Some(0.19), true), Verdict::CacheResident);
        assert_eq!(classify(Some(1.0), false), Verdict::Reached);
        assert_eq!(classify(Some(0.7), false), Verdict::Reached);
        assert_eq!(classify(Some(0.69), false), Verdict::Below);
        assert_eq!(classify(None, false), Verdict::Unknown);
    }

    #[test]
    fn bandwidth_attainment_is_a_byte_ratio_not_a_gigabyte_one() {
        // 0.7 MB / 1.467 ms ≈ 0.477 GB/s，占 68 GB/s 的 0.7%；除以 GB/s 会算出 0.0%。
        let measured = throughput(700_000, 2.1e6, 1.467e6, Some(68.0e9));
        assert!((measured.gb_s - 0.477).abs() < 0.01, "{measured:?}");
        assert!((measured.gflops - 1.43).abs() < 0.01, "{measured:?}");
        assert!((measured.bandwidth_ratio.expect("有峰值") - 0.0070).abs() < 1e-4);
        assert_eq!(throughput(1, 1.0, 0.0, Some(1.0)).bandwidth_ratio, None);
    }

    #[test]
    fn working_set_is_every_input_buffer_but_not_the_output() {
        let sample = Sample {
            m: 1,
            n: 2,
            scalars: Vec::new(),
            buffers: vec![
                Buffer {
                    name: "w",
                    values: Values::U8(vec![0; 8]),
                },
                Buffer {
                    name: "x",
                    values: Values::F32(vec![1.0; 4]),
                },
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; 64]),
                },
            ],
        };
        assert_eq!(working_set(&sample, "y"), 8 + 16);
    }
}
