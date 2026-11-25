//! 一条流水线：找算子 → 取模型 → 建立证据 → 协商 → 选 tile → emit。

use crate::build;
use cutile::cache::{Artifact, CacheKey};
use cutile::cap::{self, NoPath, Path, Proof};
use cutile::decl::OpDecl;
use cutile::dirs::Dirs;
use cutile::emit::Source;
use cutile::harness;
use cutile::model::{Model, Sample};
use cutile::plan::{self, Plan};
use cutile::prof::{Emit, Profile};
use diagnostic::{Code, Error};

/// 这次要跑的那条流水线：算子、语义模型、设备档案与执行证据。
#[derive(Debug)]
pub(crate) struct Run<'a> {
    /// 声明层的算子。
    pub op: &'a OpDecl,
    /// 算子的语义模型（代价、样本、标量参考）。
    pub model: Model,
    /// 设备档案。
    pub profile: &'a Profile,
    /// 这次执行得到的证据（协商只认它，不认档案里的 `grade`）。
    pub proof: Proof,
    /// 形状覆盖：`None` 时形状由模型的 `sample(m, n)` 给出；给了就一律按它，
    /// 让调用方能指定模型采样函数定不下来的维度（如 GEMM 的收缩轴 `K`）。
    pub shape: Option<Sample>,
}

impl<'a> Run<'a> {
    /// 从算子、档案与证据建立一条流水线。
    pub fn new(op: &'a OpDecl, profile: &'a Profile, proof: Proof) -> diagnostic::Result<Self> {
        Ok(Self {
            model: model_for(&op.name)?,
            op,
            profile,
            proof,
            shape: None,
        })
    }

    /// 这次执行用的样本：显式形状优先，否则由模型按 `(m, n)` 采样。
    ///
    /// 形状的一切判断（网格各维、形态前置、降源常量）都从这里取，避免同一形状在两处
    /// 各算一遍而漂移——`K` 尤其如此：它不是 `(m, n)` 的函数。
    pub fn sample(&self, m: usize, n: usize) -> Sample {
        match &self.shape {
            Some(sample) => sample.clone(),
            None => (self.model.sample)(m, n),
        }
    }

    /// 这次协商该用的证据。
    pub fn negotiate(&self) -> Result<Path, Box<NoPath>> {
        cap::negotiate_with(self.op, self.profile, &self.proof)
    }

    /// 把 `(m, n)` 变成可执行的一批：形状、代价、计划与源文本。
    pub fn prepare(
        &self,
        path: &Path,
        dirs: &Dirs,
        m: usize,
        n: usize,
    ) -> diagnostic::Result<Prepared> {
        let shape = (self.model.shape_ctx)(m, n)?;
        let cost = (self.model.cost)(&shape)?;
        let plan = plan::select_with_heuristics(self.op, self.profile, path, cost, &shape)?;
        let (source, key) = build::source(self.op, self.profile, path, &plan.tile, dirs)?;
        Ok(Prepared { plan, source, key })
    }

    /// 为一枚 autotune 种子构造真实产物；tile 直接进入现有缓存指纹。
    pub fn prepare_tile(
        &self,
        path: &Path,
        dirs: &Dirs,
        m: usize,
        n: usize,
        tile: &cutile::decl::TileCombo,
    ) -> diagnostic::Result<Prepared> {
        let shape = (self.model.shape_ctx)(m, n)?;
        let cost = (self.model.cost)(&shape)?;
        let plan = plan::select_tile(self.profile, path, cost, tile.clone())?;
        let (source, key) = build::source(self.op, self.profile, path, &plan.tile, dirs)?;
        Ok(Prepared { plan, source, key })
    }
}

/// 一个 `(m, n)` 的可执行产物：计划、源文本与它的缓存键。
#[derive(Debug)]
pub(crate) struct Prepared {
    /// ③ select 的产物。
    pub plan: Plan,
    /// ⑤ emit 的产物。
    pub source: Source,
    /// 源文本的身份（产物缓存键的一半）。
    pub key: CacheKey,
}

impl Prepared {
    /// 拼出可执行程序：内核源码 + 样本 + 签名派生的实参 + 重复次数。
    pub fn harness(
        &self,
        op: &OpDecl,
        sample: &Sample,
        repeat: usize,
    ) -> diagnostic::Result<String> {
        op.signature.check_sample(sample)?;
        harness::harness(&self.source.text, &op.signature, sample, repeat)
    }
}

/// 形状覆盖：命令行给的 `MxN`，或算子的生产形状。
pub(crate) fn shape(text: Option<&str>, model: &Model) -> diagnostic::Result<(u64, u64)> {
    if let Some(text) = text {
        parse_shape(text, u64::from(model.group.max(1)))
    } else {
        let (m, n) = model.shape;
        Ok((m as u64, n as u64))
    }
}

pub(crate) fn parse_shape(shape: &str, group: u64) -> diagnostic::Result<(u64, u64)> {
    let (m, n) = split(shape)?;
    let dims = (dim(m)?, dim(n)?);
    validate(shape, dims, group).map(|()| dims)
}

/// 拆成两段：缺 `x` 分隔符时把原始文本原样报回去。
fn split(shape: &str) -> diagnostic::Result<(&str, &str)> {
    shape
        .split_once('x')
        .ok_or_else(|| invalid(format!("`{shape}` 不是 `MxN` 形式")))
}

/// 单个维度：去掉两侧空白后必须是非负整数。
fn dim(text: &str) -> diagnostic::Result<u64> {
    text.trim()
        .parse()
        .map_err(|_| invalid(format!("`{text}` 不是整数")))
}

/// 形状约束：两个维度都为正，且 `M` 是组大小的整数倍。
fn validate(shape: &str, (m, n): (u64, u64), group: u64) -> diagnostic::Result<()> {
    if m > 0 && n > 0 && m.is_multiple_of(group) {
        return Ok(());
    }
    Err(invalid(format!(
        "形状 {shape} 非法：M 必须是 {group} 的正整数倍"
    )))
}

fn invalid(message: String) -> Error {
    Error::new(Code::Invalid, message)
}

/// 取算子的语义模型（代价、样本、标量参考）。声明层新增了算子却没有模型时，
/// `doctor`/`verify`/`bench` 都无法给出计划或对拍结果。
pub(crate) fn model_for(name: &str) -> diagnostic::Result<Model> {
    nn::registry::model(name).ok_or_else(|| {
        Error::new(
            Code::Missing,
            format!("算子 `{name}` 没有语义模型：`nn::registry` 的 `model()` 注册表里没有它"),
        )
    })
}

/// `verify` 的形状矩阵：最小合法、多行非方形、生产形状。
pub(crate) fn verify_shapes(model: &Model) -> Vec<(u64, u64)> {
    let group = u64::from(model.group.max(1));
    let (m, n) = model.shape;
    vec![(group, 1), (group * 2, 3), (m as u64, n as u64)]
}

/// 必须被拒绝的形状：`M` 不是组大小的整数倍。
pub(crate) fn rejected_shapes(group: u64) -> Vec<(u64, u64)> {
    if group <= 1 {
        return Vec::new();
    }
    vec![(1, group), (group + 1, 8)]
}

// ---- 真机执行：给定样本、设备分派与标量对拍 ----

use crate::witness;
use cutile::builtin::Catalogue;
use cutile::emit::Fragment;

/// 一次真机执行：调用方给的样本 → 编译 → 跑一次 → 与标量参考对拍。
#[derive(Debug, Clone, PartialEq)]
pub struct Execution {
    /// 设备名（档案里的 `id`）。
    pub device: String,
    /// 厂商。
    pub vendor: String,
    /// 协商选中的变体。
    pub variant: String,
    /// 选中的 tile。
    pub tile: String,
    /// 缓存键（源文本的身份）。
    pub key: String,
    /// 生成的片段行数。
    pub fragment: Fragment,
    /// 内核输出。
    pub output: Vec<f32>,
    /// 标量参考输出。
    pub expected: Vec<f32>,
    /// 最大相对误差。
    pub max_rel_err: f64,
    /// 对拍阈值（由算子声明的元素类型推出）。
    pub tolerance: f64,
}

/// 用一份**给定的**样本走完整条流水线并执行一次。
///
/// 与 [`crate::verify::verify`] 的区别都是刻意的：样本由调用方给（真实权重），因此不做形状矩阵、
/// 只跑这一份；执行前仍要求设备有见证证据——没有证据的后端算出来的数不是结果，只是现象；
/// 对拍不通过是错误（[`Code::Invalid`]），不是"环境问题"。
///
/// 回答什么：这份数据在这台机器上算出来的数对不对。不回答什么：这台机器一般能不能算这条算子
/// （那是形状矩阵的事），也不改档案里的等级。
///
/// # Errors
///
/// 算子没有声明或没有语义表、设备不可用、设备没有执行证据、协商失败、形状与算子粒度不符、
/// 签名与样本不一致、编译/执行失败，或对拍超出容差时返回说明。
pub fn execute(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    device: &str,
    sample: &Sample,
) -> diagnostic::Result<Execution> {
    execute_with_repeat(catalogue, dirs, op_name, device, sample, 1).map(|timed| timed.execution)
}

/// 一次对拍及这次生成的 harness 所产生的全部重复调用计时。
#[derive(Debug, Clone, PartialEq)]
pub struct TimedExecution {
    /// 数值结果与设备协商信息。
    pub execution: Execution,
    /// warmup 与校准之后的真实计时样本。
    pub timing: crate::runner::Timing,
}

/// 给定样本与重复次数，走同一条协商、emit、编译执行和对拍路径。
pub fn execute_with_repeat(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    device: &str,
    sample: &Sample,
    repeat: usize,
) -> diagnostic::Result<TimedExecution> {
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| Error::new(Code::Missing, format!("没有算子 `{op_name}`")))?;
    let profile = catalogue.resolve(device)?;
    let run = established(op, profile, dirs)?;
    one(&run, profile, dirs, sample, repeat.max(1))
}

/// 证据先于执行：C 走宿主缓存构建，CUDA 走已注册后端；其余语言在这里明确拒绝。
fn established<'a>(
    op: &'a OpDecl,
    profile: &'a Profile,
    dirs: &Dirs,
) -> diagnostic::Result<Run<'a>> {
    if !matches!(profile.meta.emit, Emit::C | Emit::Cuda) {
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "设备 `{}` 的输出语言是 `{}`：当前真机执行入口支持 C 与 CUDA",
                profile.meta.id, profile.meta.emit
            ),
        ));
    }
    let evidence = witness::establish(profile, dirs);
    let proof = evidence.proof(profile);
    if let Some(error) = evidence.failure {
        return Err(error.with("该设备缺少执行证据，真实数据不在未验证的后端上运行"));
    }
    Run::new(op, profile, proof)
}

/// 一个样本：协商、形状校验、编译执行一次、对拍、组装报告。
fn one(
    run: &Run<'_>,
    profile: &Profile,
    dirs: &Dirs,
    sample: &Sample,
    repeat: usize,
) -> diagnostic::Result<TimedExecution> {
    let path = run
        .negotiate()
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))?;
    check_sample(u64::from(run.model.group.max(1)), sample)?;
    let prepared = run.prepare(&path, dirs, sample.m, sample.n)?;
    run_once(run, profile, dirs, sample, path, &prepared, repeat)
}

/// 编译执行一次、与标量参考对拍，并组装报告（字段全部来自这次执行，不读档案自述）。
fn run_once(
    run: &Run<'_>,
    profile: &Profile,
    dirs: &Dirs,
    sample: &Sample,
    path: Path,
    prepared: &Prepared,
    repeat: usize,
) -> diagnostic::Result<TimedExecution> {
    let harness = prepared.harness(run.op, sample, repeat)?;
    let tool = crate::toolchain::detect();
    let (bits, timing, _) = run_backend(
        profile,
        dirs,
        prepared,
        &harness,
        &run.op.name,
        tool.as_ref(),
    )?;
    let tolerance = cutile::reference::tolerance(run.model.dtype);
    let (expected, max_rel_err) = model_parity(&run.model, sample, &bits, tolerance)?;
    let execution = Execution {
        device: profile.meta.id.clone(),
        vendor: profile.meta.vendor.clone(),
        variant: path.variant,
        tile: prepared.plan.tile.to_string(),
        key: prepared.key.hex(),
        fragment: Fragment::new(prepared.source.lines()),
        output: bits,
        expected,
        max_rel_err,
        tolerance,
    };
    Ok(TimedExecution { execution, timing })
}

pub(crate) type DeviceOutcome = (Vec<f32>, crate::runner::Timing, Option<Artifact>);

pub(crate) fn run_backend(
    profile: &Profile,
    dirs: &Dirs,
    prepared: &Prepared,
    harness: &str,
    label: &str,
    tool: Option<&crate::toolchain::Toolchain>,
) -> diagnostic::Result<DeviceOutcome> {
    match profile.meta.emit {
        Emit::C => build::execute(profile, tool, &prepared.key, harness, dirs)
            .map(|(outcome, artifact)| (outcome.bits, outcome.timing, Some(artifact))),
        Emit::Cuda => witness::run_cuda_outcome(profile, dirs, harness, label)
            .map(|outcome| (outcome.bits, outcome.timing, None)),
        emit => Err(Error::new(
            Code::Unsupported,
            format!("输出语言 `{emit}` 没有真机执行通道"),
        )),
    }
}

pub(crate) fn parity(
    bits: &[f32],
    want: &[f32],
    expect: usize,
    tolerance: f64,
    label: &str,
    code: Code,
) -> diagnostic::Result<f64> {
    if bits.len() != expect {
        return Err(Error::new(
            code,
            format!("{label}：内核输出 {} 个数，期望 {expect}", bits.len()),
        ));
    }
    let error = cutile::reference::max_rel_err(bits, want);
    if error > tolerance {
        return Err(Error::new(
            code,
            format!("{label}：max_rel_err {error:.3e} > {tolerance:.0e}"),
        ));
    }
    Ok(error)
}

/// 样本形状校验：m/n 非零，且 m 是形状粒度的整数倍。
fn check_sample(group: u64, sample: &Sample) -> diagnostic::Result<()> {
    if sample.m == 0 || sample.n == 0 || !(sample.m as u64).is_multiple_of(group) {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "样本形状 {}x{} 不自洽：m 必须是形状粒度 {group} 的正整数倍",
                sample.m, sample.n
            ),
        ));
    }
    Ok(())
}

/// 对拍：长度一致且最大相对误差不超阈值，返回实测误差。
fn check_parity(bits: &[f32], expected: &[f32], tolerance: f64) -> diagnostic::Result<f64> {
    parity(
        bits,
        expected,
        expected.len(),
        tolerance,
        "对拍",
        Code::Invalid,
    )
}

/// 模型参考的唯一后处理：取出 f32 值并与本次设备输出对拍。
fn model_parity(
    model: &Model,
    sample: &Sample,
    bits: &[f32],
    tolerance: f64,
) -> diagnostic::Result<(Vec<f32>, f64)> {
    let expected = (model.reference)(sample)?;
    let expected = expected.as_f32()?.to_vec();
    let max_rel_err = check_parity(bits, &expected, tolerance)?;
    Ok((expected, max_rel_err))
}

// ---- 设备语言门：用声明语言解析全部见证 ----

use crate::config::{LanguageFront, config};
use crate::runner::driver::{with_scratch, write_text};
use crate::toolchain::{Toolchain, clang, probe_declared};
use crate::witness::{Selected, declared_witnesses, device_available, source};
use cutile::atom::Witness;
use std::collections::BTreeSet;
use std::fmt;
use std::path::Path as StdPath;

/// 见证在**它自己声称的语言**里的合法性。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LanguageGate {
    /// 设备执行已经覆盖这份见证（`c` / `metal`）：门不必再开，能真跑的就不只解析。
    Covered,
    /// 通过：这门语言的解析器收下了这份翻译单元。
    Passed {
        /// 解析器的身份与参数（`clang@21.0.0 -x cuda …`）。
        front: String,
        /// 参与解析的见证条数。
        witnesses: usize,
    },
    /// 本机没有这门语言的解析器：门开不了，也不拿 host 自检冒充。
    Absent {
        /// 为什么开不了（缺哪个程序，或这门语言的前端还没有登记）。
        reason: String,
    },
    /// 见证被自己的语言拒绝：要改的是证据，不是机器。
    Rejected {
        /// 解析器的第一行报错。
        reason: String,
    },
}

impl fmt::Display for LanguageGate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Covered => formatter.write_str("设备执行已覆盖"),
            Self::Passed { front, witnesses } => {
                write!(formatter, "{front} 通过（{witnesses} 条）")
            }
            Self::Absent { reason } => write!(formatter, "不可用：{reason}"),
            Self::Rejected { reason } => write!(formatter, "不成立：{reason}"),
        }
    }
}

/// 有效检查参数：表内固定的语法检查参数 + 档案自己声明的语言版本开关。
fn args_for<'a>(gate: &'a LanguageFront, profile: &'a Profile) -> Vec<&'a str> {
    let mut arguments: Vec<&str> = gate.check.iter().map(String::as_str).collect();
    arguments.extend(
        profile
            .flags
            .iter()
            .map(String::as_str)
            .filter(|flag| flag.starts_with("-cl-std=") || flag.starts_with("-std=")),
    );
    arguments
}

/// 表行声明的解析器：`clang` 行指「本机的 clang 前端」（先问 `clang`，再退到名字里带 clang 的
/// `cc`）；其余按声明原样探测，不退回宿主编译器——拿 `cc` 去解析 AscendC 只会把「本机没有 CANN」
/// 报成一条 C 语法错误。
fn front(spec: &LanguageFront) -> Option<Toolchain> {
    let declared = spec.tool.join(" ");
    if declared == "clang" {
        return clang();
    }
    probe_declared(&declared)
}

/// 用一台设备**自己声称的语言**把它的全部已声明见证过一遍。
#[must_use]
pub fn language_gate(profile: &Profile, dirs: &Dirs) -> LanguageGate {
    if device_available(profile) {
        return LanguageGate::Covered;
    }
    let gates = &config().gates;
    let Some(entry) = gates.iter().find(|entry| entry.emit == profile.meta.emit) else {
        let known: Vec<String> = gates.iter().map(|entry| entry.emit.to_string()).collect();
        return LanguageGate::Absent {
            reason: format!(
                "`{}` 的语法前端没有登记（已登记 {}）",
                profile.meta.emit,
                known.join(" / ")
            ),
        };
    };
    match parse_all(entry, profile, dirs) {
        Ok(gate) => gate,
        // 解析器拒绝是证据的缺陷，不能与“本机没东西可用”归一类。
        Err(error) => LanguageGate::Rejected {
            reason: error.to_string(),
        },
    }
}

/// 语法门的执行器：表行决定程序与参数，这里只负责“逐份文件过一遍、失败带第一行报错回来”。
fn parse_all(
    spec: &LanguageFront,
    profile: &Profile,
    dirs: &Dirs,
) -> diagnostic::Result<LanguageGate> {
    let selected = declared_witnesses(profile)?;
    if selected.is_empty() {
        return Ok(LanguageGate::Absent {
            reason: "没有已声明的见证：门没有对象可开".to_string(),
        });
    }
    let Some(toolchain) = front(spec) else {
        return Ok(LanguageGate::Absent {
            reason: format!(
                "本机没有 `{}` 的语法前端：`{}` 不在本机",
                spec.emit,
                spec.tool.join(" ")
            ),
        });
    };
    let arguments = args_for(spec, profile);
    with_scratch(dirs, "gate", |dir| {
        write_and_parse(spec, &arguments, &selected, dirs, dir)?;
        Ok(LanguageGate::Passed {
            front: format!("{} {}", toolchain.id(), arguments.join(" ")),
            witnesses: selected.len(),
        })
    })
}

/// 把每份见证按“内嵌优先”的文本落到临时目录，再交给解析器：同一份源码可以没有磁盘副本。
fn write_and_parse(
    spec: &LanguageFront,
    arguments: &[&str],
    selected: &[Selected<'_>],
    dirs: &Dirs,
    dir: &StdPath,
) -> diagnostic::Result<()> {
    let mut written: BTreeSet<&str> = BTreeSet::new();
    for (_, _, witness) in selected {
        if written.insert(witness.file.as_str()) {
            write_one(spec, arguments, witness, dirs, dir)?;
        }
    }
    Ok(())
}

/// 一份见证：按“内嵌优先”的文本落盘，再交给它自己语言的解析器。
fn write_one(
    spec: &LanguageFront,
    arguments: &[&str],
    witness: &Witness,
    dirs: &Dirs,
    dir: &StdPath,
) -> diagnostic::Result<()> {
    let path = dir.join(&witness.file);
    write_text(&path, &source(witness, dirs)?)?;
    parse_one(spec, arguments, &path)
}

/// 解析一份翻译单元；失败时只带编译器的第一行报错（整段 stderr 进不了报告）。
fn parse_one(spec: &LanguageFront, arguments: &[&str], path: &StdPath) -> diagnostic::Result<()> {
    let program = &spec.tool[0];
    let mut command = std::process::Command::new(program);
    command.args(&spec.tool[1..]).args(arguments).arg(path);
    let output = command
        .output()
        .map_err(|error| Error::new(Code::Unsupported, format!("`{program}` 不可执行：{error}")))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let first = stderr
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("解析器没有给出说明");
    Err(Error::new(
        Code::Invalid,
        format!(
            "`{} {}` 拒绝 `{}`：{first}",
            spec.tool.join(" "),
            arguments.join(" "),
            path.display()
        ),
    ))
}

#[cfg(test)]
#[path = "../tests/pipeline_tests.rs"]
mod tests;
