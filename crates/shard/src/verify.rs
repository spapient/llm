//! `verify`：逐设备把三件事钉在一起——见证跑通、协商选中首档、内核与标量参考对拍（README §11）。

use crate::gate::LanguageGate;
use crate::pipeline::{DeviceOutcome, Prepared, Run, parse_shape, rejected_shapes, verify_shapes};
use crate::toolchain;
use crate::witness;
use cutile::builtin::Catalogue;
use cutile::cache::Artifact;
use cutile::cap::{self, NoPath, Path, Proof};
use cutile::decl::OpDecl;
use cutile::dirs::Dirs;
use cutile::emit::Fragment;
use cutile::model::Sample;
use cutile::prof::{Emit, Profile};
use diagnostic::{Code, Error};

/// 一台设备通过 `L2` 的证据：这次执行真的产生了什么。档案登记值（`tool_ver`）与实测值（`toolchain`）
/// 分开列：两者不一致不改等级，但报告必须把差异说出来。
#[derive(Debug, Clone, PartialEq)]
pub struct Verified {
    /// 用户给的设备名（`auto` 或档案名）。
    pub device: String,
    /// 这一台走的是并行网格程序还是回落单线程语义门。
    pub channel: Channel,
    /// 厂商。
    pub vendor: String,
    /// 档案 id。
    pub id: String,
    /// 协商选中的变体。
    pub variant: String,
    /// 选中的 tile。
    pub tile: String,
    /// 实际跑过的形状（形状矩阵）。
    pub shapes: Vec<(u64, u64)>,
    /// 执行通过的见证条数（每条 `grade ≥ L2` 的供给一条）。
    pub witnesses: usize,
    /// 本机自检通过的见证条数（设备执行跑不到的那些）；与 `witnesses` 相加是声明的全部见证。
    pub self_checked: usize,
    /// 与标量参考的相对误差（取全部形状的最大值）。
    pub max_rel_err: f64,
    /// 源文件行数（§16 口径）。
    pub fragment: Fragment,
    /// 档案登记的工具版本：只与 `toolchain` 对照，不参与任何判定。
    pub tool_ver: String,
    /// 实测的工具链身份（`命令@版本`）；本机没有编译器时为 `None`（那一轮只复用已落盘产物）。
    pub toolchain: Option<String>,
    /// 实测工具链是否与档案登记的版本相容（只比主版本号）：不一致不改等级，但报告要说出差异。
    pub toolchain_agrees: bool,
    /// 这次编译用的参数（档案的 `[compile].flags`）：证据与产物都对它成立。
    pub flags: String,
    /// 缓存键（十六进制）。
    pub key: String,
    /// 编译产物：形状矩阵的每一项一份（`reused` 为真表示那次没调用编译器）。
    pub artifacts: Vec<Artifact>,
    /// 全部形状的真实 harness 计时样本；CUDA 不生成 Artifact，但同样保留计时。
    pub timing: crate::runner::Timing,
}

/// 设备通道跑的是哪一份程序。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// 并行网格程序（按 `[chunk]` / `[grid]` 声明发射）。
    Parallel,
    /// 单线程语义门（该形态尚无并行实现，或本来就是串行骨架）。
    Harness,
    /// C 参考通道：串行是它的定义，不是缺口——并行形态由设备通道证。
    Reference,
}

impl Channel {
    /// 报告用的一栏。
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Parallel => "并行",
            Self::Harness => "回落单线程",
            Self::Reference => "语义基准",
        }
    }
}

/// 一台设备的结果。
#[derive(Debug, Clone, PartialEq)]
pub enum VerifyEntry {
    /// 编译、执行、对拍都在容差内。
    Passed(Box<Verified>),
    /// 未通过；`label` 是档案的呈现名（`vendor(id)`，没有档案时就是用户给的设备名）。
    Failed {
        /// 呈现名。
        label: String,
        /// 失败原因。
        error: Error,
    },
}

impl VerifyEntry {
    /// 失败原因；通过时为 `None`。
    pub fn error(&self) -> Option<&Error> {
        match self {
            Self::Passed(_) => None,
            Self::Failed { error, .. } => Some(error),
        }
    }
}

/// 一台设备的协商结论（README §16 判据「协商 3/3 命中首档」）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Negotiation {
    /// 命中降级链的首选变体；`of` 是链长。
    Head {
        /// 变体名。
        variant: String,
        /// 降级链的变体总数。
        of: usize,
    },
    /// 首选不可行，落到链上第 `step` 个变体（从 1 数起）。
    Fallback {
        /// 变体名。
        variant: String,
        /// 选中变体在链上的位置（从 1 数起）。
        step: usize,
        /// 降级链的变体总数。
        of: usize,
    },
    /// 整条链都不可行：`of` 是链长，`unmet` 是首选变体缺的原语（去重，按 `needs` 顺序）。
    NoPath {
        /// 降级链的变体总数。
        of: usize,
        /// 首选变体没被满足的原语。
        unmet: Vec<String>,
    },
    /// 没有这份档案，协商无从谈起。
    Unknown {
        /// 取档案失败的原因。
        reason: String,
    },
}

impl Negotiation {
    /// 是否命中首档：判据的唯一取值。
    pub fn head_hit(&self) -> bool {
        matches!(self, Self::Head { .. })
    }

    /// 协商是否给出了可行变体（首档或降级）。
    pub fn has_path(&self) -> bool {
        matches!(self, Self::Head { .. } | Self::Fallback { .. })
    }
}

/// 一次 `verify` 的逐设备结果。
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyReport {
    /// 算子名。
    pub op: String,
    /// 请求验证的设备（原样，含 `auto`）。
    pub devices: Vec<String>,
    /// 与 `devices` 一一对应的结果。
    pub entries: Vec<VerifyEntry>,
    /// 与 `devices` 一一对应的协商结论：只看数据，本机缺设备时也有答案。
    pub negotiations: Vec<Negotiation>,
}

impl VerifyReport {
    /// 未通过的项。
    pub fn failures(&self) -> Vec<&Error> {
        self.entries.iter().filter_map(VerifyEntry::error).collect()
    }

    /// 命中降级链首档的设备数：`(命中, 总数)`。
    pub fn head_hits(&self) -> (usize, usize) {
        (
            self.negotiations
                .iter()
                .filter(|negotiation| negotiation.head_hit())
                .count(),
            self.negotiations.len(),
        )
    }
}

/// 逐设备的协商结论，与 `devices` 一一对应；判据是 `cap::negotiate`——**按档案自述**的那一档，不是这次的 `Proof`。
pub fn negotiations(op: &OpDecl, catalogue: &Catalogue, devices: &[String]) -> Vec<Negotiation> {
    devices
        .iter()
        .map(|device| negotiate(op, catalogue, device))
        .collect()
}

/// 按档案自述协商，把结果折成一个结论。回答什么：这台设备在**这份档案**下命中哪一档、落到链上第几步；
/// 不回答什么：这次执行有没有证据。
fn negotiate(op: &OpDecl, catalogue: &Catalogue, device: &str) -> Negotiation {
    let of = op.variants.len();
    match catalogue.resolve(device) {
        Ok(profile) => match cap::negotiate(op, profile) {
            Ok(path) if path.is_head() => Negotiation::Head {
                variant: path.variant,
                of,
            },
            Ok(path) => Negotiation::Fallback {
                variant: path.variant,
                step: path.idx + 1,
                of,
            },
            Err(no_path) => unmet(&no_path),
        },
        Err(error) => Negotiation::Unknown {
            reason: error.to_string(),
        },
    }
}

/// 首选变体缺哪些原语：只报真正缺的，一个原语只报一次，顺序按 `needs`。
fn unmet(no_path: &NoPath) -> Negotiation {
    let mut prims: Vec<String> = no_path
        .needs
        .iter()
        .filter(|report| !report.is_satisfied())
        .map(|report| report.need.prim.to_string())
        .collect();
    let mut seen = std::collections::BTreeSet::new();
    prims.retain(|prim| seen.insert(prim.clone()));
    Negotiation::NoPath {
        of: no_path.chain.len(),
        unmet: prims,
    }
}

/// 逐设备的 L2 证据与协商结论。见证先跑：一条 `grade ≥ L2` 的供给如果没有执行结果，后面的形状对拍
/// 就不再是证据链的一部分；协商结论另算，它只看档案自述。
///
/// 回答什么：每台设备这次是否通过、失败原因与协商档位。不回答什么：某台设备为什么不该被选中。
///
/// # Errors
///
/// 算子不存在时返回说明；设备失败不算错误（写在 [`VerifyReport`] 里）。
pub fn verify(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    devices: &[String],
) -> diagnostic::Result<VerifyReport> {
    verify_with_repeat(catalogue, dirs, op_name, devices, 1)
}

/// 与 [`verify`] 相同，但每个形状用给定重复次数产生真实计时样本；零按一次处理。
pub fn verify_with_repeat(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    devices: &[String],
    repeat: usize,
) -> diagnostic::Result<VerifyReport> {
    let repeat = repeat.max(1);
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| Error::new(Code::Missing, format!("没有算子 `{op_name}`")))?;
    Ok(VerifyReport {
        op: op_name.to_string(),
        devices: devices.to_vec(),
        entries: devices
            .iter()
            .map(|device| one_device(op, catalogue, dirs, device, repeat))
            .collect(),
        negotiations: negotiations(op, catalogue, devices),
    })
}

/// 一台设备的结果：先取档案，再跑见证，最后才轮到形状对拍。回答什么：这次通过还是失败；不回答什么：失败该不该让整次 `verify` 退出。
fn one_device(
    op: &OpDecl,
    catalogue: &Catalogue,
    dirs: &Dirs,
    device: &str,
    repeat: usize,
) -> VerifyEntry {
    let profile = match catalogue.resolve(device) {
        Ok(profile) => profile,
        Err(error) => {
            return VerifyEntry::Failed {
                label: device.to_string(),
                error,
            };
        }
    };
    let label = cap::device_label(profile);
    let evidence = witness::establish(profile, dirs);
    let note = witness_note(evidence.device, evidence.host, &evidence.language);
    let proof = evidence.proof(profile);
    let done = match evidence.failure {
        Some(error) => Err(error),
        None => verify_one_with_repeat(op, profile, dirs, &proof, repeat),
    };
    match done {
        Ok(mut verified) => {
            verified.device = device.to_string();
            verified.witnesses = evidence.device;
            verified.self_checked = evidence.host;
            VerifyEntry::Passed(Box::new(verified))
        }
        Err(error) => VerifyEntry::Failed {
            label,
            error: match note {
                Some(note) => error.with(note),
                None => error,
            },
        },
    }
}

/// 失败文案后面补一句这次真的跑过了哪些见证。设备执行、本机自检与语言门是三种效力不同的结论，不合成
/// 一个数；三样都没有时不补句子。它只追加，不改写 `Evidence::failure`。
fn witness_note(device: usize, host: usize, gate: &LanguageGate) -> Option<String> {
    let mut parts = count_parts(device, host);
    parts.extend(gate_parts(gate));
    if parts.is_empty() {
        return None;
    }
    Some(parts.join("、"))
}

/// 两档执行的计数：各说各的，不合成一个总和。
fn count_parts(device: usize, host: usize) -> Vec<String> {
    let mut parts = Vec::new();
    if device > 0 {
        parts.push(format!("见证设备执行 {device} 条通过"));
    }
    if host > 0 {
        parts.push(format!(
            "见证自检 {host} 条通过（host C ABI，不含设备执行）"
        ));
    }
    parts
}

/// 语言门这一档：设备执行已覆盖、或被自己的语言拒绝时都不追加——前者是重复，后者已经写进
/// `Evidence::failure`，再说一遍只会把同一句报错印两次。
fn gate_parts(gate: &LanguageGate) -> Option<String> {
    match gate {
        LanguageGate::Passed { front, witnesses } => {
            Some(format!("语言门 {front} 通过（{witnesses} 条）"))
        }
        LanguageGate::Absent { reason } => Some(format!("语言门不可用：{reason}")),
        _ => None,
    }
}

/// 逐形状编译、执行并与标量参考对拍，形成一台设备的 L2 证据。
///
/// # Errors
///
/// 设备语言没有执行前端、缺编译器、协商失败、编译执行失败或任一对拍判据不成立。
pub fn verify_one(
    op: &OpDecl,
    profile: &Profile,
    dirs: &Dirs,
    proof: &Proof,
) -> diagnostic::Result<Verified> {
    verify_one_with_repeat(op, profile, dirs, proof, 1)
}

/// 单设备逐形状对拍，并保留每个形状的重复执行计时。
pub fn verify_one_with_repeat(
    op: &OpDecl,
    profile: &Profile,
    dirs: &Dirs,
    proof: &Proof,
    repeat: usize,
) -> diagnostic::Result<Verified> {
    let run = Run::new(op, profile, proof.clone())?;
    let tool = host_tool(profile)?;
    let path = run
        .negotiate()
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))?;
    check_rejected_shapes(u64::from(run.model.group.max(1)))?;
    let mut verified = blank(&run, tool.as_ref());
    verified.variant = path.variant.clone();
    Sweep {
        run: &run,
        dirs,
        repeat: repeat.max(1),
        tool,
        path,
        tolerance: cutile::reference::tolerance(run.model.dtype),
        verified,
    }
    .measure()
}

fn host_tool(profile: &Profile) -> diagnostic::Result<Option<toolchain::Toolchain>> {
    match profile.meta.emit {
        Emit::C => Ok(toolchain::detect()),
        Emit::Cuda => Ok(toolchain::probe(&profile.meta.tool)),
        _ => Err(Error::new(
            Code::Unsupported,
            format!(
                "设备 `{}` 的生产态 verify 走 prof::runtime FFI 后端岛；shard 当前通道为 C 与 CUDA",
                profile.meta.id
            ),
        )),
    }
}

struct Sweep<'a> {
    run: &'a Run<'a>,
    dirs: &'a Dirs,
    repeat: usize,
    tool: Option<toolchain::Toolchain>,
    path: Path,
    tolerance: f64,
    verified: Verified,
}

impl Sweep<'_> {
    fn measure(mut self) -> diagnostic::Result<Verified> {
        for shape in verify_shapes(&self.run.model) {
            self.cell(shape)?;
        }
        Ok(self.verified)
    }

    fn cell(&mut self, shape: (u64, u64)) -> diagnostic::Result<()> {
        let (sample, prepared) = self.prepare(shape)?;
        self.verified.channel = channel(self.run.profile.meta.emit);
        let (bits, timing, artifact) = self.host(&sample, &prepared)?;
        self.record_cell(sample, prepared, shape, bits, timing, artifact)
    }

    /// 对拍与结果登记同组，执行路径只负责产出证据。
    fn record_cell(
        &mut self,
        sample: Sample,
        prepared: Prepared,
        shape: (u64, u64),
        bits: Vec<f32>,
        timing: crate::runner::Timing,
        artifact: Option<Artifact>,
    ) -> diagnostic::Result<()> {
        let want = (self.run.model.reference)(&sample)?;
        let want = want.as_f32()?;
        let expect = sample.buf("y").map_or(sample.n, |y| y.values.len());
        let error = parity(&bits, want, expect, shape, self.tolerance)?;
        let out = &mut self.verified;
        out.max_rel_err = out.max_rel_err.max(error);
        out.shapes.push(shape);
        out.artifacts.extend(artifact);
        out.timing.samples_ns.extend(timing.samples_ns);
        out.fragment = Fragment::new(prepared.source.lines());
        out.key = prepared.key.hex();
        out.tile = prepared.plan.tile.to_string();
        Ok(())
    }

    fn prepare(&self, (m, n): (u64, u64)) -> diagnostic::Result<(Sample, Prepared)> {
        let (m, n) = (m as usize, n as usize);
        let sample = self.run.sample(m, n);
        let prepared = self.run.prepare(&self.path, self.dirs, m, n)?;
        Ok((sample, prepared))
    }

    fn host(&self, sample: &Sample, prepared: &Prepared) -> diagnostic::Result<DeviceOutcome> {
        let program = prepared.harness(self.run.op, sample, self.repeat)?;
        crate::pipeline::run_backend(
            self.run.profile,
            self.dirs,
            prepared,
            &program,
            &self.run.op.name,
            self.tool.as_ref(),
        )
    }
}

fn channel(emit: Emit) -> Channel {
    match emit {
        Emit::C => Channel::Reference,
        Emit::Cuda => Channel::Parallel,
        _ => Channel::Harness,
    }
}

fn blank(run: &Run<'_>, tool: Option<&toolchain::Toolchain>) -> Verified {
    Verified {
        device: String::new(),
        channel: Channel::Harness,
        vendor: run.profile.meta.vendor.clone(),
        id: run.profile.meta.id.clone(),
        variant: String::new(),
        tile: String::new(),
        shapes: Vec::new(),
        witnesses: 0,
        self_checked: 0,
        max_rel_err: 0.0,
        fragment: Fragment::new(0),
        tool_ver: run.profile.meta.tool_ver.clone(),
        toolchain: tool.map(toolchain::Toolchain::id),
        toolchain_agrees: tool.is_some_and(|tool| tool.agrees_with(&run.profile.meta.tool_ver)),
        flags: run.profile.flags.join(" "),
        key: String::new(),
        artifacts: Vec::new(),
        timing: crate::runner::Timing::default(),
    }
}

fn parity(
    bits: &[f32],
    want: &[f32],
    expect: usize,
    (m, n): (u64, u64),
    tolerance: f64,
) -> diagnostic::Result<f64> {
    crate::pipeline::parity(
        bits,
        want,
        expect,
        tolerance,
        &format!("形状 {m}x{n}"),
        Code::Invalid,
    )
}

fn check_rejected_shapes(group: u64) -> diagnostic::Result<()> {
    for (m, n) in rejected_shapes(group) {
        let shape = format!("{m}x{n}");
        if parse_shape(&shape, group).is_ok() {
            return Err(Error::new(
                Code::Internal,
                format!("形状 `{shape}` 必须被拒绝：M 必须是组大小 {group} 的整数倍"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! `cutile::verify` 的用例：校验判据与失败口径。
    //!
    //! 生产文件只留执行与判据，用例集中于此。

    use super::*;
    use crate::fixtures::{catalogue, profile};
    use crate::gate::LanguageGate;

    fn failed(label: &str) -> VerifyEntry {
        VerifyEntry::Failed {
            label: label.to_string(),
            error: Error::new(Code::Invalid, "测试失败"),
        }
    }

    testkit::test_cases! {
    fn witness_note_is_absent_when_nothing_ran() {
        assert!(witness_note(0, 0, &LanguageGate::Covered).is_none());
        // Rejected 不追加句子：拒绝已写进 Evidence::failure，这里再印一次只是重复。
        let rejected = LanguageGate::Rejected {
            reason: "第一行报错".to_string(),
        };
        assert!(witness_note(0, 0, &rejected).is_none());
    }

    /// 非 C 前端的生产态 verify 走 `prof::runtime` FFI 后端岛，shard 只保留 C 参考通道。
    fn the_ascend_emit_reports_the_prof_runtime_lane() {
        let profile = profile("ascend");
        let error = host_tool(&profile).expect_err("shard 不再生成设备 verify 程序");
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.message().contains("prof::runtime"), "{error}");
    }

    /// 再往外一层：整条 [`verify_one`] 的结论也必须落在 prof::runtime 后端岛上。
    fn verifying_ascend_reports_the_prof_runtime_lane() {
        let catalogue = catalogue();
        let profile = catalogue.profile("ascend").cloned().expect("ascend 档案");
        let op = catalogue.op("gemv_i4").expect("gemv_i4 声明");
        let dirs = Dirs::under(std::env::temp_dir().join("cutile-verify-ascend"));
        let error = verify_one(op, &profile, &dirs, &cutile::cap::Proof::none())
            .expect_err("shard 不再生成设备 verify 程序");
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.message().contains("prof::runtime"), "{error}");
    }

    fn witness_note_counts_device_and_host_separately() {
        let note = witness_note(2, 3, &LanguageGate::Covered).expect("有结论");
        assert!(note.contains("见证设备执行 2 条通过"), "{note}");
        assert!(note.contains("见证自检 3 条通过"), "{note}");
        // Covered 表示设备执行已覆盖，不再重复语言门。
        assert!(!note.contains("语言门"), "{note}");
    }

    fn witness_note_renders_the_language_gate_branches() {
        let passed = LanguageGate::Passed {
            front: "clang@21".to_string(),
            witnesses: 4,
        };
        assert!(
            witness_note(0, 0, &passed)
                .expect("有结论")
                .contains("语言门 clang@21 通过（4 条）")
        );
        let absent = LanguageGate::Absent {
            reason: "缺 clang".to_string(),
        };
        assert!(
            witness_note(0, 0, &absent)
                .expect("有结论")
                .contains("语言门不可用：缺 clang")
        );
    }

    fn a_negotiation_classifies_head_and_path() {
        let head = Negotiation::Head {
            variant: "v0".to_string(),
            of: 3,
        };
        assert!(head.head_hit() && head.has_path());
        let fallback = Negotiation::Fallback {
            variant: "v1".to_string(),
            step: 2,
            of: 3,
        };
        assert!(!fallback.head_hit() && fallback.has_path());
        let no_path = Negotiation::NoPath {
            of: 3,
            unmet: vec!["wmma".to_string()],
        };
        assert!(!no_path.head_hit() && !no_path.has_path());
        let unknown = Negotiation::Unknown {
            reason: "无档案".to_string(),
        };
        assert!(!unknown.head_hit() && !unknown.has_path());
    }

    fn the_report_collects_failures_and_head_hits() {
        let report = VerifyReport {
            op: "gemv".to_string(),
            devices: vec!["a".to_string(), "b".to_string()],
            entries: vec![failed("a"), failed("b")],
            negotiations: vec![
                Negotiation::Head {
                    variant: "v0".to_string(),
                    of: 1,
                },
                Negotiation::NoPath {
                    of: 1,
                    unmet: Vec::new(),
                },
            ],
        };
        assert_eq!(report.failures().len(), 2, "两个 Failed 条目各给出一条错误");
        assert_eq!(report.head_hits(), (1, 2), "两台设备里一台命中首档");
    }
    }
}
