//! `verify`：逐设备把三件事钉在一起——见证跑通、协商选中首档、内核与标量参考对拍（README §11）。
//!
//! 形状矩阵固定在 `pipeline`：生产形状、小形状、非整除形状，后者专门抓"按 16 对齐成立、按 3 不成立"
//! 这类错误。失败一律是错误而不是跳过：设备不在本机是 `unsupported`（退出码 3），声明或输入写错是
//! `invalid`（退出码 2）。两条执行通道（本机 C、nvcc/CUDA）共用同一套步骤，设备差异收在 `Lane` 里。

use crate::build;
use crate::builtin::Catalogue;
use crate::cache::Artifact;
use crate::cap::{self, NoPath, Path, Proof};
use crate::dirs::Dirs;
use crate::emit::Fragment;
use crate::gate::LanguageGate;
use crate::pipeline::{Prepared, Run, parse_shape, rejected_shapes, verify_shapes};
use crate::prof::{Emit, Profile};
use crate::witness;
use crate::{runner, toolchain};
use diagnostic::{Code, Error};
use op::decl::OpDecl;
use op::model::Sample;

/// 设备不在本机时唯一的说法：通道、协商与报告都用它。
const NO_DEVICE: &str = "本机没有该设备，只能真机验证";

/// 按 `emit` 取设备后端，取不到就是「本机没有这门语言的执行前端」。把 `.ok_or_else` 收在一处，
/// 让 CUDA / Metal 两条通道只写 `backend(emit)?`，不各自重复拼错误。
fn backend(emit: Emit) -> diagnostic::Result<&'static dyn crate::backend::Backend> {
    crate::device::spec(emit).ok_or_else(|| Error::new(Code::Unsupported, NO_DEVICE.to_string()))
}

/// 一台设备通过 `L2` 的证据：这次执行真的产生了什么。档案登记值（`tool_ver`）与实测值（`toolchain`）
/// 分开列：两者不一致不改等级，但报告必须把差异说出来。
#[derive(Debug, Clone, PartialEq)]
pub struct Verified {
    /// 用户给的设备名（`auto` 或档案名）。
    pub device: String,
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
}

/// 一台设备的结果。
#[derive(Debug, Clone, PartialEq)]
pub enum VerifyEntry {
    /// 编译、执行、对拍都在容差内。
    Passed(Verified),
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
                step: path.index + 1,
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
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| Error::new(Code::Missing, format!("没有算子 `{op_name}`")))?;
    Ok(VerifyReport {
        op: op_name.to_string(),
        devices: devices.to_vec(),
        entries: devices
            .iter()
            .map(|device| one_device(op, catalogue, dirs, device))
            .collect(),
        negotiations: negotiations(op, catalogue, devices),
    })
}

/// 一台设备的结果：先取档案，再跑见证，最后才轮到形状对拍。回答什么：这次通过还是失败；不回答什么：失败该不该让整次 `verify` 退出。
fn one_device(op: &OpDecl, catalogue: &Catalogue, dirs: &Dirs, device: &str) -> VerifyEntry {
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
        None => verify_one(op, profile, dirs, &proof),
    };
    match done {
        Ok(mut verified) => {
            verified.device = device.to_string();
            verified.witnesses = evidence.device;
            verified.self_checked = evidence.host;
            VerifyEntry::Passed(verified)
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

/// 一台设备的 L2 证据：逐形状编译、执行、与参考对拍。回答什么：产物在形状矩阵每一格上算出来的数
/// 是不是标量参考的数。不回答什么：见证是否成立、协商该不该命中首档。
///
/// # Errors
///
/// 设备不可编译、缺编译器、编译/执行失败、输出个数不符或对拍超出容差时返回说明。
pub fn verify_one(
    op: &OpDecl,
    profile: &Profile,
    dirs: &Dirs,
    proof: &Proof,
) -> diagnostic::Result<Verified> {
    let run = Run::new(op, profile, proof.clone())?;
    let (lane, tool) = Lane::ready(profile)?;
    let path = run
        .negotiate()
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))?;
    check_rejected_shapes(u64::from(run.model.group.max(1)))?;
    let mut verified = blank(&run, tool.as_ref());
    verified.variant = path.variant.clone();
    let sweep = Sweep {
        run: &run,
        dirs,
        lane,
        tool,
        path,
        tolerance: crate::reference::tolerance(run.model.dtype),
        verified,
    };
    sweep.measure()
}

/// 一个形状“怎么编译执行”：设备差异只在这里。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lane {
    /// 本机 C 编译器 + 产物缓存（命中就不调编译器）。
    Host,
    /// nvcc 临时编译、GPU 执行、回读；不进产物缓存，产物因此为空。
    Cuda,
    /// xcrun metal + metallib 临时编译、Apple GPU 执行、回读；不进产物缓存。
    Metal,
}

impl Lane {
    /// 档案声明的 `emit` 对应哪条通道，以及这条通道此刻能不能跑（`None` 表示没有编译器，只能复用产物）。本机没有执行路径时在这里 fail-fast。
    fn ready(profile: &Profile) -> diagnostic::Result<(Self, Option<toolchain::Toolchain>)> {
        match profile.meta.emit {
            Emit::C => Ok((Self::Host, toolchain::detect())),
            Emit::Cuda => Self::ready_device(profile, Self::Cuda),
            Emit::Metal => Self::ready_device(profile, Self::Metal),
            _ => Err(Error::new(Code::Unsupported, unsupported_emit(profile))),
        }
    }

    /// 加速器通道共用的「设备在不在」判断：在则经 [`crate::backend::Driver::toolchain`] 取实测
    /// 工具链返回，不在则 fail-fast（`NO_DEVICE`）。工具链身份只经 `Backend` 缝取，机制层不直接
    /// 调 `detect_nvcc` / `detect_metal`——后端实现搬去 `engine` 后这里一行不改。
    fn ready_device(
        profile: &Profile,
        lane: Self,
    ) -> diagnostic::Result<(Self, Option<toolchain::Toolchain>)> {
        match crate::device::spec(profile.meta.emit) {
            Some(backend) if backend.available(profile) => Ok((lane, backend.toolchain())),
            _ => Err(Error::new(Code::Unsupported, NO_DEVICE.to_string())),
        }
    }
}

/// 一次遍历：流水线、执行通道、选中的路径、对拍阈值，以及逐形状填的报告。
struct Sweep<'a> {
    run: &'a Run<'a>,
    dirs: &'a Dirs,
    lane: Lane,
    tool: Option<toolchain::Toolchain>,
    path: Path,
    tolerance: f64,
    verified: Verified,
}

impl Sweep<'_> {
    /// 逐形状跑一遍：一格失败即整台失败，证据链不接受“其余形状还行”；失败只给错误，不给半份报告。
    fn measure(mut self) -> diagnostic::Result<Verified> {
        for shape in verify_shapes(&self.run.model) {
            self.cell(shape)?;
        }
        Ok(self.verified)
    }

    /// 一格：样本、计划、参考与判据两条通道完全相同，只有“怎么编译执行”按通道分。
    /// 回答什么：这一格算不算数；不回答什么：这一格为什么不算（错误原样上报，不在这里解释）。
    fn cell(&mut self, shape: (u64, u64)) -> diagnostic::Result<()> {
        let (sample, prepared) = self.prepare(shape)?;
        let (bits, artifact) = self.kernel(&sample, &prepared)?;
        let want = (self.run.model.reference)(&sample)?;
        let expect = self.expected(&sample, &want);
        let error = parity(&bits, &want, expect, shape, self.tolerance)?;
        let out = &mut self.verified;
        out.max_rel_err = out.max_rel_err.max(error);
        out.shapes.push(shape);
        out.artifacts.extend(artifact);
        out.fragment = Fragment::new(prepared.source.lines());
        out.key = prepared.key.hex();
        out.tile = prepared.plan.tile.to_string();
        Ok(())
    }

    /// 形状 → 样本 + 产物：两条通道共用的前半段。
    fn prepare(&self, (m, n): (u64, u64)) -> diagnostic::Result<(Sample, Prepared)> {
        let (m, n) = (m as usize, n as usize);
        let sample = (self.run.model.sample)(m, n);
        let prepared = self.run.prepare(&self.path, self.dirs, m, n)?;
        Ok((sample, prepared))
    }

    /// 输出个数该是多少：C 通道看样本声明的 `y`，CUDA/Metal 通道看参考输出的长度。
    fn expected(&self, sample: &Sample, want: &[f32]) -> usize {
        match self.lane {
            Lane::Host => sample.buffer("y").map_or(sample.n, |y| y.values.len()),
            Lane::Cuda | Lane::Metal => want.len(),
        }
    }

    /// 一格怎么跑：返回回读的输出与产物记录（CUDA/Metal 通道每次临时编译，没有产物）。
    fn kernel(
        &self,
        sample: &Sample,
        prepared: &Prepared,
    ) -> diagnostic::Result<(Vec<f32>, Option<Artifact>)> {
        match self.lane {
            Lane::Host => self.host(sample, prepared),
            Lane::Cuda => self.device(sample, prepared, Emit::Cuda),
            Lane::Metal => self.device(sample, prepared, Emit::Metal),
        }
    }

    /// C 通道：缓存编译（命中就不调编译器），产物因此进报告。
    fn host(
        &self,
        sample: &Sample,
        prepared: &Prepared,
    ) -> diagnostic::Result<(Vec<f32>, Option<Artifact>)> {
        let program = prepared.harness(self.run.op, sample, 1)?;
        let (outcome, artifact) = build::execute(
            self.run.profile,
            self.tool.as_ref(),
            &prepared.key,
            &program,
            self.dirs,
        )?;
        Ok((outcome.bits, Some(artifact)))
    }

    /// 设备通道：nvcc / xcrun metal 临时编译、GPU 执行、回读；签名与样本先对齐（与 C 通道同一条判据）。
    ///
    /// CUDA 与 Metal 只差取哪个后端，所以共用这一份次序：两条通道若各写一遍，
    /// “先试并行、再回落语义门”就会有两份口径。
    fn device(
        &self,
        sample: &Sample,
        prepared: &Prepared,
        emit: Emit,
    ) -> diagnostic::Result<(Vec<f32>, Option<Artifact>)> {
        let (backend, launch) = self.channel(emit, sample)?;
        let program = self.program(backend, sample, prepared, &launch)?;
        let (stdout, _stderr) =
            backend.compile_run(self.run.profile, self.dirs, &program, "verify")?;
        Ok((runner::parse_bits(&stdout), None))
    }

    /// 通道前置：签名与样本先对齐（与 C 通道同一条判据），再取后端与派发几何。
    ///
    /// 这一份次序只写一次：两条设备通道若各自校验、各自算几何，那么“对齐了签名”这件事
    /// 就会有两种说法，而几何漂移正是 harness 最难查的那类错。
    fn channel(
        &self,
        emit: Emit,
        sample: &Sample,
    ) -> diagnostic::Result<(&'static dyn crate::backend::Backend, crate::backend::Launch)> {
        self.run.op.signature.check(sample)?;
        let backend = backend(emit)?;
        let launch = crate::execution::launch(self.run.op, sample)?;
        Ok((backend, launch))
    }

    /// 要编译的那份程序：并行网格（生产态形态）优先，该形态没有实现时显式回落单线程语义门。
    ///
    /// 回落必须显式，因为“并行实现缺失”和“并行跑对了”都会写进同一份台账，两者不能混为一谈。
    fn program(
        &self,
        backend: &dyn crate::backend::Backend,
        sample: &Sample,
        prepared: &Prepared,
        launch: &crate::backend::Launch,
    ) -> diagnostic::Result<crate::backend::Source> {
        let signature = &self.run.op.signature;
        let source = &prepared.source.text;
        match backend.parallel(source, signature, sample, launch) {
            Ok(program) => Ok(program),
            Err(error) if error.code() == Code::Unsupported => {
                backend.harness(source, signature, sample)
            }
            Err(error) => Err(error),
        }
    }
}

/// 空报告：设备与工具链身份先写进去，形状层面的字段随每一格填。
fn blank(run: &Run<'_>, tool: Option<&toolchain::Toolchain>) -> Verified {
    Verified {
        device: String::new(),
        vendor: run.profile.meta.vendor.to_string(),
        id: run.profile.meta.id.to_string(),
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
    }
}

/// 对拍：个数与误差两条判据，返回实测最大相对误差。个数先于值：个数不符时连参考都不必逐点比。
/// 它不判断证据等级（等级由见证决定），也不解释误差从哪来。
fn parity(
    bits: &[f32],
    want: &[f32],
    expect: usize,
    (m, n): (u64, u64),
    tolerance: f64,
) -> diagnostic::Result<f64> {
    if bits.len() != expect {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "形状 {m}x{n}：内核只输出 {} 个数，期望 {expect}",
                bits.len()
            ),
        ));
    }
    let error = crate::reference::max_rel_err(bits, want);
    if error > tolerance {
        return Err(Error::new(
            Code::Invalid,
            format!("形状 {m}x{n} 对拍失败：max_rel_err {error:.3e} > {tolerance:.0e}"),
        ));
    }
    Ok(error)
}

/// 非 C 前端的说明：设备在本机与不在本机是两句不同的话。
fn unsupported_emit(profile: &Profile) -> String {
    let declared: Vec<String> = profile
        .provides
        .iter()
        .map(|have| format!("{}={}", have.prim, have.grade))
        .collect();
    let why = if witness::device_available(profile) {
        format!(
            "设备在本机（`emit = {}`），内核执行路径尚未接通（CUDA 已接，其余仍只有见证）：\
             见证已经在这台设备上跑过，内核文本还没有对应的编译/回读路径",
            profile.meta.emit
        )
    } else {
        NO_DEVICE.to_string()
    };
    format!("{why}；档案声明的等级：{}", declared.join(" "))
}

/// 负例也是判据：组大小约束被放宽时，这里必须先在 `verify` 里失败，而不是等到错数据。
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
    use super::*;
    use crate::gate::LanguageGate;

    fn failed(label: &str) -> VerifyEntry {
        VerifyEntry::Failed {
            label: label.to_string(),
            error: Error::new(Code::Invalid, "测试失败"),
        }
    }

    #[test]
    fn witness_note_is_absent_when_nothing_ran() {
        assert!(witness_note(0, 0, &LanguageGate::Covered).is_none());
        // Rejected 不追加句子：拒绝已写进 Evidence::failure，这里再印一次只是重复。
        let rejected = LanguageGate::Rejected {
            reason: "第一行报错".to_string(),
        };
        assert!(witness_note(0, 0, &rejected).is_none());
    }

    #[test]
    fn witness_note_counts_device_and_host_separately() {
        let note = witness_note(2, 3, &LanguageGate::Covered).expect("有结论");
        assert!(note.contains("见证设备执行 2 条通过"), "{note}");
        assert!(note.contains("见证自检 3 条通过"), "{note}");
        // Covered 表示设备执行已覆盖，不再重复语言门。
        assert!(!note.contains("语言门"), "{note}");
    }

    #[test]
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

    #[test]
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

    #[test]
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
