//! `verify`：逐设备执行见证 + 协商 + 拼接 + 编译 + 与标量参考对拍。

use crate::backend;
use crate::build;
use crate::builtin::Catalogue;
use crate::cap::{self, Proof};
use crate::emit::Fragment;
use crate::pipeline::{Run, parse_shape, rejected_shapes, verify_shapes};
use crate::witness;
use diagnostic::{Code, Error};
use llm_config::Settings;
use op::decl::OpDecl;
use prof::prof::Profile;

/// 一台设备通过 L2 的证据。
#[derive(Debug, Clone, PartialEq)]
pub struct Verified {
    /// 用户给的设备名（`auto` 或档案名）。
    pub device: String,
    /// 厂商。
    pub vendor: String,
    /// 档案 id。
    pub id: String,
    /// 变体。
    pub variant: String,
    /// tile。
    pub tile: String,
    /// 实际跑过的形状（`verify` 的形状矩阵）。
    pub shapes: Vec<(u64, u64)>,
    /// 执行通过的见证条数（每条 `grade ≥ L2` 的供给一条，结论是 L2 证据）。
    pub witnesses: usize,
    /// 本机自检通过的见证条数（设备执行跑不到的那些：非 C 设备的全部，C 设备的 `L0`/`L1`）。
    ///
    /// 与 [`Verified::witnesses`] 相加等于这台设备声明的全部见证：一份见证只被执行一次，
    /// 由能给出更强结论的那个范围执行。
    pub self_checked: usize,
    /// 与标量参考的相对误差（取全部形状的最大值）。
    pub max_rel_err: f64,
    /// 源文件行数与 §16 的口径。
    pub fragment: Fragment,
    /// 档案登记的工具版本：与 [`Verified::toolchain`] 对照用，不参与任何判定。
    pub tool_ver: String,
    /// 实测的工具链身份（`命令@版本`）；本机没有编译器时为 `None`（那一轮只复用已落盘产物）。
    pub toolchain: Option<String>,
    /// 实测工具链是否与档案登记的版本相容（只比主版本号）。
    ///
    /// 不相容不改等级：证据可以在别的机器上取得，登记值说明的是"这条证据按哪个版本记的"。
    /// 但报告必须把差异说出来，否则"换了一台机器、换了一个编译器"会静悄悄地发生。
    pub toolchain_agrees: bool,
    /// 这次编译用的参数（档案的 `[compile].flags`）：证据与产物都对它成立。
    pub flags: String,
    /// 缓存键（十六进制）。
    pub key: String,
    /// 编译产物：形状矩阵的每一项一份（`reused` 为真表示那次没调用编译器）。
    pub artifacts: Vec<crate::cache::Artifact>,
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

/// 一台设备的协商结论（DESIGN §16 判据「协商 3/3 命中首档」）。
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

/// 逐设备的协商结论，与 `devices` 一一对应。
///
/// 判据是 `cap::negotiate`——**按档案自述**协商的那一档，不是这次执行得到的 `Proof`。
pub fn negotiations(op: &OpDecl, catalogue: &Catalogue, devices: &[String]) -> Vec<Negotiation> {
    devices
        .iter()
        .map(|device| negotiate(op, catalogue, device))
        .collect()
}

fn negotiate(op: &OpDecl, catalogue: &Catalogue, device: &str) -> Negotiation {
    let profile = match catalogue.resolve(device) {
        Ok(profile) => profile,
        Err(error) => {
            return Negotiation::Unknown {
                reason: error.to_string(),
            };
        }
    };
    let of = op.variants.len();
    match cap::negotiate(op, profile) {
        Ok(path) if path.is_head() => Negotiation::Head {
            variant: path.variant,
            of,
        },
        Ok(path) => Negotiation::Fallback {
            variant: path.variant,
            step: path.index + 1,
            of,
        },
        Err(no_path) => {
            // `needs` 是首选变体的完整明细（含已满足的项）：只报真正缺的那些，且一个原语只报一次。
            let mut unmet: Vec<String> = Vec::new();
            for report in no_path.needs.iter().filter(|report| !report.is_satisfied()) {
                let prim = report.need.prim.to_string();
                if !unmet.contains(&prim) {
                    unmet.push(prim)
                }
            }
            Negotiation::NoPath {
                of: no_path.chain.len(),
                unmet,
            }
        }
    }
}

/// 见证先跑：一条 `grade ≥ L2` 的供给如果没有自己的执行结果，后面的形状对拍就不再
/// 是证据链的一部分（DESIGN §11.3）。
///
/// # Errors
///
/// 算子不存在时返回说明；设备失败不算错误（写在 [`VerifyReport`] 里）。
pub fn verify(
    catalogue: &Catalogue,
    settings: &Settings,
    op_name: &str,
    devices: &[String],
) -> diagnostic::Result<VerifyReport> {
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| Error::new(Code::Missing, format!("没有算子 `{op_name}`")))?;
    let mut entries = Vec::with_capacity(devices.len());
    for device in devices {
        let profile = match catalogue.resolve(device) {
            Ok(profile) => profile,
            Err(error) => {
                entries.push(VerifyEntry::Failed {
                    label: device.clone(),
                    error,
                });
                continue;
            }
        };
        let label = format!("{}({})", profile.meta.vendor, profile.meta.id);
        // 见证先跑，且三条命令共用这一份结论：同一台设备因此可以同时“协商可行”与“见证不成立”。
        let evidence = crate::witness::establish(profile, settings);
        let witnesses = evidence.device;
        let self_checked = evidence.host;
        if let Some(error) = evidence.failure {
            entries.push(VerifyEntry::Failed {
                label,
                error: match witness_note(witnesses, self_checked, &evidence.language) {
                    Some(note) => error.with(note),
                    None => error,
                },
            });
            continue;
        }
        entries.push(
            match verify_one(op, profile, settings, &evidence.proof(profile)) {
                Ok(mut verified) => {
                    verified.device = device.clone();
                    verified.witnesses = witnesses;
                    verified.self_checked = self_checked;
                    VerifyEntry::Passed(verified)
                }
                Err(error) => VerifyEntry::Failed {
                    label,
                    error: match witness_note(witnesses, self_checked, &evidence.language) {
                        Some(note) => error.with(note),
                        None => error,
                    },
                },
            },
        );
    }
    Ok(VerifyReport {
        op: op_name.to_string(),
        devices: devices.to_vec(),
        entries,
        negotiations: negotiations(op, catalogue, devices),
    })
}

/// 失败文案后面补一句"哪些见证真的跑过了"。
///
/// 设备执行、自检、语法门是三种效力不同的结论，不能合成一个数。三样都没有时不补句子。
fn witness_note(device: usize, host: usize, gate: &crate::witness::LanguageGate) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if device > 0 {
        parts.push(format!("见证设备执行 {device} 条通过"));
    }
    if host > 0 {
        parts.push(format!(
            "见证自检 {host} 条通过（host C ABI，不含设备执行）"
        ));
    }
    match gate {
        // 设备执行已经覆盖：再说一次语法门只是重复。
        crate::witness::LanguageGate::Covered => {}
        crate::witness::LanguageGate::Passed { front, witnesses } => {
            parts.push(format!("语言门 {front} 通过（{witnesses} 条）"));
        }
        crate::witness::LanguageGate::Absent { reason } => {
            parts.push(format!("语言门不可用：{reason}"));
        }
        // 拒绝已经写进 `Evidence::failure`，这里再说一遍只会把同一句报错印两次。
        crate::witness::LanguageGate::Rejected { .. } => {}
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("、"))
    }
}

/// 一台设备的 L2 证据：逐形状编译、执行、与参考对拍。
///
/// # Errors
///
/// 设备不可编译、缺编译器、编译/执行失败、输出个数不符或对拍超出容差时返回说明。
pub fn verify_one(
    op: &OpDecl,
    profile: &Profile,
    settings: &Settings,
    proof: &Proof,
) -> diagnostic::Result<Verified> {
    let run = Run::new(op, profile, proof.clone())?;
    if profile.meta.emit == crate::prof::Emit::Cuda {
        return verify_one_cuda(&run, settings);
    }
    if profile.meta.emit != crate::prof::Emit::C {
        return Err(Error::new(Code::Unsupported, unsupported_emit(profile)));
    }

    let path = run
        .negotiate()
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))?;
    let group = u64::from(run.model.group.max(1));
    check_rejected_shapes(group)?;

    // 没有编译器不等于没有结论：产物若已落盘，`build::execute` 会按产物旁的身份文件复用它。
    let tool = backend::detect();
    let tolerance = crate::reference::tolerance(run.model.dtype);

    let mut max_rel_err = 0.0f64;
    let mut shapes = Vec::new();
    let mut artifacts = Vec::new();
    let mut fragment = Fragment::new(0);
    let mut key = String::new();
    let mut tile = String::new();

    for (m, n) in verify_shapes(&run.model) {
        let (m, n) = (m as usize, n as usize);
        let sample = (run.model.sample)(m, n);
        let prepared = run.prepare(&path, settings, m, n)?;
        let run_harness = prepared.harness(op, &sample, 1)?;
        // 产物进缓存：命中就不调用编译器，L2 证据因此不必每次重编一遍。
        let (outcome, artifact) = build::execute(
            profile,
            tool.as_ref(),
            &prepared.key,
            &run_harness,
            settings,
        )?;
        let expected_len = sample
            .buffer("y")
            .map(|buffer| buffer.values.len())
            .unwrap_or(sample.n);
        if outcome.bits.len() != expected_len {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "形状 {m}x{n}：内核只输出 {} 个数，期望 {expected_len}",
                    outcome.bits.len()
                ),
            ));
        }
        let want = (run.model.reference)(&sample)?;
        let error = crate::reference::max_rel_err(&outcome.bits, &want);
        if error > tolerance {
            // 对拍失败不是“环境问题”：产物不满足它自己声明的语义，类别是 `invalid`。
            return Err(Error::new(
                Code::Invalid,
                format!("形状 {m}x{n} 对拍失败：max_rel_err {error:.3e} > {tolerance:.0e}"),
            ));
        }
        max_rel_err = max_rel_err.max(error);
        fragment = Fragment::new(prepared.source.lines());
        key = prepared.key.hex();
        tile = prepared.plan.tile.to_string();
        shapes.push((m as u64, n as u64));
        artifacts.push(artifact);
    }

    Ok(Verified {
        device: String::new(),
        vendor: profile.meta.vendor.to_string(),
        id: profile.meta.id.to_string(),
        variant: path.variant,
        tile,
        shapes,
        witnesses: 0,
        self_checked: 0,
        max_rel_err,
        fragment,
        tool_ver: profile.meta.tool_ver.clone(),
        toolchain: tool.as_ref().map(backend::Toolchain::id),
        toolchain_agrees: tool
            .as_ref()
            .is_some_and(|tool| tool.agrees_with(&profile.meta.tool_ver)),
        flags: profile.flags.join(" "),
        key,
        artifacts,
    })
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
        "本机没有该设备，只能真机验证".to_string()
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

/// 设备可用性 guard：设备不在本机时给出统一的 Unsupported，可用时放行。
fn require_device(profile: &Profile) -> diagnostic::Result<()> {
    if witness::device_available(profile) {
        Ok(())
    } else {
        Err(Error::new(
            Code::Unsupported,
            "本机没有该设备，只能真机验证".to_string(),
        ))
    }
}

/// CUDA 设备的 L2 证据：逐形状生成 `.cu`、nvcc 编译、GPU 单线程执行、回读、与标量参考对拍。
///
/// 与 C 路径同形状、同参考、同容差，差别只在编译执行走 [`crate::witness::compile_run_cuda`]。
/// 当前不进 build 产物缓存（每次临时编译，与 witness 路径一致），故 `artifacts` 为空，如实标注。
///
/// # Errors
///
/// 协商失败、编译/执行失败、输出个数不符或对拍超出容差时返回说明。
fn verify_one_cuda(run: &Run<'_>, settings: &Settings) -> diagnostic::Result<Verified> {
    // 设备不在本机（无 nvcc 工具链）时 fail-fast：后面的编译/回读整条都无意义，
    // 统一返回 Unsupported，由 verify 主流程再挂上“见证自检 N 条通过”的说明。
    require_device(run.profile)?;
    let path = run
        .negotiate()
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))?;
    let tolerance = crate::reference::tolerance(run.model.dtype);
    let nvcc = backend::detect_nvcc();

    let mut max_rel_err = 0.0f64;
    let mut shapes = Vec::new();
    let mut fragment = Fragment::new(0);
    let mut key = String::new();
    let mut tile = String::new();

    for (m, n) in verify_shapes(&run.model) {
        let (m, n) = (m as usize, n as usize);
        let sample = (run.model.sample)(m, n);
        let prepared = run.prepare(&path, settings, m, n)?;
        // CUDA 走自己的 harness；签名与样本仍要先对齐（与 C 路径同一条判据）。
        run.op.signature.check(&sample)?;
        let program = backend::cuda_harness(&prepared.source.text, &run.op.signature, &sample)?;
        let (stdout, _stderr) =
            witness::compile_run_cuda(run.profile, settings, &program, "verify")?;
        let bits = backend::parse_bits(&stdout);
        let want = (run.model.reference)(&sample)?;
        let error = cuda_parity(&bits, &want, m, n, tolerance)?;
        max_rel_err = max_rel_err.max(error);
        fragment = Fragment::new(prepared.source.lines());
        key = prepared.key.hex();
        tile = prepared.plan.tile.to_string();
        shapes.push((m as u64, n as u64));
    }

    Ok(Verified {
        device: String::new(),
        vendor: run.profile.meta.vendor.to_string(),
        id: run.profile.meta.id.to_string(),
        variant: path.variant,
        tile,
        shapes,
        witnesses: 0,
        self_checked: 0,
        max_rel_err,
        fragment,
        tool_ver: run.profile.meta.tool_ver.clone(),
        toolchain: nvcc.as_ref().map(backend::Toolchain::id),
        toolchain_agrees: nvcc
            .as_ref()
            .is_some_and(|tool| tool.agrees_with(&run.profile.meta.tool_ver)),
        flags: run.profile.flags.join(" "),
        key,
        artifacts: Vec::new(),
    })
}

/// CUDA 对拍：先核对输出个数，再算最大相对误差，超容差即失败。
fn cuda_parity(
    bits: &[f32],
    want: &[f32],
    m: usize,
    n: usize,
    tolerance: f64,
) -> diagnostic::Result<f64> {
    if bits.len() != want.len() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "形状 {m}x{n}：内核只输出 {} 个数，期望 {}",
                bits.len(),
                want.len()
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
