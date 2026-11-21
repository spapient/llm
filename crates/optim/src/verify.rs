//! `verify`：逐设备把三件事钉在一起——见证跑通、协商选中首档、内核与标量参考对拍（README §11）。
//!
//! 形状矩阵固定在 `pipeline`：生产形状、小形状、非整除形状，后者专门抓"按 16 对齐成立、按 3 不成立"
//! 这类错误。失败一律是错误而不是跳过：设备不在本机是 `unsupported`（退出码 3），声明或输入写错是
//! `invalid`（退出码 2）。两条执行通道（本机 C、nvcc/CUDA）共用同一套步骤，设备差异收在 `Lane` 里。

use crate::build;
use crate::gate::LanguageGate;
use crate::pipeline::{Prepared, Run, parse_shape, rejected_shapes, verify_shapes};
use crate::witness;
use crate::{runner, toolchain};
use cutile::builtin::Catalogue;
use cutile::cache::Artifact;
use cutile::cap::{self, NoPath, Path, Proof};
use cutile::dirs::Dirs;
use cutile::emit::Fragment;
use cutile::prof::{Emit, Profile};
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
}

/// 设备通道跑的是哪一份程序。
///
/// 三种都是「算对了」，但证据强度不同：并行网格程序证的是生产态形态（每线程一元素 / 每块一线程 /
/// `[grid]` 各维一线程），回落单线程只证语义，C 参考通道则是串行语义的定义本身。台账混记会让
/// 「该形态缺并行实现」这条缺口永久隐身，因此它是报告字段，不是日志噪声。
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

mod sweep;

#[cfg(test)]
pub(crate) use sweep::Lane;
pub use sweep::verify_one;

#[cfg(test)]
#[path = "../tests/verify_tests.rs"]
mod tests;
