//! 见证执行器（README §11.3）：每条已声明的供给都要真的跑一次自己的见证。
//!
//! 档案校验只能证明"文件在、函数名写在源码里"；本模块把见证源码编成可执行文件，逐条调用
//! `int func(float *out)` 并与 `witness.expect` 对拍。两种范围（[`Scope`]）合起来覆盖每条声明：
//! [`run_all`] 是设备执行，[`self_check`] 只跑设备执行跑不到的、证明"算得对"而不改变 `grade`。
//! 第三种结论是语法门（[`LanguageGate`]）：没有真机时交给 clang 的对应语言前端解析，仍不是设备证据。

pub use crate::prof::Emit;

use crate::atom::{DType, Prim, Provide, Witness};
use crate::cap::Proof;
use crate::device::{self, HOST_FLAGS};
use crate::dirs::Dirs;
use crate::prof::Profile;
use crate::reference;
use diagnostic::{Code, Error};
use std::fmt;
use std::path::{Path, PathBuf};

pub use crate::device::{compile_run_cuda, compile_run_metal};
pub use crate::gate::{LanguageGate, language_gate};

/// 一条待执行的见证：`(原语, 实现名, 声明)`；前后端都在传这三元组。
pub type Selected<'a> = (Prim, String, &'a Witness);

/// 一次见证最多写出多少个值（ABI 上限）；程序生成时把它烘焙进缓冲区尺寸字面量。
pub const MAX_OUT: usize = Witness::MAX_OUT;
/// 跑这台设备的全部见证，得到三条命令共用的证据；失败也是一条要报的事实，所以不返回 `Result`。
#[must_use]
pub fn establish(profile: &Profile, dirs: &Dirs) -> Evidence {
    let executed = device_available(profile);
    let mut failure = None;
    let (device, proof) = device_evidence(profile, dirs, executed, &mut failure);
    let host = host_self_check(profile, dirs, &mut failure);
    let language = language_gate(profile, dirs);
    // 被自己的语言拒绝是证据缺陷，与"本机没有这台设备"不是同一类事。
    if let LanguageGate::Rejected { reason } = &language {
        failure.get_or_insert_with(|| Error::new(Code::Invalid, reason.clone()));
    }
    Evidence {
        basis: if executed {
            Basis::Executed
        } else {
            Basis::Declared
        },
        device,
        host,
        failure,
        device_params: profile.flags.join(" "),
        host_params: HOST_FLAGS.join(" "),
        language,
        proof,
    }
}

/// 设备执行那一半：只有**设备本来可用**时跑不起来才算失败，否则会让 `doctor` 把 GPU 说成"见证写错了"。
fn device_evidence(
    profile: &Profile,
    dirs: &Dirs,
    executed: bool,
    failure: &mut Option<Error>,
) -> (usize, Proof) {
    let runs = match run_all(profile, dirs) {
        Ok(runs) => runs,
        Err(error) => {
            if executed {
                *failure = Some(error);
            }
            return (0, Proof::none());
        }
    };
    let mut proof = Proof::none();
    for run in &runs {
        proof.insert(run.prim, &run.impl_name);
    }
    (runs.len(), proof)
}

/// 本机自检那一半：host C 前端能编能跑，证明源码可编译、`expect` 算得对（不改等级）。
fn host_self_check(profile: &Profile, dirs: &Dirs, failure: &mut Option<Error>) -> usize {
    match self_check(profile, dirs) {
        Ok(runs) => runs.len(),
        Err(error) => {
            failure.get_or_insert(error);
            0
        }
    }
}

/// 一次见证执行能证明什么：范围不同，结论的效力就不同。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// 设备执行：这台机器就是该设备（`c` 是 CPU、`metal` 是本机 GPU）；`L2` 还要内核跑通并对拍。
    Device,
    /// 本机自检：host C 编译并执行，证明源码可编译、`expect` 算得对；**不**证明设备具备该能力。
    Host,
}

/// 一条见证的执行结论。
#[derive(Debug, Clone, PartialEq)]
pub struct WitnessRun {
    /// 原语。
    pub prim: Prim,
    /// 实现名。
    pub impl_name: String,
    /// 入口函数名。
    pub func: String,
    /// 实际写入的元素个数。
    pub values: usize,
    /// 与 `expect` 的相对误差。
    pub max_rel_err: f64,
    /// 这次结论来自设备执行还是本机自检。
    pub scope: Scope,
}

/// 一条供给上的见证三元组；没挂见证返回 `None`。装配口径只此一份，校验与自检两条路径共用。
fn provide_selected(provide: &Provide) -> Option<Selected<'_>> {
    provide
        .witness
        .as_ref()
        .map(|witness| (provide.prim, provide.impl_name.clone(), witness))
}

/// 档案里的全部已声明见证；缺见证的 `grade ≥ L2` 是错误而不是跳过（README §11.3）。
pub(crate) fn declared_witnesses(profile: &Profile) -> diagnostic::Result<Vec<Selected<'_>>> {
    let mut selected = Vec::new();
    for provide in &profile.provides {
        if provide.witness.is_none() && provide.grade.is_verified() {
            return Err(Error::new(
                Code::Missing,
                format!(
                    "`{}`（{}）声明了 {}，但没有可执行的见证：等级必须是跑出来的结论",
                    provide.prim, provide.impl_name, provide.grade
                ),
            ));
        }
        if let Some(entry) = provide_selected(provide) {
            selected.push(entry);
        }
    }
    Ok(selected)
}

/// 设备执行覆盖不到的见证：设备不在本机时全部进自检，在本机时 [`run_all`] 已更严格地跑过，不必重跑。
pub(crate) fn self_checkable(profile: &Profile) -> Vec<Selected<'_>> {
    if device_available(profile) {
        return Vec::new();
    }
    profile
        .provides
        .iter()
        .filter_map(provide_selected)
        .collect()
}

/// 见证源码的根目录：与 `Catalogue` 的装载根保持一致（`data.root` 优先）。
pub(crate) fn witness_root(dirs: &Dirs) -> PathBuf {
    dirs.data().map_or_else(prim::dir, prim::layer_root)
}

/// 见证源码：内置设备读编译期内嵌副本（发布机没有源码树），`data.root` 或未登记的设备走磁盘。
pub(crate) fn source(witness: &Witness, dirs: &Dirs) -> diagnostic::Result<String> {
    let ext = Path::new(&witness.file)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    let vendor = witness.file.split('/').next().unwrap_or("");
    if dirs.data().is_none()
        && let Some(text) = prim::builtin_witness(vendor, ext)
    {
        return Ok(text.to_string());
    }
    let path = witness_root(dirs).join(&witness.file);
    std::fs::read_to_string(&path).map_err(|error| {
        Error::new(
            Code::Io,
            format!("见证 `{}` 不可读：{error}", path.display()),
        )
    })
}

/// 逐条对拍：个数走 stderr、值走 stdout，两路与调用顺序一一对应；两个前端共用编译、计数与对拍口径。
pub(crate) fn compare(
    selected: &[Selected<'_>],
    scope: Scope,
    counts: &[i64],
    bits: &[f32],
) -> diagnostic::Result<Vec<WitnessRun>> {
    check_counts(selected, counts)?;
    let mut cursor = Cursor::new(bits);
    let mut runs = Vec::with_capacity(selected.len());
    for ((prim, impl_name, witness), &count) in selected.iter().zip(counts) {
        let got = cursor.next(*prim, witness, count)?;
        runs.push(WitnessRun {
            prim: *prim,
            impl_name: impl_name.clone(),
            func: witness.func.clone(),
            values: got.len(),
            max_rel_err: checked(*prim, witness, got, reference::tolerance(DType::F32))?,
            scope,
        });
    }
    cursor.finish().map(|()| runs)
}

/// 计数行必须与调用顺序一一对应：少一条就是前端漏调了见证。
fn check_counts(selected: &[Selected<'_>], counts: &[i64]) -> diagnostic::Result<()> {
    let (got, want) = (counts.len(), selected.len());
    if got == want {
        return Ok(());
    }
    Err(Error::new(
        Code::Internal,
        format!("见证执行只回报了 {got} 条计数，期望 {want} 条：前端没有按顺序调用每条见证"),
    ))
}

/// 与 `expect` 对拍：超过阈值就是这条见证不成立（顺带打印两个数组，差在哪一眼可见）。
fn checked(prim: Prim, witness: &Witness, got: &[f32], tolerance: f64) -> diagnostic::Result<f64> {
    let error = reference::max_rel_err(got, &witness.expect);
    if error <= tolerance {
        return Ok(error);
    }
    Err(Error::new(
        Code::Invalid,
        format!(
            "`{prim}` 的见证 `{}` 不成立：得到 {got:?}，期望 {:?}，max_rel_err {error:.3e} > {tolerance:.0e}",
            witness.func, witness.expect
        ),
    ))
}

/// 一维输出流上的读取位置：按计数行切段，每段才能与自己的 `expect` 对齐，两个前端共用这一套切法。
struct Cursor<'a> {
    bits: &'a [f32],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bits: &'a [f32]) -> Self {
        Self { bits, offset: 0 }
    }

    /// 下一条见证的输出段：计数必须等于 `expect` 的长度，值必须真的回来了。
    fn next(&mut self, prim: Prim, witness: &Witness, count: i64) -> diagnostic::Result<&'a [f32]> {
        let expected = witness.expect.len();
        if usize::try_from(count).unwrap_or(usize::MAX) != expected {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{prim}` 的见证 `{}` 写了 {count} 个值，档案声明期望 {expected} 个",
                    witness.func
                ),
            ));
        }
        let got = self.bits.get(self.offset..self.offset + expected);
        let got = got.ok_or_else(|| self.short(prim, witness))?;
        self.offset += expected;
        Ok(got)
    }

    /// 输出必须正好被计数行说完：多出来的值说明计数行与实际打印不一致。
    fn finish(&self) -> diagnostic::Result<()> {
        if self.offset == self.bits.len() {
            return Ok(());
        }
        Err(Error::new(
            Code::Internal,
            format!(
                "见证输出多出 {} 个值：计数行与实际打印不一致",
                self.bits.len() - self.offset
            ),
        ))
    }

    /// 段不够长：前端回报的计数比它真正打印出来的多。
    fn short(&self, prim: Prim, witness: &Witness) -> Error {
        Error::new(
            Code::Internal,
            format!(
                "`{prim}` 的见证 `{}` 只回来了 {} 个值（偏移 {} 处需要 {} 个）",
                witness.func,
                self.bits.len(),
                self.offset,
                witness.expect.len()
            ),
        )
    }
}

/// 一台设备的共享执行流程：生成程序 → 跑 → 逐条对拍；差异全在 [`crate::device::Spec`] 的两行指针里。
pub(crate) fn execute(
    spec: &crate::device::Spec,
    profile: &Profile,
    dirs: &Dirs,
    selected: &[Selected<'_>],
    scope: Scope,
) -> diagnostic::Result<Vec<WitnessRun>> {
    let program = (spec.program)(profile, dirs, selected, scope)?;
    let (counts, bits) = (spec.run)(profile, dirs, &program, selected.len(), scope)?;
    compare(selected, scope, &counts, &bits)
}

/// 在本机执行档案里**每一条**已声明的见证（`provides` 通常共用一份源码，整份只编译一次）；不按等级
/// 筛：`grade` 是执行的结论，先筛会让不敢写 `L2` 的设备永远拿不到运行结论。
///
/// # Errors
///
/// 执行不了这台设备、缺编译器或驱动、文件不可读、函数不存在、编译/执行失败、返回值个数不符或
/// 对拍超出容差时，返回带原语与函数名的说明。
pub fn run_all(profile: &Profile, dirs: &Dirs) -> diagnostic::Result<Vec<WitnessRun>> {
    let selected = declared_witnesses(profile)?;
    if selected.is_empty() {
        return Ok(Vec::new());
    }
    let spec = device::spec(profile.meta.emit).ok_or_else(|| {
        let known: Vec<String> = device::specs()
            .iter()
            .map(|entry| entry.emit.to_string())
            .collect();
        Error::new(
            Code::Unsupported,
            format!(
                "本机没有该设备（`{}` 的 emit 是 {}）：本机接了的执行前端是 {}，\
                 可先跑 `self_check` 的 host 自检，但自检不构成设备证据",
                profile.meta.vendor,
                profile.meta.emit,
                known.join(" / ")
            ),
        )
    })?;
    execute(spec, profile, dirs, &selected, Scope::Device)
}

/// 这台设备是否就在本机：`c` 的设备是 CPU 永远在，`metal` 看工具链；不在表里的 `emit` 没有执行前端，一律当作不在。
#[must_use]
pub fn device_available(profile: &Profile) -> bool {
    device::spec(profile.meta.emit).is_some_and(|spec| (spec.available)(profile))
}

/// 对**设备执行跑不到**的见证做本机自检，范围与 [`run_all`] 互补（设备在本机时由更强的那个范围给结论）。
/// 见证的 ABI 是 host C，与片段语言无关：跳过档案的 `preamble` 与 `flags`，MSL 的 `thread` 前置一行
/// `#define` 只为自检。自检通过**不**说明设备具备能力，因此不改写任何 `grade`。
///
/// # Errors
///
/// 缺 host C 编译器、文件不可读、函数不存在、编译/执行失败、返回值个数不符或对拍超出容差时，
/// 返回带设备、原语与函数名的说明。
pub fn self_check(profile: &Profile, dirs: &Dirs) -> diagnostic::Result<Vec<WitnessRun>> {
    let selected = self_checkable(profile);
    if selected.is_empty() {
        return Ok(Vec::new());
    }
    let c_spec = device::spec(Emit::C)
        .ok_or_else(|| Error::new(Code::Internal, "C 自检前端没有登记到设备描述表"))?;
    execute(c_spec, profile, dirs, &selected, Scope::Host).map_err(|error| {
        error.with(format!(
            "`{}({})` 的见证自检不成立",
            profile.meta.vendor, profile.meta.id
        ))
    })
}
// 证据的形状：字段分三组，报告必须分开写（执行 / 依据 / 不成立的那一步）；合成一个 `bool` 会让
// "设备不在本机"与"见证写错了"长得一样。

/// 这次协商的依据：`Executed` 是"这台机器刚刚跑通了见证"，`Declared` 是"档案这么写，本机
/// 执行不了这台设备"。报告必须写出是哪一种，否则读者会把自述当成证据。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// 能力来自这次执行：这台机器就是该设备（`c` 就是 CPU，`metal` 就是本机 GPU）。
    Executed,
    /// 本机不能执行这台设备：协商只能按档案自述。
    Declared,
}

impl fmt::Display for Basis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Executed => "这次执行",
            Self::Declared => "档案自述（本机执行不了这台设备）",
        })
    }
}

/// 一台设备这一次的证据：跑通了哪些见证、依据是什么、哪一步不成立；三条命令共用它，否则同一台设备能同时"协商可行"与"见证不成立"。
#[derive(Debug, Clone, PartialEq)]
pub struct Evidence {
    /// 协商的依据。
    pub basis: Basis,
    /// 设备执行通过的见证条数（这台机器就是该设备，结论能归到具体供给）。
    pub device: usize,
    /// 本机自检通过的见证条数（设备执行跑不到的那些）。
    pub host: usize,
    /// 这次没能确立的证据：`None` 表示每条已声明的见证都有执行结论；本机没有这台设备不算失败（见 [`Evidence::basis`]）。
    pub failure: Option<Error>,
    /// 设备执行的编译参数：参数一变产物身份就变（[`crate::cache::ArtifactKey`]），报告写出它，读者才知道证据适用于哪一次代码生成。
    pub device_params: String,
    /// 本机自检的编译参数（`HOST_FLAGS`）：与设备执行不是同一套，必须分开报。
    pub host_params: String,
    /// 见证在**它自己声称的语言**里的合法性：不是设备证据（没有一行在设备上执行），所以不改 `grade`，但被拒绝是硬错误。
    pub language: LanguageGate,
    /// 真的被证明的供给（执行结论）。
    pub(crate) proof: Proof,
}

impl Evidence {
    /// 协商该用的证据：`Executed` 时是执行结论，`Declared` 时退回档案自述——本机执行不了
    /// 这台设备，没有第二条路，所以依据必须由 [`Evidence::basis`] 说出来。
    #[must_use]
    pub fn proof(&self, profile: &Profile) -> Proof {
        match self.basis {
            Basis::Executed => self.proof.clone(),
            Basis::Declared => Proof::declared(profile),
        }
    }
}

#[cfg(test)]
#[path = "../tests/witness_tests.rs"]
mod tests;
