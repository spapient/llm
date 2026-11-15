//! 语法门：设备不在本机时，用一门语言自己的解析器把见证过一遍。
//!
//! 它不是执行，也不改任何 `grade`：只证明"这段见证在它自己声称的语言里语法成立"。缺了它会有
//! 具体的假绿——`metal/witness.metal` 曾写成 `float *out`，host `cc` 能过，而 MSL 没有默认
//! 地址空间，于是那份证据从未在自己声称的语言里存在过。

use crate::atom::Witness;
use crate::device::scratch;
use crate::dirs::Dirs;
use crate::prof::{Emit, Profile};
use crate::toolchain::clang;
use crate::witness::{Selected, declared_witnesses, device_available, source};
use diagnostic::{Code, Error};
use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;

/// 见证在**它自己声称的语言**里的合法性。
///
/// 这是第三种结论：设备执行（[`crate::witness::Scope::Device`]）说"这台机器跑得动"，本机自检（[`crate::witness::Scope::Host`]）
/// 说"这段代码在 host C 里算得对"，语法门说"这份文件在它声称的语言里编得过"。中间的缝隙不是
/// 理论问题：`prim/metal/witness.metal` 曾长期只有 host `cc` 看过，而 MSL 根本接受不了
/// `float *out`（没有默认地址空间）。
///
/// 门不执行任何一行，所以**不改变 `grade`**；但被拒绝是证据本身的缺陷，[`crate::witness::establish`] 把它记进
/// [`crate::witness::Evidence::failure`]，与"本机编不了"分得清清楚楚。
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
        /// 为什么开不了（缺哪个程序、或这门语言没有开源前端）。
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

/// 一门语言的语法前端：与执行层 [`crate::device::Spec`] 同构，差异全是表内常量行——解析器程序与
/// 固定语法检查参数；没有独立的前端类型，也没有按语言单态化的泛型函数。
struct GateSpec {
    /// 这门语言。
    emit: Emit,
    /// 解析器：程序名 + 固定前缀参数（`["clang"]`）。
    tool: &'static [&'static str],
    /// 只做语法检查的参数；源文件路径由执行器追加在末尾。
    check: &'static [&'static str],
}

/// 全部语法前端：**加一门语言就加一行**，与 `const SPECS` 同一套登记方式。
const GATES: &[GateSpec] = &[
    GateSpec {
        emit: Emit::Cuda,
        tool: &["clang"],
        check: &[
            "-x",
            "cuda",
            "--cuda-host-only",
            "-nocudainc",
            "-nocudalib",
            "-fsyntax-only",
        ],
    },
    GateSpec {
        emit: Emit::Opencl,
        tool: &["clang"],
        check: &["-x", "cl", "-cl-std=CL3.0", "-fsyntax-only"],
    },
];

/// 用一台设备**自己声称的语言**把它的全部已声明见证过一遍。
///
/// 设备执行跑得到的设备（`c` / `metal`）返回 [`LanguageGate::Covered`]：门是自检的替代品，
/// 能真跑的就不再只解析。其余设备按 `emit` 找语言前端；找不到就说找不到，不降级成别的东西。
#[must_use]
pub fn language_gate(profile: &Profile, dirs: &Dirs) -> LanguageGate {
    if device_available(profile) {
        return LanguageGate::Covered;
    }
    let Some(entry) = GATES.iter().find(|entry| entry.emit == profile.meta.emit) else {
        let known: Vec<String> = GATES.iter().map(|entry| entry.emit.to_string()).collect();
        return LanguageGate::Absent {
            reason: format!(
                "本机没有 `{}` 的语法前端（接了 {}）",
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
///
/// 源码一律经 [`source`] 取：内嵌副本与 `data.root` 覆盖都由它决定。先前这里直接把
/// `witness_root(dirs).join(file)` 交给 clang，于是在没有源码树的发布机上会报成
/// “见证被自己的语言拒绝”——真相是“本机没有那份源码”，分类与指向的人都错了。
fn parse_all(spec: &GateSpec, profile: &Profile, dirs: &Dirs) -> diagnostic::Result<LanguageGate> {
    let selected = declared_witnesses(profile)?;
    if selected.is_empty() {
        return Ok(LanguageGate::Absent {
            reason: "没有已声明的见证：门没有对象可开".to_string(),
        });
    }
    let Some(toolchain) = clang() else {
        return Ok(LanguageGate::Absent {
            reason: format!("`{}` 不在本机", spec.tool.join(" ")),
        });
    };
    let dir = scratch(dirs, "gate");
    let parsed = write_and_parse(spec, &selected, dirs, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    parsed?;
    Ok(LanguageGate::Passed {
        front: format!("{} {}", toolchain.id(), spec.check.join(" ")),
        witnesses: selected.len(),
    })
}

/// 把每份见证按“内嵌优先”的文本落到临时目录，再交给解析器：同一份源码可以没有磁盘副本。
fn write_and_parse(
    spec: &GateSpec,
    selected: &[Selected<'_>],
    dirs: &Dirs,
    dir: &Path,
) -> diagnostic::Result<()> {
    let mut written: BTreeSet<&str> = BTreeSet::new();
    for (_, _, witness) in selected {
        if written.insert(witness.file.as_str()) {
            write_one(spec, witness, dirs, dir)?;
        }
    }
    Ok(())
}

/// 一份见证：按“内嵌优先”的文本落盘，再交给它自己语言的解析器。
fn write_one(
    spec: &GateSpec,
    witness: &Witness,
    dirs: &Dirs,
    dir: &Path,
) -> diagnostic::Result<()> {
    let path = dir.join(&witness.file);
    write_unit(&path, &source(witness, dirs)?)?;
    parse_one(spec, &path)
}

/// 写一份翻译单元：`witness.file` 形如 `<vendor>/<name>.<ext>`，父目录要自己建。
fn write_unit(path: &Path, text: &str) -> diagnostic::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            Error::new(Code::Io, format!("创建 {} 失败：{error}", parent.display()))
        })?;
    }
    std::fs::write(path, text)
        .map_err(|error| Error::new(Code::Io, format!("写 {} 失败：{error}", path.display())))
}

/// 解析一份翻译单元；失败时只带编译器的第一行报错（整段 stderr 进不了报告）。
fn parse_one(spec: &GateSpec, path: &Path) -> diagnostic::Result<()> {
    let program = spec.tool[0];
    let mut command = std::process::Command::new(program);
    command.args(&spec.tool[1..]).args(spec.check).arg(path);
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
            spec.check.join(" "),
            path.display()
        ),
    ))
}

#[cfg(test)]
#[path = "gate_tests.rs"]
mod tests;
