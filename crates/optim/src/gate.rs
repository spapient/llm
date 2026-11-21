//! 语法门：设备不在本机时，用一门语言自己的解析器把见证过一遍。

use crate::runner::driver::{scratch, write_text};
use crate::toolchain::{Toolchain, clang, probe_declared};
use crate::witness::{Selected, declared_witnesses, device_available, source};
use cutile::atom::Witness;
use cutile::dirs::Dirs;
use cutile::prof::{Emit, Profile};
use diagnostic::{Code, Error};
use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;

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

/// 一门语言的语法前端：与执行层 [`crate::device::Spec`] 同构，差异全是表内常量行——解析器程序与
/// 固定语法检查参数；没有独立的前端类型，也没有按语言单态化的泛型函数。
struct GateSpec {
    /// 这门语言。
    emit: Emit,
    /// 解析器：程序名 + 固定前缀参数（`["clang"]`）。`["clang"]` 指本机的 clang 前端；其余名字按档案
    /// 里的写法原样探测——`ccec` 随 CANN 安装，宿主 `cc` 不认识昇腾的设备语言开关。
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
        // 语言版本不在这里写死：它是档案的事（`compile.flags`），见 [`GateSpec::check`]。
        check: &["-x", "cl", "-fsyntax-only"],
    },
    GateSpec {
        emit: Emit::C,
        tool: &["clang"],
        check: &["-x", "c", "-fsyntax-only"],
    },
    GateSpec {
        emit: Emit::Metal,
        // `metal` 只在 Xcode 工具链里，经 `xcrun` 找：`tool` 多词写法由 [`probe_declared`] 拆开。
        tool: &["xcrun", "-sdk", "macosx", "metal"],
        check: &["-fsyntax-only"],
    },
    GateSpec {
        emit: Emit::Ascendc,
        // `-xcce` 让 ccec 按昇腾设备语言解析这份翻译单元（不带它，ccec 只当普通宿主 C++ 编，
        // 见证里的 `__aicore__` 会被无声忽略）；`dav-c220` 是 `id = "910b"` 那一档的核芯名。
        // 认证别的核芯时这一行要按 algorithm 拆开登记，不能沿用 910B 的名字。
        tool: &["ccec"],
        check: &["-xcce", "--cce-aicore-algorithm=dav-c220", "-fsyntax-only"],
    },
];

impl GateSpec {
    /// 有效检查参数：表内固定的语法检查参数 + 档案自己声明的语言版本开关。
    fn check<'a>(&'a self, profile: &'a Profile) -> Vec<&'a str> {
        let mut arguments: Vec<&str> = self.check.to_vec();
        arguments.extend(
            profile
                .flags
                .iter()
                .map(String::as_str)
                .filter(|flag| flag.starts_with("-cl-std=") || flag.starts_with("-std=")),
        );
        arguments
    }
}

/// 表行声明的解析器：`clang` 行指「本机的 clang 前端」（先问 `clang`，再退到名字里带 clang 的
/// `cc`）；其余按声明原样探测，不退回宿主编译器——拿 `cc` 去解析 AscendC 只会把「本机没有 CANN」
/// 报成一条 C 语法错误。
fn front(spec: &GateSpec) -> Option<Toolchain> {
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
    let Some(entry) = GATES.iter().find(|entry| entry.emit == profile.meta.emit) else {
        let known: Vec<String> = GATES.iter().map(|entry| entry.emit.to_string()).collect();
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
fn parse_all(spec: &GateSpec, profile: &Profile, dirs: &Dirs) -> diagnostic::Result<LanguageGate> {
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
    let dir = scratch(dirs, "gate");
    let arguments = spec.check(profile);
    let parsed = write_and_parse(spec, &arguments, &selected, dirs, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    parsed?;
    Ok(LanguageGate::Passed {
        front: format!("{} {}", toolchain.id(), arguments.join(" ")),
        witnesses: selected.len(),
    })
}

/// 把每份见证按“内嵌优先”的文本落到临时目录，再交给解析器：同一份源码可以没有磁盘副本。
fn write_and_parse(
    spec: &GateSpec,
    arguments: &[&str],
    selected: &[Selected<'_>],
    dirs: &Dirs,
    dir: &Path,
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
    spec: &GateSpec,
    arguments: &[&str],
    witness: &Witness,
    dirs: &Dirs,
    dir: &Path,
) -> diagnostic::Result<()> {
    let path = dir.join(&witness.file);
    write_text(&path, &source(witness, dirs)?)?;
    parse_one(spec, arguments, &path)
}

/// 解析一份翻译单元；失败时只带编译器的第一行报错（整段 stderr 进不了报告）。
fn parse_one(spec: &GateSpec, arguments: &[&str], path: &Path) -> diagnostic::Result<()> {
    let program = spec.tool[0];
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
#[path = "../tests/gate_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/fragments_tests.rs"]
mod fragments_tests;
