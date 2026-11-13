//! 语法门：设备不在本机时，用一门语言自己的解析器把见证过一遍。

use crate::device::clang;
use crate::prof::{Emit, Profile};
use crate::witness::{Front, Language, declared_witnesses, device_available, witness_root};
use diagnostic::{Code, Error};
use llm_config::Settings;
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

/// 本机接了的语法前端：与 `FRONTENDS` 同构，存的也是单态化后的函数地址。
struct GateEntry {
    /// 这门语言。
    emit: Emit,
    /// 用这门语言的解析器过一遍全部已声明的见证。
    parse: fn(&Profile, &Settings) -> diagnostic::Result<LanguageGate>,
}

/// 全部语法前端：**加一门语言就加一行**。
const GATES: &[GateEntry] = &[
    GateEntry {
        emit: <CudaFront as Language>::EMIT,
        parse: parse_all::<CudaFront>,
    },
    GateEntry {
        emit: <OpenClFront as Language>::EMIT,
        parse: parse_all::<OpenClFront>,
    },
];

/// clang 的 CUDA 前端：`--cuda-host-only` 只解析 host 那一半——见证的 ABI 就是 host C，
/// 而 `__syncthreads()` 这类设备内建在 host 函数里必须被拒。
#[annotation::frontend(emit = "cuda", ext = "cu", tag = "gate")]
pub struct CudaFront;

impl Front for CudaFront {
    const TOOL: &'static [&'static str] = &["clang"];
    const CHECK: &'static [&'static str] = &[
        "-x",
        "cuda",
        "--cuda-host-only",
        "-nocudainc",
        "-nocudalib",
        "-fsyntax-only",
    ];
}

/// clang 的 OpenCL 前端：`double` 在没开 `cl_khr_fp64` 的 OpenCL C 里是错误，host `cc` 却照收。
#[annotation::frontend(emit = "opencl", ext = "cl", tag = "gate")]
pub struct OpenClFront;

impl Front for OpenClFront {
    const TOOL: &'static [&'static str] = &["clang"];
    const CHECK: &'static [&'static str] = &["-x", "cl", "-cl-std=CL3.0", "-fsyntax-only"];
}

/// 用一台设备**自己声称的语言**把它的全部已声明见证过一遍。
///
/// 设备执行跑得到的设备（`c` / `metal`）返回 [`LanguageGate::Covered`]：门是自检的替代品，
/// 能真跑的就不再只解析。其余设备按 `emit` 找语言前端；找不到就说找不到，不降级成别的东西。
#[must_use]
pub fn language_gate(profile: &Profile, settings: &Settings) -> LanguageGate {
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
    match (entry.parse)(profile, settings) {
        Ok(gate) => gate,
        // 解析器拒绝是证据的缺陷，不能与“本机没东西可用”归一类。
        Err(error) => LanguageGate::Rejected {
            reason: error.to_string(),
        },
    }
}

/// 语法门的执行器：`F` 决定程序与参数，这里只负责“逐份文件过一遍、失败带第一行报错回来”。
fn parse_all<F: Front>(profile: &Profile, settings: &Settings) -> diagnostic::Result<LanguageGate> {
    let selected = declared_witnesses(profile)?;
    if selected.is_empty() {
        return Ok(LanguageGate::Absent {
            reason: "没有已声明的见证：门没有对象可开".to_string(),
        });
    }
    let Some(toolchain) = clang() else {
        return Ok(LanguageGate::Absent {
            reason: format!("`{}` 不在本机", F::TOOL.join(" ")),
        });
    };
    let mut files: BTreeSet<&str> = BTreeSet::new();
    for (_, _, witness) in &selected {
        files.insert(witness.file.as_str());
    }
    for file in files {
        parse_one::<F>(&witness_root(settings).join(file))?;
    }
    Ok(LanguageGate::Passed {
        front: format!("{} {}", toolchain.id(), F::CHECK.join(" ")),
        witnesses: selected.len(),
    })
}

/// 解析一份翻译单元；失败时只带编译器的第一行报错（整段 stderr 进不了报告）。
pub(crate) fn parse_one<F: Front>(path: &Path) -> diagnostic::Result<()> {
    let program = F::TOOL[0];
    let mut command = std::process::Command::new(program);
    command.args(&F::TOOL[1..]).args(F::CHECK).arg(path);
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
            F::TOOL.join(" "),
            F::CHECK.join(" "),
            path.display()
        ),
    ))
}
