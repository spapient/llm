//! 见证执行器（DESIGN §11.3）：每条已声明的供给都要真的跑一次自己的见证。
//!
//! 档案校验只能证明"文件在、函数名写在源码里"；本模块把见证源码编成可执行文件，逐条调用
//! `int func(float *out)` 并与 `witness.expect` 对拍。没有跑通的提供不会通过 `verify`，
//! 因此 `grade` 不再是作者的自述：它是这次执行的结论。
//!
//! 执行分两种范围（[`Scope`]），两者不重叠而合起来覆盖每一条声明：
//! [`run_all`] 是**设备执行**，只要这台设备在本机（`c` 永远在，`metal` 看工具链）就把档案里
//! 每一条声明的见证跑一遍；[`self_check`] 是**本机自检**，只跑设备执行跑不到的那些——
//! 它证明"见证写了什么、算得对不对"，不证明设备具备该能力，所以不改变任何 `grade`。
//!
//! 选择不按等级：`grade ≥ L2` 才执行，等于让没敢写 `L2` 的设备永远拿不到运行结论——
//! "自证不具备"。等级是执行的**结论**，不是执行的前提。`L2` 仍然要内核在本机跑通并对拍
//! （[`crate::verify_one`]），见证执行只完成其中的见证那一段。
//!
//! 前端按 `emit` 选：`c` 走 `cc`（ABI 就是 host C）；`metal` 走 `xcrun metal` → `metallib`，
//! 再由一个 host 驱动把见证发射到 GPU、回读。后者不是锦上添花：MSL 没有默认地址空间，
//! `float *out` 在 `.metal` 里根本编不过，而 host `cc` 只因为它是纯 C 才放行——没有真前端，
//! "见证"就只是写在数据里的一句话。
//!
//! 第三种结论是**语法门**（[`LanguageGate`]）：设备不在本机时，host 自检只能说明"这段代码在 C
//! 里成立"，而同一个 Metal 事故说明这句话不够——那份见证从未在它声称的语言里编过一次。于是
//! `cuda` / `opencl` 另行交给 clang 的对应**语言**前端解析一遍（不需要真机、不需要驱动），
//! `ascendc` 没有开源前端就如实缺席。语法门不执行任何一行，仍然不是设备证据。
//!
//! # 前端 = 注解 + trait + const 泛型
//!
//! 一台设备的**语言身份**（`Emit`、产物后缀、缓存标签）由 `#[frontend(..)]` 在编译期生成
//! （[`Language`]），**怎么跑**由手写的 [`Device`] 实现给出，两者由 [`Frontend`] 的超 trait
//! 约束绑死：只标注解不写执行、或写了执行没标注解，都拿不到 [`Frontend`]。
//!
//! 分发是一张 `const` 表 `FRONTENDS`：每一项都是单态化后的具体函数，没有 `match`、没有
//! `dyn`。新增一台设备 = 标一个类型 + 在表里加一行；本模块里没有任何需要同步修改的分支。
//! 每条见证的写出上限 [`MAX_OUT`] 以 const 泛型参数传进前端，生成出来的程序里缓冲区大小
//! 是字面量，执行期不存在"上限是多少"这个变量。

pub use cutile::prof::Emit;

use cutile::atom::{DType, Prim, Witness};
use cutile::backend;
use cutile::build;
use cutile::cap::Proof;
use cutile::prof::Profile;
use cutile::reference;
use diagnostic::{Code, Error};
use llm_config::Settings;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

/// 一条待执行的见证：`(原语, 实现名, 声明)`。
///
/// 前后端与本模块内部都在传这三元组，起个别名，免得每处各写一遍元组类型。
pub type Selected<'a> = (Prim, String, &'a Witness);

/// 一次见证最多写出多少个值（ABI 上限，DESIGN §11.3）。
///
/// 它是**编译期**常量，并以 const 泛型参数的形式传进两个前端：生成的程序里缓冲区大小是
/// 字面量，执行期不再有"上限是多少"这个变量，也没有按上限做的运行期分支。
pub const MAX_OUT: usize = Witness::MAX_OUT;

/// 注解生成的**语言身份**。
///
/// 三个关联常量都来自 `#[frontend(..)]`，因此编译器知道它们的具体取值；`EXT` 与
/// `EMIT.ext()` 的一致性由注解展开出的 `const _` 断言在编译期检查，写错即编译失败。
/// 这里不写任何执行逻辑：身份可以完全由注解生成，"怎么跑"不行。
pub trait Language {
    /// 执行语言（[`Emit`] 闭集里的一个）。
    const EMIT: Emit;
    /// 见证与片段的文件后缀，必须等于 `EMIT.ext()`。
    const EXT: &'static str;
    /// 产物缓存的标签。
    const TAG: &'static str;
}

/// 手写的执行能力：本机有没有这台设备、怎么把见证编起来跑。
///
/// 与 [`Language`] 分开是因为两者的来源不同：语言身份可以完全由注解生成，而"怎么跑"要调
/// 编译器、要写驱动，注解生成不了。[`Frontend`] 要求两者同时成立，于是注解与执行不会脱节。
pub trait Device: Sized {
    /// 这台设备此刻是否就在本机。
    fn available(profile: &Profile) -> bool;

    /// 生成"跑一次全部见证"的程序文本；`OUT` 是每条见证的写出上限。
    fn program<const OUT: usize>(
        profile: &Profile,
        settings: &Settings,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> diagnostic::Result<String>;

    /// 让程序跑起来，返回 `(每条写入个数, 按顺序回读的值)`。
    fn run(
        profile: &Profile,
        settings: &Settings,
        program: &str,
        count: usize,
        scope: Scope,
    ) -> diagnostic::Result<(Vec<i64>, Vec<f32>)>;
}

/// 执行前端的完整契约：注解给出的语言 + 手写给出的执行。
///
/// 空实现体是刻意的：它不提供任何默认方法，只把两条 bound 合成一条，
/// 于是"泛型入口要求 `Frontend`"同时意味着"注解与执行都在"。
pub trait Frontend: Language + Device {}
impl<T: Language + Device> Frontend for T {}

/// 一种语言的**语法前端**：把一份翻译单元交给这门语言自己的解析器，只解析、不链接、不执行。
///
/// 它与 [`Device`] 是同一份契约的两半：身份（[`Language`]）由 `#[frontend(..)]` 生成，"怎么解析"
/// 手写。`cuda` / `opencl` 借的是 clang 的这两个**语言**前端，不是某一台设备的前端，所以不需要
/// 真机；`ascendc` 没有开源前端，`GATES` 里因此没有它。
pub trait Front: Language {
    /// 解析器：程序名 + 固定前缀参数（`["clang"]`）。
    const TOOL: &'static [&'static str];
    /// 只做语法检查的参数；源文件路径由执行器追加在末尾。
    const CHECK: &'static [&'static str];
}

/// 见证在**它自己声称的语言**里的合法性。
///
/// 这是第三种结论：设备执行（[`Scope::Device`]）说"这台机器跑得动"，本机自检（[`Scope::Host`]）
/// 说"这段代码在 host C 里算得对"，语法门说"这份文件在它声称的语言里编得过"。中间的缝隙不是
/// 理论问题：`prim/metal/witness.metal` 曾长期只有 host `cc` 看过，而 MSL 根本接受不了
/// `float *out`（没有默认地址空间）。
///
/// 门不执行任何一行，所以**不改变 `grade`**；但被拒绝是证据本身的缺陷，[`establish`] 把它记进
/// [`Evidence::failure`]，与"本机编不了"分得清清楚楚。
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

/// 本机接了的执行前端。
///
/// 每项都是单态化的具体函数：`drive::<CBackend>` 与 `drive::<MetalBackend>` 是两个不同的
/// 实例，表里存的是它们的地址。运行期只多一次"哪一项"的比较，没有枚举匹配、没有虚表。
struct Entry {
    /// 这门语言。
    emit: Emit,
    /// 设备是否在本机。
    available: fn(&Profile) -> bool,
    /// 在本机执行全部已声明的见证。
    run: fn(&Profile, &Settings, &[Selected<'_>]) -> diagnostic::Result<Vec<WitnessRun>>,
}

/// 全部前端：**加设备就加一行**，不需要改任何 `match`。
const FRONTENDS: &[Entry] = &[
    Entry {
        emit: <CBackend as Language>::EMIT,
        available: <CBackend as Device>::available,
        run: drive::<CBackend>,
    },
    Entry {
        emit: <MetalBackend as Language>::EMIT,
        available: <MetalBackend as Device>::available,
        run: drive::<MetalBackend>,
    },
];

/// 按 `emit` 找前端；找不到就是"本机没有这门语言的执行前端"。
fn frontend(emit: Emit) -> Option<&'static Entry> {
    FRONTENDS.iter().find(|entry| entry.emit == emit)
}

/// 一个前端的设备执行入口：把 `OUT` 定成 ABI 上限后交给共享的 [`execute`]。
fn drive<F: Frontend>(
    profile: &Profile,
    settings: &Settings,
    selected: &[Selected<'_>],
) -> diagnostic::Result<Vec<WitnessRun>> {
    execute::<F, MAX_OUT>(profile, settings, selected, Scope::Device)
}

/// 两个 `&str` 在编译期是否逐字节相同。
///
/// `str` 的 `==` 不是 `const`，而注解里的 `ext` 必须与 `Emit::ext()` 在编译期对上——
/// 这条判据就是注解展开出的那两个 `const _` 断言用的比较。
#[doc(hidden)]
#[must_use]
pub const fn same_text(left: &str, right: &str) -> bool {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

/// 一次见证执行能证明什么：范围不同，结论的效力就不同。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// 设备执行：这台机器就是该设备（`c` 的 CPU、`metal` 的本机 GPU），通过即拿到
    /// `Scope::Device` 结论；`L2` 还要内核跑通并对拍，见 [`crate::verify_one`]。
    Device,
    /// 本机自检：用 host C 前端编译并执行见证函数，证明源码可编译、`expect` 算得对；
    /// **不**证明设备具备这条能力，所以不改变等级。
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

/// 在本机执行档案里**每一条**已声明的见证。
///
/// 一台设备的 `provides` 通常共用一份见证源码，因此整份文件只编译一次、逐函数调用。
/// 选哪些不按等级看：`grade` 是执行的结论，先按等级筛会让不敢写 `L2` 的设备永远拿不到
/// 运行结论。设备不在这台机器上（非 `c` 且工具链/设备缺失）才是拒绝执行的理由。
///
/// # Errors
///
/// 本机执行不了这台设备、缺编译器或驱动、见证文件不可读、函数不存在、编译/执行失败、
/// 返回值个数不符或对拍超出容差时，返回带原语与函数名的说明。
pub fn run_all(profile: &Profile, settings: &Settings) -> diagnostic::Result<Vec<WitnessRun>> {
    let selected = declared_witnesses(profile)?;
    if selected.is_empty() {
        return Ok(Vec::new());
    }
    let entry = frontend(profile.meta.emit).ok_or_else(|| {
        let known: Vec<String> = FRONTENDS
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
    (entry.run)(profile, settings, &selected)
}

/// 这台设备是否就在本机（见证可以在它上面真跑）。
///
/// `c` 的设备就是 CPU，永远在；`metal` 看 Metal 工具链是否可用（可用就证明这台机器有 GPU
/// 和编译器）——本机有没有设备是**事实**，不能用一个配置字段代替。不在表里的 `emit` 没有
/// 执行前端，一律当作不在：这一句就是"未接前端"与"接前端但本机没设备"的分界。
#[must_use]
pub fn device_available(profile: &Profile) -> bool {
    frontend(profile.meta.emit).is_some_and(|entry| (entry.available)(profile))
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
fn parse_one<F: Front>(path: &Path) -> diagnostic::Result<()> {
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

/// 本机的 clang：先问 `clang`，再退到 `cc`（macOS 与多数发行版的 `cc` 就是 clang）。
///
/// 判据是“这个程序是不是 clang”，不是“有没有叫 clang 的文件”：`-x cuda` / `-x cl` 是
/// clang 的前端，gcc 没有。
fn clang() -> Option<backend::Toolchain> {
    backend::probe("clang")
        .or_else(|| backend::detect().filter(|toolchain| toolchain.banner.contains("clang")))
}

/// `xcrun metal` 的实际工具链：命令可用、能自报版本时才算数。
///
/// 不用 `[meta].tool` 判断：那是人写的登记值（`xcrun metal`），而这里要回答的是
/// "这台机器现在能不能把 `.metal` 编成 `metallib`"。
fn metal_toolchain() -> Option<backend::Toolchain> {
    let output = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).to_string()
        + &String::from_utf8_lossy(&output.stderr);
    Some(backend::Toolchain {
        command: "xcrun metal".to_string(),
        version: backend::parse_version(&text),
        banner: text
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())?
            .to_string(),
    })
}

/// 对**设备执行跑不到**的见证做本机自检。
///
/// 范围与 [`run_all`] 互补，合起来刚好覆盖每一条已声明的见证：设备在本机时
/// [`run_all`] 已经跑过全部声明，同一份见证不再重复自检——结论由能给出更强结论的那个
/// 范围给出；设备不在本机时全部声明都在自检范围内。
///
/// 见证的 ABI 是 host C（`int func(float *)`），与片段语言无关：这里跳过档案的
/// `[compile].preamble` 与 `flags`（`#include <cuda_fp16.h>`、`-arch=sm_100a` 这类只对
/// 真机编译器有意义），用 host 编译器把见证源码与 harness 编成可执行文件跑一遍。
/// 非 C 源的地址空间限定符（MSL 的 `thread`）在 host C 里没有对应物，自检前置一行
/// `#define thread`；这只影响自检，不影响设备前端（那里走真正的 `xcrun metal`）。
///
/// 结论的边界写在 [`Scope::Host`] 里：自检通过只说明"见证源码能编译、`expect` 算得对"，
/// **不**说明设备具备这条能力。因此自检不改写任何 `grade`——`L2` 仍要设备执行（[`run_all`]）。
///
/// # Errors
///
/// 缺 host C 编译器、见证文件不可读、函数不存在、编译/执行失败、返回值个数不符或对拍
/// 超出容差时，返回带设备、原语与函数名的说明。
pub fn self_check(profile: &Profile, settings: &Settings) -> diagnostic::Result<Vec<WitnessRun>> {
    let selected = self_checkable(profile);
    if selected.is_empty() {
        return Ok(Vec::new());
    }
    execute::<CBackend, MAX_OUT>(profile, settings, &selected, Scope::Host).map_err(|error| {
        error.with(format!(
            "`{}({})` 的见证自检不成立",
            profile.meta.vendor, profile.meta.id
        ))
    })
}

/// 档案里的全部已声明见证；缺见证的 `grade ≥ L2` 是错误而不是跳过（DESIGN §11.3）。
fn declared_witnesses(profile: &Profile) -> diagnostic::Result<Vec<Selected<'_>>> {
    let mut selected = Vec::new();
    for provide in &profile.provides {
        let Some(witness) = provide.witness.as_ref() else {
            if provide.grade.is_verified() {
                return Err(Error::new(
                    Code::Missing,
                    format!(
                        "`{}`（{}）声明了 {}，但没有可执行的见证：等级必须是跑出来的结论",
                        provide.prim, provide.impl_name, provide.grade
                    ),
                ));
            }
            continue;
        };
        selected.push((provide.prim, provide.impl_name.clone(), witness));
    }
    Ok(selected)
}

/// 设备执行覆盖不到的见证：只有“本机没有这台设备”时全部进自检。
///
/// 设备在本机时 [`run_all`] 已经把每一条声明跑过一遍：设备执行比 host 自检严格，
/// 同一份见证没有必要再在主机上重跑一次。
fn self_checkable(profile: &Profile) -> Vec<Selected<'_>> {
    if device_available(profile) {
        return Vec::new();
    }
    profile
        .provides
        .iter()
        .filter_map(|provide| {
            provide
                .witness
                .as_ref()
                .map(|witness| (provide.prim, provide.impl_name.clone(), witness))
        })
        .collect()
}

/// 这次协商的依据。
///
/// 两者不能混着说：`Executed` 是"这台机器刚刚跑通了见证"，`Declared` 是"档案这么写，本机
/// 执行不了这台设备"。报告里必须写出是哪一种，否则读者会把自述当成证据。
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

/// 一台设备这一次的证据：跑通了哪些见证、依据是什么、哪一步不成立。
///
/// 它是 `doctor` / `verify` / `bench` 共用的那一份结论。分开算的代价不是性能，而是**不一致**：
/// 三条命令各自读一遍档案时，同一台设备可以同时"协商可行"（doctor）与"见证不成立"（verify）。
#[derive(Debug, Clone, PartialEq)]
pub struct Evidence {
    /// 协商的依据。
    pub basis: Basis,
    /// 设备执行通过的见证条数（这台机器就是该设备，结论能归到具体供给）。
    pub device: usize,
    /// 本机自检通过的见证条数（设备执行跑不到的那些）。
    pub host: usize,
    /// 这次没能确立的证据：见证编译/执行/对拍失败，或声明了 `L2` 却没有见证。
    ///
    /// `None` 表示档案里每一条已声明的见证都有执行结论。本机没有这台设备不算证据失败——
    /// 它写进 [`Evidence::basis`]，而不是这里。
    pub failure: Option<Error>,
    /// 设备执行的编译参数（档案的 `[compile].flags`）。
    ///
    /// 证据是对着这组参数编出的程序取得的：参数一变，产物身份就变（见
    /// [`cutile::cache::ArtifactKey`]），见证会重编重跑。报告里写出它，读者才能知道这份证据
    /// 适用于哪一次代码生成——片段、工具链、参数三者合起来才是能力本身。
    pub device_params: String,
    /// 本机自检的编译参数（[`HOST_FLAGS`]）：与设备执行不是同一套，必须分开报。
    pub host_params: String,
    /// 见证在**它自己声称的语言**里的合法性（[`LanguageGate`]）。
    ///
    /// 设备执行跑得到的设备是 [`LanguageGate::Covered`]；其余设备按 `emit` 找语言前端，
    /// 用这门语言自己的解析器把见证源码过一遍。它不是设备证据（没有一行在设备上执行），
    /// 所以不改任何 `grade`；但被拒绝是硬错误——在自己语言里编不过的不算见证。
    pub language: LanguageGate,
    /// 真的被证明的供给（执行结论）。
    proof: Proof,
}

impl Evidence {
    /// 协商该用的证据。
    ///
    /// [`Basis::Executed`] 时是执行结论；[`Basis::Declared`] 时退回档案自述——本机执行不了
    /// 这台设备，没有第二条路，所以依据必须由 [`Evidence::basis`] 说出来。
    #[must_use]
    pub fn proof(&self, profile: &Profile) -> Proof {
        match self.basis {
            Basis::Executed => self.proof.clone(),
            Basis::Declared => Proof::declared(profile),
        }
    }
}

/// 跑这台设备的全部见证，得到三条命令共用的证据。
///
/// 不返回 `Result`：`doctor` 的职责正是解释"为什么不行"，一次见证执行失败也是一条要报的
/// 事实，而不是让它连报告都交不出来。要失败语义的调用方（`verify` / `bench`）读
/// [`Evidence::failure`] 自己决定。
#[must_use]
pub fn establish(profile: &Profile, settings: &Settings) -> Evidence {
    let executed = device_available(profile);
    let mut proof = Proof::none();
    let mut device = 0;
    let mut failure = None;
    match run_all(profile, settings) {
        Ok(runs) => {
            device = runs.len();
            for run in runs {
                proof.insert(run.prim, &run.impl_name);
            }
        }
        // 设备不在本机时，这里的第一件事就是"本机没有这台设备"：它是设备的存在性结论，
        // 不是证据不成立。写进 failure 会让 `doctor` 把一台 GPU 说成"见证写错了"。
        Err(error) => {
            if executed {
                failure = Some(error);
            }
        }
    }
    let host = match self_check(profile, settings) {
        Ok(runs) => runs.len(),
        Err(error) => {
            failure = failure.or(Some(error));
            0
        }
    };
    let language = language_gate(profile, settings);
    if let LanguageGate::Rejected { reason } = &language {
        // 被自己的语言拒绝是证据缺陷：它与"本机没有这台设备"不是同一类事。
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

/// 自检用的 host C 参数。
///
/// 设备档案的 `[compile].flags` 在这台机器上要么不存在（`#include <cuda_fp16.h>`），要么
/// 只对真机编译器有意义（`-arch=sm_100a`），所以自检固定用这一套。它与设备执行的那一套
/// 是两个不同的编译参数集，报告的 `[证据]` 段因此分开写。
pub const HOST_FLAGS: [&str; 2] = ["-O2", "-std=c99"];

/// 一个前端的共享执行流程：生成程序 → 跑 → 逐条对拍。
///
/// `F` 与 `OUT` 都参与单态化，所以"生成什么、怎么跑、怎么比"各只有一份实现；差异全部压进
/// [`Device::program`] 与 [`Device::run`]，这里不认识任何具体语言。
fn execute<F: Frontend, const OUT: usize>(
    profile: &Profile,
    settings: &Settings,
    selected: &[Selected<'_>],
    scope: Scope,
) -> diagnostic::Result<Vec<WitnessRun>> {
    let program = F::program::<OUT>(profile, settings, selected, scope)?;
    let (counts, bits) = F::run(profile, settings, &program, selected.len(), scope)?;
    compare(selected, scope, &counts, &bits)
}

/// C 前端：见证本身就是一段 host C，`cc` 把它与 harness 编成可执行文件跑一遍。
///
/// 语言身份（`Emit::C` / `c` / `witness`）由 `#[frontend]` 生成，`EXT` 与 `Emit::ext()` 的
/// 一致性由注解展开出的 `const _` 断言在编译期检查；本实现只写"怎么跑"。
#[annotation::frontend(emit = "c", ext = "c", tag = "witness")]
pub struct CBackend;

impl Device for CBackend {
    fn available(_profile: &Profile) -> bool {
        // `c` 的设备就是本机 CPU：它永远在。
        true
    }

    fn program<const OUT: usize>(
        profile: &Profile,
        settings: &Settings,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> diagnostic::Result<String> {
        let (preamble, _) = c_compile_inputs(profile, scope);
        let mut sources: BTreeMap<String, String> = BTreeMap::new();
        for (prim, _, witness) in selected {
            if sources.contains_key(&witness.file) {
                continue;
            }
            let text =
                source(witness, settings).map_err(|error| error.with(format!("`{prim}` 的")))?;
            if !text.contains(&format!("{}(", witness.func)) {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "`{prim}` 的见证 `{}` 里没有函数 `{}(float *)`",
                        witness.file, witness.func
                    ),
                ));
            }
            sources.insert(witness.file.clone(), text);
        }

        let mut program = preamble;
        if !program.is_empty() && !program.ends_with('\n') {
            program.push('\n');
        }
        program.push_str("#include <stdio.h>\n\n");
        // 自检用 host C 前端：MSL 的地址空间限定符在 C 里没有对应物，`thread` 必须能被忽略。
        // 只影响自检——设备前端走真正的 `xcrun metal`，不经过这一行。
        if scope == Scope::Host
            && selected
                .iter()
                .any(|(_, _, witness)| witness.file.ends_with(".metal"))
        {
            program.push_str("#define thread\n");
        }
        for text in sources.values() {
            program.push_str(text);
            if !program.ends_with('\n') {
                program.push('\n');
            }
            program.push('\n');
        }
        program.push_str(&harness::<OUT>(selected));
        Ok(program)
    }

    fn run(
        profile: &Profile,
        settings: &Settings,
        program: &str,
        _count: usize,
        scope: Scope,
    ) -> diagnostic::Result<(Vec<i64>, Vec<f32>)> {
        let (_, flags) = c_compile_inputs(profile, scope);
        // 设备执行与自检是两个不同的缓存命名空间：同一段文本在两套参数下编出的产物不同。
        let tag = match scope {
            Scope::Device => Self::TAG,
            Scope::Host => "self",
        };
        // 见证程序也进产物缓存：第二次 `verify` 只在第一次调用编译器。
        let (outcome, _artifact) =
            build::run_program(backend::detect().as_ref(), &flags, tag, program, settings)?;
        Ok((parse_counts(&outcome.stderr), outcome.bits))
    }
}

/// C 前端的编译输入：前置声明 + 参数。
///
/// 自检只走 host C ABI：设备的 `[compile]` 前置声明与参数在这台机器上要么不存在
/// （`#include <cuda_fp16.h>`），要么只对真机编译器有意义（`-arch=sm_100a`）。
fn c_compile_inputs(profile: &Profile, scope: Scope) -> (String, Vec<String>) {
    match scope {
        Scope::Device => (profile.preamble.clone(), profile.flags.clone()),
        Scope::Host => (
            String::new(),
            HOST_FLAGS.iter().map(|flag| (*flag).to_string()).collect(),
        ),
    }
}

/// 逐条对拍：个数走 stderr、值走 stdout，两路必须与调用顺序一一对应。
///
/// C 与 Metal 两个前端只差"怎么让见证跑起来"：编译、计数与对拍完全共用，否则同一份 `expect`
/// 会在两条路径上被两种口径判定。
fn compare(
    selected: &[Selected<'_>],
    scope: Scope,
    counts: &[i64],
    bits: &[f32],
) -> diagnostic::Result<Vec<WitnessRun>> {
    if counts.len() != selected.len() {
        return Err(Error::new(
            Code::Internal,
            format!(
                "见证执行只回报了 {} 条计数，期望 {} 条：前端没有按顺序调用每条见证",
                counts.len(),
                selected.len()
            ),
        ));
    }

    let tolerance = reference::tolerance(DType::F32);
    let mut offset = 0usize;
    let mut runs = Vec::with_capacity(selected.len());
    for ((prim, impl_name, witness), &count) in selected.iter().zip(counts) {
        let written = usize::try_from(count).unwrap_or(usize::MAX);
        if written != witness.expect.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{prim}` 的见证 `{}` 写了 {count} 个值，档案声明期望 {} 个",
                    witness.func,
                    witness.expect.len()
                ),
            ));
        }
        let end = offset + written;
        let got = bits.get(offset..end).ok_or_else(|| {
            Error::new(
                Code::Internal,
                format!(
                    "`{prim}` 的见证 `{}` 只回来了 {} 个值（第 {} 段需要 {written} 个）",
                    witness.func,
                    bits.len(),
                    runs.len() + 1
                ),
            )
        })?;
        let error = reference::max_rel_err(got, &witness.expect);
        if error > tolerance {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{prim}` 的见证 `{}` 不成立：得到 {got:?}，期望 {:?}，max_rel_err {error:.3e} > {tolerance:.0e}",
                    witness.func, witness.expect
                ),
            ));
        }
        runs.push(WitnessRun {
            prim: *prim,
            impl_name: impl_name.clone(),
            func: witness.func.clone(),
            values: written,
            max_rel_err: error,
            scope,
        });
        offset = end;
    }
    if offset != bits.len() {
        return Err(Error::new(
            Code::Internal,
            format!(
                "见证输出多出 {} 个值：计数行与实际打印不一致",
                bits.len() - offset
            ),
        ));
    }
    Ok(runs)
}

/// 见证源码：内置设备读编译期内嵌副本（发布机没有源码树），`data.root` 或未登记的设备走磁盘。
fn source(witness: &Witness, settings: &Settings) -> diagnostic::Result<String> {
    let ext = std::path::Path::new(&witness.file)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    let vendor = witness.file.split('/').next().unwrap_or("");
    if settings.data_root().is_none()
        && let Some(text) = prim::builtin_witness(vendor, ext)
    {
        return Ok(text.to_string());
    }
    let path = witness_root(settings).join(&witness.file);
    std::fs::read_to_string(&path).map_err(|error| {
        Error::new(
            Code::Io,
            format!("见证 `{}` 不可读：{error}", path.display()),
        )
    })
}

/// 见证源码的根目录：与 `Catalogue` 的装载根保持一致（`data.root` 优先）。
fn witness_root(settings: &Settings) -> PathBuf {
    settings
        .data_root()
        .map_or_else(prim::dir, prim::layer_root)
}

/// 生成 `main()`：逐条调用见证函数，值走 stdout，个数走 stderr。
///
/// 个数必须单独走一条通道：stdout 只放 f32 位模式，`backend::parse_bits` 才能原样解码；
/// 把计数混进 stdout 会让"写了几个"与"值是多少"互相污染。
fn harness<const OUT: usize>(selected: &[Selected<'_>]) -> String {
    // `OUT` 是 const 泛型参数：`max` 在这个实例里就是一个常量，生成的代码里所有上限都是字面量。
    let max = OUT;
    let mut text = format!("\nint main(void) {{\n    float out[{max}];\n    int n;\n");
    for (_, _, witness) in selected {
        text.push_str("    n = ");
        text.push_str(&witness.func);
        text.push_str("(out);\n");
        text.push_str("    fprintf(stderr, \"witness %d\\n\", n);\n");
        text.push_str("    if (n < 0) n = 0;\n");
        text.push_str(&format!("    if (n > {max}) n = {max};\n"));
        text.push_str("    for (int i = 0; i < n; ++i) {\n");
        text.push_str("        union { float f; unsigned int u; } v;\n");
        text.push_str("        v.f = out[i];\n");
        text.push_str("        printf(\"%u\\n\", v.u);\n");
        text.push_str("    }\n");
    }
    text.push_str("    return 0;\n}\n");
    text
}

/// Metal 前端：MSL → `metallib` → host 驱动发射到 GPU → 回读。
///
/// 为什么不能用 host C 自检代替：MSL 没有默认地址空间，`float *out` 在 `.metal` 里是语法错误，
/// 而 `cc` 只因为源码是纯 C 才放行——验证的是"这段 C 能编译"，不是"这条 Metal 见证存在"。
/// 本实现把同一个文件交给真正的 Metal 编译器，并让它在 GPU 上跑一遍。
///
/// 语言身份（`Emit::Metal` / `metal` / `witness`）由 `#[frontend]` 生成，本实现只写"怎么跑"。
#[annotation::frontend(emit = "metal", ext = "metal", tag = "witness")]
pub struct MetalBackend;

impl Device for MetalBackend {
    fn available(_profile: &Profile) -> bool {
        // 工具链在，就说明这台机器有 Metal 编译器；再往下才是"有没有 GPU"。
        metal_toolchain().is_some()
    }

    fn program<const OUT: usize>(
        profile: &Profile,
        settings: &Settings,
        selected: &[Selected<'_>],
        _scope: Scope,
    ) -> diagnostic::Result<String> {
        metal_program::<OUT>(profile, settings, selected)
    }

    fn run(
        profile: &Profile,
        settings: &Settings,
        program: &str,
        count: usize,
        _scope: Scope,
    ) -> diagnostic::Result<(Vec<i64>, Vec<f32>)> {
        // 设备参数来自档案：`-std=metal3.1` 之类决定了这段 MSL 按哪个语言版本编。
        run_metal(program, count, &profile.flags, settings)
    }
}

/// MSL 源：档案的 `[compile].preamble` + 见证源码 + 一个按 `tid` 分派的 kernel。
///
/// 每条见证一个 thread：`tid` 是见证在本次执行里的下标，写出的个数与值分别落到 `counts` 与
/// `out[tid * OUT ..]`。`OUT` 是 const 泛型参数，生成的 MSL 里所有缓冲区尺寸都是字面量。
fn metal_program<const OUT: usize>(
    profile: &Profile,
    settings: &Settings,
    selected: &[Selected<'_>],
) -> diagnostic::Result<String> {
    let max = OUT;
    let mut program = profile.preamble.clone();
    if !program.is_empty() && !program.ends_with('\n') {
        program.push('\n');
    }
    let mut files: BTreeMap<String, String> = BTreeMap::new();
    for (prim, _, witness) in selected {
        if files.contains_key(&witness.file) {
            continue;
        }
        let text = source(witness, settings).map_err(|error| error.with(format!("`{prim}` 的")))?;
        if !text.contains(&format!("{}(", witness.func)) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{prim}` 的见证 `{}` 里没有函数 `{}(thread float *)`",
                    witness.file, witness.func
                ),
            ));
        }
        files.insert(witness.file.clone(), text);
    }
    for text in files.values() {
        program.push_str(text);
        if !program.ends_with('\n') {
            program.push('\n');
        }
        program.push('\n');
    }

    program.push_str(&format!(
        "kernel void cutile_witness(device float *out [[buffer(0)]],\n\
         \x20                          device int *counts [[buffer(1)]],\n\
         \x20                          uint tid [[thread_position_in_grid]]) {{\n\
         \x20   thread float local[{max}];\n\
         \x20   for (int i = 0; i < {max}; ++i) local[i] = 0.0f;\n\
         \x20   int n = 0;\n\
         \x20   switch (tid) {{\n"
    ));
    for (index, (_, _, witness)) in selected.iter().enumerate() {
        program.push_str(&format!(
            "        case {index}: n = {}(local); break;\n",
            witness.func
        ));
    }
    program.push_str(&format!(
        "        default: break;\n\
         \x20   }}\n\
         \x20   int cap = n < 0 ? 0 : (n > {max} ? {max} : n);\n\
         \x20   counts[tid] = cap;\n\
         \x20   for (int i = 0; i < {max}; ++i) out[tid * {max} + i] = i < cap ? local[i] : 0.0f;\n\
         }}\n"
    ));
    Ok(program)
}

/// 编译并执行一次 Metal 见证程序，返回 `(每条写入个数, 按顺序回读的值)`。
///
/// 三处调用都要看得见失败原因：`xcrun metal`（语言层）、`metallib`（链接）、`clang`（驱动）；
/// 驱动自身再分"没有 Metal 设备"与"运行失败"两种。诊断带上完整输出与源文件路径，否则
/// `float *out` 这种错误只会以一句 "compile failed" 出现。
fn run_metal(
    program: &str,
    count: usize,
    flags: &[String],
    settings: &Settings,
) -> diagnostic::Result<(Vec<i64>, Vec<f32>)> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = settings.work_dir().join(format!(
        "witness-metal-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let outcome = (|| {
        std::fs::create_dir_all(&dir).map_err(|error| {
            Error::new(Code::Io, format!("创建 {} 失败：{error}", dir.display()))
        })?;
        let msl = dir.join("witness.metal");
        let air = dir.join("witness.air");
        let metalib = dir.join("witness.metallib");
        let driver = dir.join("driver.m");
        let exe = dir.join("driver");
        std::fs::write(&msl, program)
            .map_err(|error| Error::new(Code::Io, format!("写 {} 失败：{error}", msl.display())))?;

        run_tool(
            "xcrun metal",
            std::process::Command::new("xcrun")
                .args(["-sdk", "macosx", "metal"])
                .args(flags)
                .arg("-c")
                .arg(&msl)
                .arg("-o")
                .arg(&air),
            &msl,
        )?;
        run_tool(
            "xcrun metallib",
            std::process::Command::new("xcrun")
                .args(["-sdk", "macosx", "metallib"])
                .arg(&air)
                .arg("-o")
                .arg(&metalib),
            &msl,
        )?;

        let compiler = backend::detect().ok_or_else(|| {
            Error::new(
                Code::Unsupported,
                "缺少 host C 编译器：Metal 见证需要一个 host 驱动把内核发射到 GPU".to_string(),
            )
        })?;
        std::fs::write(&driver, METAL_DRIVER).map_err(|error| {
            Error::new(Code::Io, format!("写 {} 失败：{error}", driver.display()))
        })?;
        run_tool(
            &compiler.command,
            std::process::Command::new(&compiler.command)
                .args(["-fobjc-arc", "-O2", "-o"])
                .arg(&exe)
                .arg(&driver)
                .args(["-framework", "Foundation", "-framework", "Metal"]),
            &driver,
        )?;

        let run = std::process::Command::new(&exe)
            .arg(&metalib)
            .arg(count.to_string())
            .output()
            .map_err(|error| {
                Error::new(Code::Io, format!("运行 {} 失败：{error}", exe.display()))
            })?;
        if !run.status.success() {
            return Err(Error::new(
                Code::Unsupported,
                format!(
                    "Metal 见证执行失败（退出码 {:?}）：{}",
                    run.status.code(),
                    String::from_utf8_lossy(&run.stderr).trim()
                ),
            ));
        }
        let stderr = String::from_utf8_lossy(&run.stderr).to_string();
        Ok((
            parse_counts(&stderr),
            backend::parse_bits(&String::from_utf8_lossy(&run.stdout)),
        ))
    })();
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

/// 跑一条工具命令；非零退出码时把命令、完整输出与源文件路径一并带进诊断。
fn run_tool(
    name: &str,
    command: &mut std::process::Command,
    source: &std::path::Path,
) -> diagnostic::Result<()> {
    let output = command
        .output()
        .map_err(|error| Error::new(Code::Io, format!("调用 `{name}` 失败：{error}")))?;
    if output.status.success() {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!(
            "`{name}` 执行失败：\n{}\n{}\n源文件：{}",
            String::from_utf8_lossy(&output.stdout).trim(),
            String::from_utf8_lossy(&output.stderr).trim(),
            source.display()
        ),
    ))
}

/// 把 `metallib` 里的 `cutile_witness` 发射到 GPU，回读个数与值。
///
/// 计数走 stderr、值走 stdout——与 C harness 同一口径，`compare` 因此只认一条路径。
/// 缓冲区用 `MTLResourceStorageModeShared`：要看的是值，不是带宽。
const METAL_DRIVER: &str = r#"#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: driver <metallib> <count>\n");
            return 2;
        }
        const int count = atoi(argv[2]);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "本机没有 Metal 设备\n");
            return 3;
        }
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        if (!library) {
            fprintf(stderr, "metallib 装载失败：%s\n", error.localizedDescription.UTF8String);
            return 4;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"cutile_witness"];
        if (!function) {
            fprintf(stderr, "metallib 里没有 cutile_witness\n");
            return 5;
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            fprintf(stderr, "管线创建失败：%s\n", error.localizedDescription.UTF8String);
            return 6;
        }
        id<MTLBuffer> values = [device newBufferWithLength:(NSUInteger)count * 8 * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> counts = [device newBufferWithLength:(NSUInteger)count * sizeof(int)
                                                   options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:values offset:0 atIndex:0];
        [encoder setBuffer:counts offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)count, 1, 1)];
        [encoder endEncoding];
        [buffer commit];
        [buffer waitUntilCompleted];
        if (buffer.error) {
            fprintf(stderr, "内核执行失败：%s\n", buffer.error.localizedDescription.UTF8String);
            return 7;
        }
        const int *written = (const int *)counts.contents;
        const float *out = (const float *)values.contents;
        for (int i = 0; i < count; ++i) {
            fprintf(stderr, "witness %d\n", written[i]);
        }
        fflush(stderr);
        for (int i = 0; i < count; ++i) {
            int n = written[i];
            if (n < 0) n = 0;
            if (n > 8) n = 8;
            for (int j = 0; j < n; ++j) {
                union { float f; unsigned int u; } v;
                v.f = out[i * 8 + j];
                printf("%u\n", v.u);
            }
        }
    }
    return 0;
}
"#;

/// 从 stderr 读出每条见证声明的写入个数（按调用顺序）。
fn parse_counts(stderr: &str) -> Vec<i64> {
    stderr
        .lines()
        .filter_map(|line| line.trim().strip_prefix("witness "))
        .filter_map(|value| value.trim().parse::<i64>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutile::builtin::Catalogue;

    fn settings() -> Settings {
        let roots = llm_config::Roots::new(std::env::temp_dir().join("cutile-witness-tests"));
        Settings::<llm_config::Cutile>::defaults(&roots)
    }

    fn cpu() -> Profile {
        Catalogue::try_builtin()
            .expect("内置数据")
            .profile("cpu")
            .cloned()
            .expect("cpu 档案")
    }

    #[test]
    fn every_declared_witness_executes_on_the_device() {
        if backend::detect().is_none() {
            return;
        }
        let profile = cpu();
        let declared = profile.provides.len();
        let runs = run_all(&profile, &settings()).expect("见证必须全部通过");
        assert_eq!(runs.len(), declared, "每条已声明的见证都要有一条执行结果");
        assert!(
            runs.iter().all(|run| run.scope == Scope::Device),
            "cpu 就是本机，它的见证拿到的必须是设备结论"
        );
        assert!(runs.iter().all(|run| run.max_rel_err <= 1e-5), "{runs:?}");
    }

    #[test]
    fn a_wrong_expectation_is_a_failure() {
        if backend::detect().is_none() {
            return;
        }
        let mut profile = cpu();
        profile.provides[0].witness.as_mut().expect("有见证").expect[0] += 1.0;
        let error = run_all(&profile, &settings()).expect_err("期望值不对必须失败");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("quant"), "{error}");
    }

    #[test]
    fn a_witness_must_write_exactly_the_declared_number_of_values() {
        if backend::detect().is_none() {
            return;
        }
        let mut profile = cpu();
        profile.provides[0]
            .witness
            .as_mut()
            .expect("有见证")
            .expect
            .truncate(1);
        let error = run_all(&profile, &settings()).expect_err("个数不符必须失败");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("写了 2 个值"), "{error}");
    }

    #[test]
    fn a_missing_function_is_a_failure_without_compiling() {
        let mut profile = cpu();
        profile.provides[0].witness.as_mut().expect("有见证").func = "witness_absent".to_string();
        let error = run_all(&profile, &settings()).expect_err("函数不存在必须失败");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("witness_absent"), "{error}");
    }

    fn device(vendor: &str) -> Profile {
        Catalogue::try_builtin()
            .expect("内置数据")
            .profile(vendor)
            .cloned()
            .unwrap_or_else(|| panic!("{vendor} 档案"))
    }

    /// 每台设备的每条已声明见证恰好被执行一次，且由能给出更强结论的范围执行。
    ///
    /// 设备在本机时全部声明走设备执行，自检一条不跑；设备不在本机时全部进自检。
    /// 两个范围不重叠也没有遗漏——这是“每条声明都有结论”的判据。
    #[test]
    fn every_declared_witness_is_executed_once_in_the_strongest_scope() {
        if backend::detect().is_none() {
            return;
        }
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        for profile in catalogue.profiles() {
            let vendor = profile.meta.vendor.as_str();
            let declared = profile
                .provides
                .iter()
                .filter(|provide| provide.witness.is_some())
                .count();
            let runs = self_check(profile, &settings())
                .unwrap_or_else(|error| panic!("{vendor} 的见证自检失败：{error}"));
            assert!(
                runs.iter().all(|run| run.scope == Scope::Host),
                "{vendor}: 自检的结论不能被当成设备证据"
            );
            let executed = if device_available(profile) {
                run_all(profile, &settings())
                    .unwrap_or_else(|error| panic!("{vendor} 的设备执行失败：{error}"))
                    .len()
            } else {
                0
            };
            assert_eq!(
                runs.len() + executed,
                declared,
                "{vendor}: 自检 {} + 设备执行 {executed} ≠ 声明的 {declared}",
                runs.len()
            );
            assert!(
                device_available(profile) == (runs.is_empty()),
                "{vendor}: 设备在本机时不该再跑一遍自检（同一份见证只执行一次）"
            );
        }
    }

    /// `cpu` 的 `L0` 见证此前两头落空：设备执行只挑 `grade ≥ L2`，自检又只跑非 C 设备。
    /// 按等级筛选执行会让"没敢写 `L2`"变成"永远拿不到结论"，所以执行不按等级选：
    /// 低等级声明同样在本机跑一遍，`grade` 才不会两头都不成立。
    #[test]
    fn a_cpu_l0_witness_is_executed_on_the_device_too() {
        if backend::detect().is_none() {
            return;
        }
        let profile = cpu();
        let unverified = profile
            .provides
            .iter()
            .filter(|provide| !provide.grade.is_verified())
            .map(|provide| provide.prim)
            .collect::<Vec<_>>();
        assert!(!unverified.is_empty(), "前提：cpu 至少有一条 L0 声明");
        let runs = run_all(&profile, &settings()).expect("L0 见证也必须能跑");
        let checked = runs.iter().map(|run| run.prim).collect::<Vec<_>>();
        for prim in unverified {
            assert!(
                checked.contains(&prim),
                "`{prim}` 的 L0 见证没有被执行：{checked:?}"
            );
        }
    }

    /// Metal 的见证只能在 GPU 上验：本机没有 Metal 工具链时这条用例不成立，直接跳过。
    ///
    /// 它同时是"见证必须在自己的语言里存在"的判据：MSL 没有默认地址空间，写 `float *out`
    /// 会在这里以 `xcrun metal` 的编译错误失败，而不是像 host `cc` 那样放行。
    #[test]
    fn every_metal_witness_runs_on_the_gpu() {
        let profile = device("metal");
        if !device_available(&profile) {
            return;
        }
        let runs = run_all(&profile, &settings()).expect("Metal 见证必须全部通过");
        assert_eq!(runs.len(), profile.provides.len(), "每条声明都要有结果");
        assert!(
            runs.iter().all(|run| run.scope == Scope::Device),
            "GPU 上跑出来的结论不能被降级成自检"
        );
        assert!(runs.iter().all(|run| run.max_rel_err <= 1e-5), "{runs:?}");
    }

    /// 见证用错地址空间时，报错必须来自真正的 Metal 前端，而不是被 host C 放过。
    ///
    /// 这就是 `witness.metal` 曾经的写法：`int witness_x(float *out)` 在 C 里合法、在 MSL 里
    /// 是语法错误，`cc` 自检一路绿灯，等于这份"证据"从未在它声称的语言里存在过。
    #[test]
    fn a_plain_c_signature_does_not_survive_the_metal_front_end() {
        if !device_available(&device("metal")) {
            return;
        }
        let program = "#include <metal_stdlib>\nusing namespace metal;\n\
                       int witness_bad(float *out) { out[0] = 1.0f; return 1; }\n";
        let profile = device("metal");
        let error = MetalBackend::run(&profile, &settings(), program, 1, Scope::Device)
            .expect_err("MSL 必须要求显式地址空间");
        let text = error.to_string();
        assert!(text.contains("xcrun metal"), "{text}");
        assert!(text.contains("address space"), "{text}");
    }

    /// 自检走 host C ABI：设备的 `[compile]` 前置声明与参数不能泄进来（这台机器上既没有
    /// `cuda_fp16.h`，也没有 `nvcc`）。把 preamble 换成一个必然编不过的 `#error`，自检仍必须通过。
    #[test]
    fn the_self_check_ignores_the_device_preamble_and_flags() {
        if backend::detect().is_none() {
            return;
        }
        let mut profile = device("nvidia");
        profile.preamble = "#error 设备专属前置声明\n".to_string();
        profile.flags = vec!["-arch=sm_100a".to_string(), "-nocudainc".to_string()];
        let runs = self_check(&profile, &settings()).expect("自检不得使用设备 preamble/flags");
        assert!(!runs.is_empty(), "nvidia 有已声明的见证");
    }

    /// 自检是硬判据：期望值错一位就失败，而不是“没设备所以不知道”。
    #[test]
    fn a_wrong_expectation_fails_even_the_host_self_check() {
        if backend::detect().is_none() {
            return;
        }
        let mut profile = device("ascend");
        profile.provides[0].witness.as_mut().expect("有见证").expect[0] += 1.0;
        let error = self_check(&profile, &settings()).expect_err("期望值不对必须失败");
        assert!(error.to_string().contains("见证自检不成立"), "{error}");
        assert!(error.to_string().contains("ascend"), "{error}");
    }

    /// 一台设备的见证必须在**它自己声称的语言**里编得过：host 自检只说明"这段代码在 C 里
    /// 成立"，而 `float *out` 事故证明这句话不够。
    ///
    /// `cuda` / `opencl` 有 clang 的语言前端，门必须盖住每一条声明；`ascendc` 没有开源前端，
    /// 门如实缺席——缺席也要说清是哪门语言，不能是空白。
    #[test]
    fn every_accelerator_witness_parses_in_its_own_language() {
        for (vendor, selector) in [("nvidia", "-x cuda"), ("intel", "-x cl")] {
            let profile = device(vendor);
            let declared = profile
                .provides
                .iter()
                .filter(|provide| provide.witness.is_some())
                .count();
            match language_gate(&profile, &settings()) {
                LanguageGate::Passed { front, witnesses } => {
                    assert!(front.contains(selector), "{vendor}: {front}");
                    assert_eq!(witnesses, declared, "{vendor}: 门要盖住每一条声明");
                }
                // 本机没有 clang 时门开不了：这条用例的前提不成立。
                LanguageGate::Absent { reason } if reason.contains("不在本机") => {}
                other => panic!("{vendor} 的见证没有在 `{selector}` 里过一遍：{other}"),
            }
        }
        match language_gate(&device("ascend"), &settings()) {
            LanguageGate::Absent { reason } => assert!(
                reason.contains("ascendc"),
                "缺席也要说清是哪门语言：{reason}"
            ),
            other => panic!("AscendC 没有开源前端，门不该开成 `{other}`"),
        }
    }

    /// 设备执行跑得到时，门不必再开：能真跑的就不该只解析。
    #[test]
    fn a_witness_that_runs_on_the_device_needs_no_language_gate() {
        assert_eq!(language_gate(&cpu(), &settings()), LanguageGate::Covered);
        let metal = device("metal");
        assert_eq!(
            matches!(language_gate(&metal, &settings()), LanguageGate::Covered),
            device_available(&metal),
            "只有能在设备上真跑时才免开语言门"
        );
    }

    /// 门夹在"host C 收"与"这门语言不收"之间：同一份源码在 `cc` 里合法、在 OpenCL C 里是
    /// 错误（`double` 没开 `cl_khr_fp64`）。两者同时成立，门才不是自检的重复。
    #[test]
    fn a_witness_its_own_language_rejects_is_a_defect() {
        let Some(toolchain) = backend::detect() else {
            return;
        };
        if clang().is_none() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("cutile-language-gate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("临时目录");
        let source = dir.join("witness.cl");
        std::fs::write(
            &source,
            "int witness_double(float *out) { double x = 2.0; out[0] = (float)x; return 1; }\n",
        )
        .expect("写见证");

        let rejected = parse_one::<OpenClFront>(&source).expect_err("OpenCL C 不允许裸 double");
        assert!(rejected.to_string().contains("witness.cl"), "{rejected}");
        assert!(rejected.to_string().contains("-x cl"), "{rejected}");

        // 同一份源码换个后缀交给 host C：`cc` 一路绿灯——差就在这一段。
        let host = dir.join("witness.c");
        std::fs::copy(&source, &host).expect("复制");
        let status = std::process::Command::new(&toolchain.command)
            .args(["-std=c99", "-fsyntax-only"])
            .arg(&host)
            .status()
            .expect("跑 host C 编译器");
        assert!(status.success(), "前提：同一份源码在 host C 里合法");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn counts_are_read_from_stderr_in_order() {
        assert_eq!(parse_counts("witness 1\nwitness 2\n"), vec![1, 2]);
        assert_eq!(parse_counts("noise\n"), Vec::<i64>::new());
    }
}
