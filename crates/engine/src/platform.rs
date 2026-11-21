//! 六平台执行合同（规则 0.5 / §5.12 / §5.14 的 `E_PLAT`）。
//!
//! `lld.md` 把六行固定成 **CUDA / HIP / Metal / GPU / x86 / ARM**，并且给了一条不能自证的纪律：
//! 缺设备只能 `OPEN`，**不删行、不写成已交付**。此前这条纪律只活在文档里——代码侧没有任何地方
//! 能回答"这一行现在什么状态"，于是"已交付"只能靠人记。本模块把那一格落成可判定的：一行 = 一份
//! 静态合同（[`Row`]）+ 一条带证据的状态（[`Standing`]），判定经由宿主证词（[`Host`]），
//! 缺证据就返回 `E_PLAT`。
//!
//! # 四个决定
//!
//! **状态只有两种，而且互不替代**：[`Standing::Covered`] 必带证据串、[`Standing::Open`] 必带理由
//! [`Why`]。没有"大概能跑"这第三种状态——那正是 `E_PLAT` 要禁止的：`OPEN` 不等于降到 Should
//! （§1.12），也不等于"先按已交付算"。
//!
//! **CPU 行的证据是架构本身，不是工具链**：x86 / ARM 两行的 `Covered` 由 [`Host::arch`] 直接推，
//! 不经过 [`Host::device`]。这样任何宿主实现都无法把"x86 行覆盖"写成一句空话（例如让
//! `device(row) = true` 一劳永逸），也不需要为 CPU 行另造一次编译器探测。
//!
//! **设备行要两件**：设备探测位 + 编译器实测身份。少一件就 `OPEN`，但两种缺法的理由分开报
//! （[`Why::NoDevice`] / [`Why::NoToolchain`]）——"这台机器没有 A800"与"这台机器有 GPU 却装不上
//! nvcc"对结论的影响完全不同，合成一句"不可用"会让台账失去诊断价值。
//!
//! **GPU 行不另开第七行**：非 CUDA / 非 HIP / 非 Metal 的通用 GPU 合同（厂商 MMA + 同构 ABI）
//! 就是 [`Row::Gpu`]，`vulkan` / `sycl` / `spirv` 都映射进它。[`Row::from_language`] 是这张映射表
//! 的唯一入口，多开一行等于把"六平台"偷偷改成七平台。
//!
//! # 本模块不拥有什么
//!
//! 不测数值、不判 E6 的通过率（那是协议卡 `tools::card` 的事）、不编译内核（`backends` /
//! `dialect`）、不碰设备内存（`metal`）。它只回答"这一行现在能不能开门"，以及开门时手里那张
//! 证据长什么样。

use diagnostic::{Code, Error, Result};

/// 报错出处：六平台合同的判定点只有这一个。
const SITE: &str = "engine::platform";

/// 一行执行合同。**次序是合同的一部分**（规则 0.5）：CUDA / HIP / Metal / GPU / x86 / ARM。
///
/// `#[repr(usize)]` 与模块内那张规格表的下标一一对应；`specs_line_up_with_the_rows` 钉住这件事，
/// 于是加一行必须同时改表、改长度断言、改所有权登记——想漏一行都做不到。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(usize)]
pub enum Row {
    /// NVIDIA 数据中心 / 工作站 GPU：CUTLASS 类 WGMMA prefill + XQA decode。
    Cuda = 0,
    /// AMD GPU：MFMA / WMMA。
    Hip = 1,
    /// Apple Silicon GPU：`simdgroup matrix`。
    Metal = 2,
    /// 非 CUDA / HIP / Metal 的**通用** GPU 合同：只承诺厂商 MMA + 同构 ABI。
    Gpu = 3,
    /// x86-64 CPU：AMX / AVX 有序点积。
    X86 = 4,
    /// ARM CPU：NEON / SME 有序点积。
    Arm = 5,
}

/// 一行的 ABI 形态（§5.10 / §5.12）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Abi {
    /// 同构 ABI：设备侧与宿主侧对同一算式给出同一结果，差别只在数值容差（§10.4 的 `1e-5 / 1e-6`）。
    Homogeneous,
    /// 有序点积：求和次序是合同的一部分——换次序就是换算法，不是换实现。
    OrderedDot,
}

/// 一行的静态合同：名字、一等原语、ABI，以及"覆盖需不需要编译器证据"。
///
/// 私有：合同对外只经 [`Row`] 的访问器读出，调用方拿不到一个可以就地改字段的句柄。
struct Spec {
    name: &'static str,
    primitive: &'static str,
    abi: Abi,
    compiler: bool,
}

/// 六行合同表。下标 = [`Row`] 的判别式（`#[repr(usize)]` 与这张表必须同序）。
static SPECS: [Spec; 6] = [
    Spec {
        name: "cuda",
        primitive: "厂商 MMA（CUTLASS/WGMMA）+ 同构 ABI",
        abi: Abi::Homogeneous,
        compiler: true,
    },
    Spec {
        name: "hip",
        primitive: "厂商 MMA（MFMA/WMMA）+ 同构 ABI",
        abi: Abi::Homogeneous,
        compiler: true,
    },
    Spec {
        name: "metal",
        primitive: "simdgroup matrix + 同构 ABI",
        abi: Abi::Homogeneous,
        compiler: true,
    },
    Spec {
        name: "gpu",
        primitive: "厂商 MMA + 同构 ABI",
        abi: Abi::Homogeneous,
        compiler: false,
    },
    Spec {
        name: "x86",
        primitive: "AMX/AVX 有序点积",
        abi: Abi::OrderedDot,
        compiler: false,
    },
    Spec {
        name: "arm",
        primitive: "NEON/SME 有序点积",
        abi: Abi::OrderedDot,
        compiler: false,
    },
];

/// 执行栈里的语言名 → 合同行。`vulkan` / `sycl` / `spirv` 一律并入 `gpu` 行。
const LANGUAGES: &[(&str, Row)] = &[
    ("cuda", Row::Cuda),
    ("nvcc", Row::Cuda),
    ("hip", Row::Hip),
    ("hipcc", Row::Hip),
    ("rocm", Row::Hip),
    ("metal", Row::Metal),
    ("msl", Row::Metal),
    ("vulkan", Row::Gpu),
    ("spirv", Row::Gpu),
    ("sycl", Row::Gpu),
    ("gpu", Row::Gpu),
    ("x86", Row::X86),
    ("x86_64", Row::X86),
    ("avx", Row::X86),
    ("arm", Row::Arm),
    ("aarch64", Row::Arm),
    ("neon", Row::Arm),
];

impl prof::atom::Closed for Row {
    /// 六行，**永不删行**（规则 0.5）：缺设备的行也在里面，只是状态为 [`Standing::Open`]。
    ///
    /// 次序即合同次序，也是那张规格表的下标（`#[repr(usize)]` 的判别式）；名字表的唯一来源是
    /// 这里，`Row::as_str` / `Row::ALL` / `Row::parse` 都是它的转发。
    const ALL: &'static [Self] = &[
        Self::Cuda,
        Self::Hip,
        Self::Metal,
        Self::Gpu,
        Self::X86,
        Self::Arm,
    ];

    /// 合同名，小写、稳定，台账与文档里引用的就是它。
    fn as_str(self) -> &'static str {
        self.spec().name
    }

    fn index(self) -> usize {
        self as usize
    }
}

prof::inherent_names!(Row);

impl Row {
    /// 本行的静态合同（私有表里那一格）。
    fn spec(self) -> &'static Spec {
        &SPECS[self as usize]
    }

    /// 一等原语：一行承诺"靠什么算矩阵"，不是"支持什么指令"。
    #[must_use]
    pub fn primitive(self) -> &'static str {
        self.spec().primitive
    }

    /// ABI 形态：GPU 行同构、CPU 行有序点积。
    #[must_use]
    pub fn abi(self) -> Abi {
        self.spec().abi
    }

    /// 覆盖这一行是否需要编译器实测证据（设备行要，CPU 行不要：证据是架构本身）。
    #[must_use]
    pub fn compiler(self) -> bool {
        self.spec().compiler
    }

    /// 语言名 → 合同行。大小写与首尾空白不敏感；不认识的名字返回 `None`（不猜成 `gpu`）。
    #[must_use]
    pub fn from_language(language: &str) -> Option<Self> {
        let key = language.trim().to_ascii_lowercase();
        LANGUAGES
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, row)| *row)
    }
}

/// 一行的当前状态：**只有两种**，`Covered` 必带证据、`Open` 必带理由。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Standing {
    /// 有实证：`evidence` 是台账里要能对得上的那一行（架构 + 原语，或编译器实测身份 + 原语）。
    Covered {
        /// 覆盖证据：`aarch64 · NEON/SME 有序点积`、`nvcc@12.6 · 厂商 MMA（CUTLASS/WGMMA）+ 同构 ABI`。
        evidence: String,
    },
    /// 缺证据。`OPEN` 不等于降 Must（§1.12），也不等于"先按已交付算"。
    Open {
        /// 缺哪一种证据。
        why: Why,
    },
}

/// 缺的是哪一种证据。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Why {
    /// 本机没有这一行的设备（CPU 行则是架构不符）。
    NoDevice,
    /// 有设备探测位但探测不到该行的编译器——装了编译器不等于有设备，两者分开报。
    NoToolchain,
}

impl Why {
    /// 稳定短名，台账与测试引用它。
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoDevice => "no-device",
            Self::NoToolchain => "no-toolchain",
        }
    }

    /// 写进错误与台账的那一句：理由必须说清"缺什么才算覆盖"。
    #[must_use]
    pub const fn sentence(self) -> &'static str {
        match self {
            Self::NoDevice => "本机没有该行的设备证据，不得假装跑在该行",
            Self::NoToolchain => "本机探测不到该行的编译器，覆盖要设备与编译器两件",
        }
    }
}

impl Standing {
    /// 稳定短名：`covered` / `open`。
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Covered { .. } => "covered",
            Self::Open { .. } => "open",
        }
    }

    /// 这一行现在能不能开门。
    #[must_use]
    pub const fn is_covered(&self) -> bool {
        matches!(self, Self::Covered { .. })
    }

    /// 覆盖证据；`Open` 时没有证据可给（而不是给一句占位文本）。
    #[must_use]
    pub fn evidence(&self) -> Option<&str> {
        match self {
            Self::Covered { evidence } => Some(evidence),
            Self::Open { .. } => None,
        }
    }
}

/// 宿主架构闭集：只用来回答 x86 / ARM 两行有没有设备证据。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arch {
    /// x86-64。
    X86_64,
    /// ARM64（`aarch64` / `arm64`）。
    Aarch64,
    /// 其余架构：两行 CPU 合同都不覆盖，不硬塞进最近的那一行。
    Other,
}

impl Arch {
    /// 本进程所在宿主的架构——`std::env::consts::ARCH` 的唯一读取点。
    #[must_use]
    pub fn of_host() -> Self {
        match std::env::consts::ARCH {
            "x86_64" | "x86" => Self::X86_64,
            "aarch64" | "arm64" => Self::Aarch64,
            _ => Self::Other,
        }
    }

    /// 稳定短名，也是覆盖证据里的架构段。
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::X86_64 => "x86_64",
            Self::Aarch64 => "aarch64",
            Self::Other => "other",
        }
    }
}

/// 一台宿主的证词：判定只经由它，于是"在 A800 上跑"与"台账声称在 A800 上跑"能分开验。
///
/// **CPU 行不由这里回答**：x86 / ARM 的证据是架构本身（[`Host::arch`] 直接推），任何宿主实现都
/// 无法把"x86 行覆盖"写成一句空话。
pub trait Host {
    /// 宿主架构。
    fn arch(&self) -> Arch;

    /// 这一行的设备在不在。只对设备行提问；CPU 行由 [`Arch`] 回答。
    fn device(&self, row: Row) -> bool;

    /// 这一行的编译器实测身份（`nvcc@12.6`、`xcrun metal@32023`），探测不到时 `None`。
    fn toolchain(&self, row: Row) -> Option<String>;
}

/// 真实本机：设备与工具链都走**已有**的探测，不新写第二份。
///
/// Metal 走 `metal::MetalDevice::system_default`（进程内默认设备），CUDA / Metal 的编译器走
/// `toolchain::detect_nvcc` / `detect_metal`（`nvcc --version`、`xcrun -sdk macosx metal --version`）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Machine;

impl Host for Machine {
    fn arch(&self) -> Arch {
        Arch::of_host()
    }

    fn device(&self, row: Row) -> bool {
        match row {
            Row::Metal => metal_device(),
            // CUDA / HIP / GPU：本仓没有进程内设备探针。**没有探针不等于没有设备，也不等于有**——
            // 缺证据就保持 OPEN（`E_PLAT`），这比猜一句"本机应该有"诚实。
            _ => false,
        }
    }

    fn toolchain(&self, row: Row) -> Option<String> {
        match row {
            Row::Cuda => crate::toolchain::detect_nvcc().map(|found| found.id()),
            Row::Metal => crate::toolchain::detect_metal().map(|found| found.id()),
            // HIP / GPU 没有编译器探测位。要让它们从 OPEN 变成 COVERED，得先有探测，而不是先有结论。
            _ => None,
        }
    }
}

/// 进程内默认 Metal 设备在不在（非 macOS 恒为 `false`，连符号都不引）。
#[cfg(target_os = "macos")]
fn metal_device() -> bool {
    crate::metal::executor::MetalDevice::system_default().is_some()
}

/// 非 macOS：Metal 行没有设备证据。
#[cfg(not(target_os = "macos"))]
fn metal_device() -> bool {
    false
}

/// 这一行现在是什么状态。
///
/// 判定次序是"设备 → 编译器"：先说清有没有设备，再说清有没有编译器。反过来的话，"这台机器连
/// GPU 都没有"会被报成"缺 nvcc"，台账就会把工程问题说成环境问题。
#[must_use]
pub fn standing(row: Row, host: &impl Host) -> Standing {
    if !present(row, host) {
        return Standing::Open { why: Why::NoDevice };
    }
    match identity(row, host) {
        Some(segment) => Standing::Covered {
            evidence: compose(row, &segment),
        },
        None => Standing::Open {
            why: Why::NoToolchain,
        },
    }
}

/// 证据串 = 身份段 + 一等原语。GPU 行靠原语自证时身份段为空，此时不再把原语写第二遍。
fn compose(row: Row, segment: &str) -> String {
    if segment.is_empty() {
        row.primitive().to_string()
    } else {
        format!("{segment} · {}", row.primitive())
    }
}

/// 设备证据：CPU 行由架构直接回答，其余行交给宿主。
fn present(row: Row, host: &impl Host) -> bool {
    match row {
        Row::X86 => host.arch() == Arch::X86_64,
        Row::Arm => host.arch() == Arch::Aarch64,
        _ => host.device(row),
    }
}

/// 覆盖证据的身份段：CPU 行是架构名，设备行是编译器实测身份。
fn identity(row: Row, host: &impl Host) -> Option<String> {
    match row {
        Row::X86 | Row::Arm => Some(host.arch().name().to_string()),
        Row::Cuda | Row::Hip | Row::Metal => host.toolchain(row),
        // GPU 行的一等原语就是"厂商 MMA"：设备证据成立时它自证，不再单独要求编译器；
        // 身份段因此可以为空，由 [`compose`] 决定不写第二遍。
        Row::Gpu => Some(host.toolchain(row).unwrap_or_default()),
    }
}

/// 六行台账：**长度就是六**——缺的行没法从类型里消失（`[Standing; 6]` 而不是 `Vec`）。
#[must_use]
pub fn audit(host: &impl Host) -> [Standing; 6] {
    std::array::from_fn(|index| standing(Row::ALL[index], host))
}

/// 开门判定：该行没有覆盖证据就返回 `E_PLAT`（[`Code::Unsupported`]），不让调用方"看着办"。
///
/// `E_PLAT` 专指"设备/编译器缺失"这一种状态（§5.14），因此判定与状态出自同一条 [`standing`]，
/// 不存在"台账说 OPEN、开门却放行"的两份真值。
///
/// # Errors
///
/// 该行为 [`Standing::Open`] 时返回 [`Code::Unsupported`]；消息里带行名、缺哪一种证据与那句判据。
pub fn require(row: Row, host: &impl Host) -> Result<()> {
    match standing(row, host) {
        Standing::Covered { .. } => Ok(()),
        Standing::Open { why } => Err(Error::new(
            Code::Unsupported,
            format!(
                "{} 行状态 OPEN（错误码 E_PLAT）：{}",
                row.as_str(),
                why.sentence()
            ),
        )
        .with(SITE)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一台可以随便摆布证词的宿主：测试只摆事实，不依赖本机装了什么。
    struct Fake {
        arch: Arch,
        device: bool,
        toolchain: Option<&'static str>,
    }

    impl Fake {
        const fn bare(arch: Arch) -> Self {
            Self {
                arch,
                device: false,
                toolchain: None,
            }
        }
    }

    impl Host for Fake {
        fn arch(&self) -> Arch {
            self.arch
        }

        fn device(&self, _row: Row) -> bool {
            self.device
        }

        fn toolchain(&self, _row: Row) -> Option<String> {
            self.toolchain.map(str::to_string)
        }
    }

    #[test]
    fn specs_line_up_with_the_rows() {
        assert_eq!(Row::ALL.len(), SPECS.len(), "六行不缺不多");
        for (index, row) in Row::ALL.iter().enumerate() {
            assert_eq!(*row as usize, index, "判别式必须等于表下标");
            assert_eq!(
                row.as_str(),
                SPECS[index].name,
                "{} 的名字与表错位",
                row.as_str()
            );
        }
    }

    #[test]
    fn the_six_rows_come_in_the_contract_order() {
        let names: Vec<&str> = Row::ALL.iter().map(|row| row.as_str()).collect();
        assert_eq!(names, ["cuda", "hip", "metal", "gpu", "x86", "arm"]);
    }

    #[test]
    fn gpu_rows_speak_homogeneous_abi_and_cpu_rows_ordered_dot() {
        for &row in Row::ALL {
            let expected = if row.compiler() || row == Row::Gpu {
                Abi::Homogeneous
            } else {
                Abi::OrderedDot
            };
            assert_eq!(row.abi(), expected, "{} 的 ABI 形态不符", row.as_str());
        }
        assert!(!Row::X86.primitive().is_empty(), "原语不能是空话");
    }

    #[test]
    fn vulkan_and_sycl_map_into_the_gpu_row() {
        for language in ["vulkan", "SYCL", " spirv ", "gpu"] {
            assert_eq!(Row::from_language(language), Some(Row::Gpu), "{language}");
        }
        assert_eq!(Row::from_language("hipcc"), Some(Row::Hip));
        assert_eq!(
            Row::from_language("xcrun metal"),
            None,
            "不猜：整串不是名字"
        );
        assert_eq!(Row::from_language("rocm"), Some(Row::Hip));
    }

    #[test]
    fn every_row_is_audited_even_when_the_host_has_nothing() {
        let host = Fake::bare(Arch::Other);
        let ledger = audit(&host);
        assert_eq!(ledger.len(), Row::ALL.len(), "缺设备也不许少一行");
        for (row, standing) in Row::ALL.iter().zip(&ledger) {
            assert_eq!(standing.as_str(), "open", "{} 不该凭空覆盖", row.as_str());
            assert_eq!(standing.evidence(), None);
            assert_eq!(
                standing,
                &Standing::Open { why: Why::NoDevice },
                "{} 缺的是设备",
                row.as_str()
            );
        }
    }

    #[test]
    fn a_cpu_row_is_covered_by_the_architecture_alone() {
        let host = Fake::bare(Arch::Aarch64);
        let arm = standing(Row::Arm, &host);
        assert!(arm.is_covered(), "架构对得上就该覆盖，不看工具链");
        let evidence = arm.evidence().expect("覆盖必带证据");
        assert!(evidence.contains("aarch64"), "{evidence}");
        assert!(evidence.contains(Row::Arm.primitive()), "{evidence}");
        assert_eq!(
            standing(Row::X86, &host),
            Standing::Open { why: Why::NoDevice },
            "aarch64 宿主不覆盖 x86 行"
        );
    }

    #[test]
    fn a_device_row_needs_both_the_device_and_the_compiler() {
        let device_only = Fake {
            arch: Arch::Aarch64,
            device: true,
            toolchain: None,
        };
        assert_eq!(
            standing(Row::Cuda, &device_only),
            Standing::Open {
                why: Why::NoToolchain
            },
            "有设备没编译器：理由必须是编译器，不是设备"
        );

        let both = Fake {
            arch: Arch::Aarch64,
            device: true,
            toolchain: Some("nvcc@12.6"),
        };
        let covered = standing(Row::Cuda, &both);
        let evidence = covered.evidence().expect("两件都有才覆盖");
        assert!(evidence.contains("nvcc@12.6"), "{evidence}");
        assert!(evidence.contains("CUTLASS/WGMMA"), "{evidence}");
    }

    #[test]
    fn the_gpu_row_self_certifies_on_the_vendor_primitive() {
        let host = Fake {
            arch: Arch::X86_64,
            device: true,
            toolchain: None,
        };
        let gpu = standing(Row::Gpu, &host);
        assert!(
            gpu.is_covered(),
            "GPU 行的一等原语就是厂商 MMA，不必另要编译器"
        );
        assert_eq!(gpu.evidence(), Some(Row::Gpu.primitive()));
    }

    #[test]
    fn require_reports_e_plat_instead_of_passing() {
        let host = Fake::bare(Arch::Aarch64);
        let error = require(Row::Cuda, &host).expect_err("缺设备必须开门失败");
        assert_eq!(error.code(), Code::Unsupported, "错误码就是 E_PLAT");
        let text = error.to_string();
        assert!(text.contains("cuda"), "{text}");
        assert!(text.contains("OPEN"), "{text}");
        assert!(text.contains(Why::NoDevice.sentence()), "{text}");
        require(Row::Arm, &host).expect("架构覆盖的行必须放行");
    }

    #[test]
    fn stable_short_names() {
        assert_eq!(
            Standing::Covered {
                evidence: String::new()
            }
            .as_str(),
            "covered"
        );
        assert_eq!(Standing::Open { why: Why::NoDevice }.as_str(), "open");
        assert_eq!(Why::NoDevice.as_str(), "no-device");
        assert_eq!(Why::NoToolchain.as_str(), "no-toolchain");
        assert_eq!(Arch::Other.name(), "other");
    }

    #[test]
    fn the_machine_tells_this_host_without_guessing() {
        let host = Machine;
        assert_eq!(host.arch(), Arch::of_host(), "架构只从一处读");
        let ledger = audit(&host);
        assert_eq!(ledger.len(), 6);
        for (row, standing) in Row::ALL.iter().zip(&ledger) {
            assert!(
                matches!(standing.as_str(), "covered" | "open"),
                "{} 只能是两种状态",
                row.as_str()
            );
            if standing.is_covered() {
                assert!(!standing.evidence().unwrap_or_default().is_empty());
            }
        }
        match host.arch() {
            Arch::Aarch64 => {
                assert!(
                    ledger[Row::Arm as usize].is_covered(),
                    "aarch64 宿主覆盖 ARM 行"
                );
                assert!(!ledger[Row::X86 as usize].is_covered());
            }
            Arch::X86_64 => {
                assert!(
                    ledger[Row::X86 as usize].is_covered(),
                    "x86_64 宿主覆盖 x86 行"
                );
                assert!(!ledger[Row::Arm as usize].is_covered());
            }
            Arch::Other => {
                assert!(!ledger[Row::Arm as usize].is_covered());
                assert!(!ledger[Row::X86 as usize].is_covered());
            }
        }
    }
}
