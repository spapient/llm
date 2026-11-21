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

mod arch;

pub use arch::{Arch, Host, Machine, Standing, Why, audit, require, standing};
#[cfg(test)]
#[cfg(test)]
#[cfg(test)]
#[path = "../tests/platform.rs"]
mod tests;

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
pub(crate) struct Spec {
    pub(super) name: &'static str,
    primitive: &'static str,
    abi: Abi,
    compiler: bool,
}

/// 六行合同表。下标 = [`Row`] 的判别式（`#[repr(usize)]` 与这张表必须同序）。
pub(crate) static SPECS: [Spec; 6] = [
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
