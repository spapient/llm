use super::*;

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
pub trait Host {
    /// 宿主架构。
    fn arch(&self) -> Arch;

    /// 这一行的设备在不在。只对设备行提问；CPU 行由 [`Arch`] 回答。
    fn device(&self, row: Row) -> bool;

    /// 这一行的编译器实测身份（`nvcc@12.6`、`xcrun metal@32023`），探测不到时 `None`。
    fn toolchain(&self, row: Row) -> Option<String>;
}

/// 真实本机：设备与工具链都走**已有**的探测，不新写第二份。
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

/// 这一行现在是什么状态。
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
