//! 规划策略与预算：[`Budget`] 是硬约束，[`MemoryPolicy`] 决定预算内怎么走，
//! [`Choice`] 记录实际走了哪条路。

/// 显存预算（字节）：只是一个数，不读环境变量也不读 TOML（配置在 `cli::settings`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Budget {
    bytes: u64,
}

impl Budget {
    /// 由字节数构造；`0` 也是合法值——它会以 [`diagnostic::Code::Invalid`] 报"放不下"，而不是被静默改写。
    pub const fn new(bytes: u64) -> Self {
        Self { bytes }
    }

    /// 预算字节数。
    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

/// 计划走的是哪条路。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Choice {
    /// 预算装得下全部激活：不留检查点，也不重放。
    KeepAll,
    /// 预算不够：按段保留检查点，段内重放。
    Recompute,
}

/// 显存规划策略：预算是硬约束，策略决定在约束内偏好全留还是主动分段重放。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum MemoryPolicy {
    /// 默认：预算装得下就全留，装不下才分段重放；策略让位于实测字节数。
    #[default]
    Auto,
    /// 强制全留：忽略预算松紧，保留全部候选激活，不安排重放（峰值可能超过预算，由调用方兜底）。
    KeepAll,
    /// 强制分段：即使预算宽裕也放检查点、段内重放，用显存换可预测的峰值与释放节奏。
    Recompute,
}

impl MemoryPolicy {
    /// 策略短名：诊断与报告里的稳定标签。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::KeepAll => "keep-all",
            Self::Recompute => "recompute",
        }
    }
}
