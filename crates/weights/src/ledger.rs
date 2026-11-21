//! 谱隙证书账本：L0 判定、秩选择与证书同源，只有一个入口 [`certify`]。
//!
//! LLD §4.6 的压缩链第一格是「L0 谱隙证书门控（拒压回退）」，§4.7 又要求无损 PASS 必须
//! `ε_synth = 0` 且 artifact 往返逐位一致。这里把两条落成一次分解：同一次 f64 SVD 同时给出受认证
//! 的秩、该秩下的低秩项 `L`、显式残差 `C = W − L`，以及一张自证的 [`Note`]。
//!
//! 关键取舍是 `ε_alg` 的取法：用**实测后验残差** `‖A − UΣVᵀ‖_F`（§4.7 A 的 `verified_residual`），
//! 不拿 `ε_mach·σ_1` 冒充严格界；`ε_obs` 记 0——被分解的矩阵就是落盘的那一份，观测侧没有第二份
//! 数据。判定严格：`γ_r > 2ε` 才 Pass，等号一律 Fail。
//!
//! §4.2.1 要求证书声明**用了哪种** `ε_alg`，§4.3.1 要求声明这张界是 L0 诊断还是 L0' 候选，
//! §4.8 要求 `E_decomp` 与 `E_comp` 分开记、不得重复计入。三条都不是注释而是可执行约束，因此它们
//! 在 [`Note`] 里是闭集字段（[`EpsMethod`]、[`Bound`]、`comp`），由 [`check`] 逐条把关：候选筛查
//! 口径签不出 `Pass`，L0' 候选界签不出无损，逐位往返的补偿项不得再记一遍误差。
//!
//! 旧版把它拆成 `spectral` / `truncate` / `note` 三个函数，每个都重算一遍 SVD，而秩由调用方拍给
//! （`axes` 里取 `min(m,n)/2`）：证书因此只能事后记录、从不参与决策。现在秩由证书给出，判定与产出
//! 同源，[`certify`] 返回 `None` 就是「拒压」本身。

use diagnostic::{Code, Error, Result};

/// 账本 schema 标识。
pub const SCHEMA: &str = "llm.cert.v1";

/// 秩选择稳定性状态（闭集，码位稳定）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stable {
    /// 谱隙满足 `γ_r > 2ε`。
    Pass = 0,
    /// 不满足严格阈值。
    Fail = 1,
    /// 未使用秩选择，无稳定性可判。
    Open = 2,
}

impl Stable {
    pub(crate) fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Self::Pass),
            1 => Ok(Self::Fail),
            2 => Ok(Self::Open),
            other => Err(bad(&format!("未知 Stable 码 {other}"))),
        }
    }
}

/// `ε_alg` 的取法（§4.2.1：证书必须声明方法；缺方法即无效，`ε_mach·σ_1` 不算无条件界）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpsMethod {
    /// 已验证的反向误差界。本仓不产出此值——`ε_mach·σ_1` 不是无条件界，而真正可证的向后误差
    /// 界还没有落地，留码位是为了读端能认出别人签的那种证书。
    Backward = 0,
    /// 已验证残差：本次分解的后验 `‖A − UΣVᵀ‖_F`，[`certify`] 用的就是这条。
    Residual = 1,
    /// 只够做候选筛查（幂迭代等）：**不能**支撑证书，见 [`check`]。
    Candidate = 2,
}

impl EpsMethod {
    pub(crate) fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Self::Backward),
            1 => Ok(Self::Residual),
            2 => Ok(Self::Candidate),
            other => Err(bad(&format!("未知 EpsMethod 码 {other}"))),
        }
    }

    /// 账本里的字符串写法（闭集，码位稳定）。
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Backward => "verified_backward_error",
            Self::Residual => "verified_residual",
            Self::Candidate => "candidate_only",
        }
    }
}

/// 这张证书用的是哪种界（§4.3.1 / §4.7 的 `bound_type`）。
///
/// L0 是**保守诊断界**，L0' 是**研究候选界**——后者从证书链撤下过，所以它永远签不出无损证书：
/// 把候选界当证书用，是把「还没证完」读成「已经证完」。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    /// L0 诊断界：只回答「秩可不可判」，[`certify`] 产出的就是这个。
    Diagnostic = 0,
    /// L0' 候选界：研究口径，只能当候选，不当证书。
    Candidate = 1,
    /// 未声明界：只在没有使用秩选择时允许。
    None = 2,
}

impl Bound {
    pub(crate) fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Self::Diagnostic),
            1 => Ok(Self::Candidate),
            2 => Ok(Self::None),
            other => Err(bad(&format!("未知 Bound 码 {other}"))),
        }
    }

    /// 账本里的字符串写法（闭集，码位稳定）。
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Diagnostic => "l0_diagnostic",
            Self::Candidate => "l0prime_candidate_open",
            Self::None => "none",
        }
    }
}

/// 重构模式（§4.4 第 3 步：纯截断 / 显式残差补偿互斥，不同模式用不同误差项，不得混用）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// 纯截断：只保存低秩项 `L`，误差项 `E_decomp = ‖W−L‖₂`。
    Trunc = 0,
    /// 显式残差补偿：保存 `L + C`（`C = W−L`），误差项 `E_comp = ‖C−Ĉ‖₂`。
    Residual = 1,
}

impl Mode {
    pub(crate) fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Self::Trunc),
            1 => Ok(Self::Residual),
            other => Err(bad(&format!("未知 Mode 码 {other}"))),
        }
    }
}

/// 谱隙稳定性判定：`γ_r > 2ε` 才 Pass，等号/小于号 Fail。
#[must_use]
pub fn stable(gap: f64, eps: f64) -> Stable {
    if gap > 2.0 * eps {
        Stable::Pass
    } else {
        Stable::Fail
    }
}

/// 受认证的秩选择：秩、低秩项 `L`、显式残差 `C`，以及该选择的证书。
#[derive(Clone, Debug)]
pub struct Certified {
    /// 受认证的秩 `r`（1-based，`1 <= r < min(m,n)`）。
    pub rank: usize,
    /// 低秩项 `L = U[:, :r] Σ[:r] V[:, :r]ᵀ`（行主序 `m*n` f32）。
    pub low: Vec<f32>,
    /// 显式残差 `C = W − L`（行主序 `m*n` f32）。
    pub rest: Vec<f32>,
    /// 低秩项的因子：`L = left[m,r] · right[r,n]`，按行主序 `m*r` 与 `r*n`。
    ///
    /// 落盘存的是这两个因子而不是 `L` 本身——`L` 是整张 `m×n`，把它存一遍等于没省；因子只有
    /// `(m+n)·r`。因子以 `f64` 保留分解的原始精度，读端用同一个 [`factor_low`] 重算即可逐位复现
    /// `low`（`f32` 因子会把 `Σ uσv` 舍入两次，逐位判据因此过不了）。
    pub(crate) left: Vec<f64>,
    /// 右因子，行主序 `r*n`（见 [`Certified::left`]）。
    pub(crate) right: Vec<f64>,
    /// 这次选择的证书。
    pub note: Note,
}

/// `left · right` 的 `m×n` 重构，行主序 `f64`。
///
/// 累加次序固定为「行 → 列 → 秩」且不做分块：写端用它算 `low`、读端用它还原 `low`，
/// 任何一方换次序都会让逐位判据失效——所以这里不是性能热点，是**同一份算术**。
pub(crate) fn factor_low(
    left: &[f64],
    right: &[f64],
    (m, n): (usize, usize),
    rank: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for c in 0..n {
            let mut sum = 0.0;
            for j in 0..rank {
                sum += left[i * rank + j] * right[j * n + c];
            }
            out[i * n + c] = sum;
        }
    }
    out
}

/// 可判秩域内的最大秩：`σ_r − σ_{r+1} > 2ε` 的最后一点。
///
/// 取最大值而不是第一个通过的点，因为信号尾部的谱隙最大、噪声底上方的相邻奇异值差会缩到 `ε`
/// 量级，[`stable`] 自然把它们排除。返回的就是**噪声底之前的那个秩**，而不是拍脑袋的
/// `min(m,n)/2`；一处谱隙都不满足时返回 `None`（拒压）。
///
/// 阈值 `ε` 是**无损**口径的算法残差（机器精度量级）：对带可测噪声的满秩矩阵，相邻谱隙远大于
/// `2ε`，本函数会如实返回满秩（该矩阵要无损就必须满秩）。**有损**量化补偿侧的信号秩选择用
/// [`crate::svd::signal_rank`]（相对谱隙阈值 [`crate::svd::REL_GAP`]），满秩噪声矩阵返回
/// `None`（拒压回退）——两者口径不同，不可互换。
pub(crate) fn certified_rank(sigma: &[f64], eps: f64) -> Option<usize> {
    (1..sigma.len())
        .rev()
        .find(|&r| stable(sigma[r - 1] - sigma[r], eps) == Stable::Pass)
}

/// 一张张量的谱隙证书账目（闭集强类型字段）。
#[derive(Clone, Debug, PartialEq)]
pub struct Note {
    /// 目标秩 r（1-based）。
    pub rank: u32,
    /// 是否使用秩选择。
    pub used: bool,
    /// 重构模式（纯截断 / 显式残差补偿）。
    pub mode: Mode,
    /// 稳定性判定。
    pub stable: Stable,
    /// `ε_alg` 的取法（缺方法即无效）。
    pub method: EpsMethod,
    /// 界类型（L0 诊断 / L0' 候选 / 未声明）。
    pub bound: Bound,
    /// 谱隙 γ_r。
    pub gap: f64,
    /// 总扰动 ε = ε_obs + ε_alg。
    pub eps: f64,
    /// 截断误差 E_decomp。
    pub decomp: f64,
    /// 补偿误差 E_comp：只在残差补偿模式存在，逐位往返时精确为零。
    pub comp: f64,
    /// 逐位往返是否一致（无损判定）。
    pub equal: bool,
    /// 是否无损。
    pub lossless: bool,
}

pub(crate) fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("ledger")
}
pub use crate::ledger_impl::{audit, certify, check};

#[cfg(test)]
#[path = "../tests/ledger.rs"]
mod tests;
