//! L3 端到端误差合成律（LLD §4.5）：逐层证书 → 网络级误差上界，以及这张证书**管到哪**。
//!
//! 定理本身只有两条递推（`a_0 = R`，`e_0 = 0`）：
//!
//! ```text
//! a_i = L_i · ρ̂_i · a_{i-1} + b_i
//! e_i ≤ L_i · (ρ_i · e_{i-1} + ε_i · a_{i-1})
//! ```
//!
//! `ρ_i = ‖W_i‖₂`、`ρ̂_i = ‖Ŵ_i‖₂`、`ε_i = ‖W_i − Ŵ_i‖₂`、`L_i` 是该层非线性的 Lipschitz 上界、
//! `b_i = ‖φ_i(0)‖₂`、`R = ‖x‖₂`。落地时有四条取舍，它们决定这份上界能不能当证书用：
//!
//! - **`L_i` 没有默认值**：`ReLU`/`GELU`/`SiLU` 都不是全局 1-Lipschitz，把 `L_i` 取 1 会凭空
//!   削掉一个因子。本模块因此不提供任何"内置激活"常量，`L_i` 只能由调用方给出实测或已证的值。
//! - **`ρ̂_i` 不冒充 `ρ_i`**：`ρ̂ ≤ ρ` 只有在真正验证过时才成立，闭式里左侧用 `ρ̂`、右侧用 `ρ`，
//!   两者不互相替换（替换就是把"压缩没放大"当成前提，而那正是要证的东西）。
//! - **适用范围要说出来**：LayerNorm / 残差 / 注意力 / SSM / MoE / 门控都在这条定理之外。移出的
//!   段数记在 [`Chain::with_uncovered`]，于是网络级判定可能只到 [`Ruling::Partial`]——
//!   "线性层无损"不等于"整网可证无损"。
//! - **上界为零不等于逐位还原**：`ε = 0` 只说明证书这一项为零，`Ŵ = W` 要由实际 artifact 的往返
//!   给出；缺了后者就是 [`Ruling::Unverified`]，而不是悄悄算成功。

use crate::bad;
use diagnostic::Result;

use crate::ledger::{Bound, EpsMethod, Note, Stable};

/// 一段网络的 L3 事实：一个可适用段的五个量。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rung {
    rho: f64,
    rho_hat: f64,
    eps: f64,
    lipschitz: f64,
    offset: f64,
}

impl Rung {
    /// 由五个量构造；任一为负或非有限都在这里拒绝（负的范数不是"更紧的界"，是记账错误）。
    pub fn new(rho: f64, rho_hat: f64, eps: f64, lipschitz: f64, offset: f64) -> Result<Self> {
        let facts = [
            ("ρ", rho),
            ("ρ̂", rho_hat),
            ("ε", eps),
            ("L", lipschitz),
            ("b", offset),
        ];
        for (name, value) in facts {
            check(name, value)?;
        }
        Ok(Self {
            rho,
            rho_hat,
            eps,
            lipschitz,
            offset,
        })
    }

    /// 从谱隙证书取 `ε`；`ρ`、`ρ̂`、`L`、`b` 仍由调用方给。
    pub fn from_note(
        note: &Note,
        rho: f64,
        rho_hat: f64,
        lipschitz: f64,
        offset: f64,
    ) -> Result<Self> {
        if note.stable != Stable::Pass {
            return Err(bad("证书没过 L0 门：谱隙不足以支撑这次秩选择"));
        }
        if !note.equal {
            return Err(bad("证书报告往返不一致：ε 没有可用的载体"));
        }
        if note.method == EpsMethod::Candidate || note.bound == Bound::Candidate {
            return Err(bad(
                "证书用的是候选口径（幂迭代 / L0'）：它的 ε 只够排序，不够当上界",
            ));
        }
        Self::new(rho, rho_hat, note.eps, lipschitz, offset)
    }

    /// `‖W‖₂` 的上界。
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// `‖Ŵ‖₂` 的上界。
    pub fn rho_hat(&self) -> f64 {
        self.rho_hat
    }

    /// `‖W − Ŵ‖₂` 的上界。
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// 该层非线性的 Lipschitz 上界 `L_i`。
    pub fn lipschitz(&self) -> f64 {
        self.lipschitz
    }

    /// `b_i = ‖φ_i(0)‖₂`。
    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// 这一段自身是否无损：`ε` 精确为零。
    pub fn lossless(&self) -> bool {
        self.eps == 0.0
    }
}

/// 一段纳入 L3 的网络：输入界 `R` 与逐段事实。
#[derive(Clone, Debug)]
pub struct Chain {
    input: f64,
    rungs: Vec<Rung>,
    uncovered: usize,
}

impl Chain {
    /// 由输入界与逐段事实构造；`input` 必须非负有限（`R = ‖x‖₂` 的上界）。
    pub fn new(input: f64, rungs: Vec<Rung>) -> Result<Self> {
        check("R", input)?;
        Ok(Self {
            input,
            rungs,
            uncovered: 0,
        })
    }

    /// 声明有几段被移出 L3（LayerNorm / 残差 / 注意力 / SSM / MoE / 门控）。
    #[must_use]
    pub fn with_uncovered(mut self, count: usize) -> Self {
        self.uncovered = count;
        self
    }

    /// 输入界 `R`。
    pub fn input(&self) -> f64 {
        self.input
    }

    /// 逐段事实（层序）。
    pub fn rungs(&self) -> &[Rung] {
        &self.rungs
    }

    /// 被移出 L3 的段数。
    pub fn uncovered(&self) -> usize {
        self.uncovered
    }

    /// 合成两侧界：`a_L`（激活幅度上界）与 `e_L`（误差上界），以及适用时的闭式值。
    pub fn bracket(&self) -> Result<Bracket> {
        let (activation, error) = walk(&self.rungs, self.input)?;
        Ok(Bracket {
            activation,
            error,
            closed: closed_form(&self.rungs, self.input),
            lossless: self.rungs.iter().all(Rung::lossless),
            uncovered: self.uncovered,
        })
    }
}

/// 一次合成的结果：两个界，加"这张证书管到哪"的三个事实。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bracket {
    activation: f64,
    error: f64,
    closed: Option<f64>,
    lossless: bool,
    uncovered: usize,
}

impl Bracket {
    /// `a_L`：激活幅度的上界。它随段数增长，所以 `ε` 靠后的段罚得更重。
    pub fn activation(&self) -> f64 {
        self.activation
    }

    /// `e_L`：网络输出与压缩网络输出之差的范数上界。
    pub fn error(&self) -> f64 {
        self.error
    }

    /// 闭式 `R·Σ ε_i (Π_{j<i} ρ̂_j)(Π_{j>i} ρ_j)`；只在**所有** `L_i = 1` 且 `b_i = 0` 时给出。
    pub fn closed_form(&self) -> Option<f64> {
        self.closed
    }

    /// 每段 `ε` 都是零（上界为零）。
    pub fn is_lossless(&self) -> bool {
        self.lossless
    }

    /// 被移出 L3 的段数。
    pub fn uncovered(&self) -> usize {
        self.uncovered
    }

    /// 网络级判定：`bitwise` 是调用方对**真实 artifact** 的往返结论。
    pub fn ruling(&self, bitwise: bool) -> Ruling {
        if !self.lossless {
            return Ruling::Refuse;
        }
        if !bitwise {
            return Ruling::Unverified;
        }
        if self.uncovered > 0 {
            return Ruling::Partial;
        }
        Ruling::Lossless
    }
}

/// 网络级四态判定（闭集，码位与短名稳定）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ruling {
    /// 所有段 `ε = 0`、无移出段、artifact 往返逐位：可以签发整网无损证书。
    Lossless,
    /// 所有段 `ε = 0` 且往返逐位，但有段在 L3 之外：只能签发"适用段无损"。
    Partial,
    /// 所有段 `ε = 0`，但缺 artifact 往返证据：上界为零不算还原。
    Unverified,
    /// 有非零分项：本次压缩有损，按 §4.5 拒绝并回退到无损路径。
    Refuse,
}

impl Ruling {
    /// 账本里的字符串写法（闭集，码位稳定）。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Lossless => "lossless",
            Self::Partial => "partial",
            Self::Unverified => "unverified",
            Self::Refuse => "refuse",
        }
    }
}

/// 逐段推进两侧界；每一步都必须是有限数。
fn walk(rungs: &[Rung], input: f64) -> Result<(f64, f64)> {
    let mut activation = input;
    let mut error = 0.0f64;
    for (index, rung) in rungs.iter().enumerate() {
        error = rung.lipschitz * (rung.rho * error + rung.eps * activation);
        activation = rung.lipschitz * rung.rho_hat * activation + rung.offset;
        if !(error.is_finite() && activation.is_finite()) {
            return Err(bad(format!("第 {index} 段越出 f64：无穷大的上界不是上界")));
        }
    }
    Ok((activation, error))
}

/// 闭式：只在全部 `L_i = 1` 且 `b_i = 0` 时成立，其它情形整条不给。
fn closed_form(rungs: &[Rung], input: f64) -> Option<f64> {
    let applies = rungs
        .iter()
        .all(|rung| rung.lipschitz == 1.0 && rung.offset == 0.0);
    if !applies {
        return None;
    }
    Some(input * latent_sum(rungs))
}

/// `Σ ε_i (Π_{j<i} ρ̂_j)(Π_{j>i} ρ_j)`：左乘用 `ρ̂`、右乘用 `ρ`。
fn latent_sum(rungs: &[Rung]) -> f64 {
    // 后缀积先算好，避免每项重扫一遍；`suffix[i]` 是第 i 段**之后**所有 `ρ` 的积。
    let mut suffix = vec![1.0f64; rungs.len() + 1];
    for (index, rung) in rungs.iter().enumerate().rev() {
        suffix[index] = suffix[index + 1] * rung.rho;
    }
    let mut prefix = 1.0f64;
    let mut total = 0.0f64;
    for (index, rung) in rungs.iter().enumerate() {
        total += rung.eps * prefix * suffix[index + 1];
        prefix *= rung.rho_hat;
    }
    total
}

/// 单个量必须是有限非负数：负的范数不是"更紧的界"，`NaN` 更会让整条比较链静默失效。
fn check(label: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(bad(format!("{label} 必须是有限的非负数，实际 {value}")));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/l3.rs"]
mod tests;
