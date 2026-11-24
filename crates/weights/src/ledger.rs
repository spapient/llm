//! 谱隙证书账本：L0 判定、秩选择与证书同源，只有一个入口 [`certify`]。

use crate::bad;
use diagnostic::Result;
use storage::leb::{self, Cursor};
use storage::sink::Sink;

use crate::svd;

// 闭集枚举由 crate 级 `macros` 的 `closed_enum!` 生成：码位、反向解析与短名同出一处声明。

/// 账本 schema 标识。
pub const SCHEMA: &str = "llm.cert.v1";

closed_enum! {
    /// 秩选择稳定性状态（闭集，码位稳定）。
    pub enum Stable {
        /// 谱隙满足 `γ_r > 2ε`。
        Pass = 0,
        /// 不满足严格阈值。
        Fail = 1,
        /// 未使用秩选择，无稳定性可判。
        Open = 2,
    }
}

closed_enum! {
    /// `ε_alg` 的取法（§4.2.1：证书必须声明方法；缺方法即无效，`ε_mach·σ_1` 不算无条件界）。
    pub enum EpsMethod {
        /// 已验证的反向误差界。本仓不产出此值——`ε_mach·σ_1` 不是无条件界，而真正可证的向后误差
        /// 界还没有落地，留码位是为了读端能认出别人签的那种证书。
        Backward = 0 => "verified_backward_error",
        /// 已验证残差：本次分解的后验 `‖A − UΣVᵀ‖_F`，[`certify`] 用的就是这条。
        Residual = 1 => "verified_residual",
        /// 只够做候选筛查（幂迭代等）：**不能**支撑证书，见 [`check_note`]。
        Candidate = 2 => "candidate_only",
    }
}

closed_enum! {
    /// 这张证书用的是哪种界（§4.3.1 / §4.7 的 `bound_type`）。
    pub enum Bound {
        /// L0 诊断界：只回答「秩可不可判」，[`certify`] 产出的就是这个。
        Diagnostic = 0 => "l0_diagnostic",
        /// L0' 候选界：研究口径，只能当候选，不当证书。
        Candidate = 1 => "l0prime_candidate_open",
        /// 未声明界：只在没有使用秩选择时允许。
        None = 2 => "none",
    }
}

closed_enum! {
    /// 重构模式（§4.4 第 3 步：纯截断 / 显式残差补偿互斥，不同模式用不同误差项，不得混用）。
    pub enum Mode {
        /// 纯截断：只保存低秩项 `L`，误差项 `E_decomp = ‖W−L‖₂`。
        Trunc = 0,
        /// 显式残差补偿：保存 `L + C`（`C = W−L`），误差项 `E_comp = ‖C−Ĉ‖₂`。
        Residual = 1,
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
    pub(crate) left: Vec<f64>,
    /// 右因子，行主序 `r*n`（见 [`Certified::left`]）。
    pub(crate) right: Vec<f64>,
    /// 这次选择的证书。
    pub note: Note,
}

/// 可判秩域内的最大秩：`σ_r − σ_{r+1} > 2ε` 的最后一点。
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

/// L0 门：压前判定，通过则一次给出受认证的秩与两项产物。
pub fn certify(w: &[f32], m: usize, n: usize) -> Result<Option<Certified>> {
    let k = shape(w.len(), m, n)?;
    if k < 2 {
        // `1 <= r < min(m,n)` 无解：秩选择不适用，不颁发证书（拒压），也不是「无损」。
        return Ok(None);
    }
    decompose(w, m, n, k)
}

/// 一次 f64 分解加秩判定：`k = min(m,n) >= 2` 由 [`certify`] 保证。
fn decompose(w: &[f32], m: usize, n: usize, k: usize) -> Result<Option<Certified>> {
    let s = svd::Spectral::open(w, m, n)?;
    // ε 只有算法项：后验残差是本次分解的实测值，不是机器精度的估计。
    let eps = svd::residual(&s.a, &s.svd.reconstruct(k));
    let Some(rank) = certified_rank(&s.svd.sigma, eps) else {
        return Ok(None);
    };
    assemble(w, &s.svd, (m, n), rank, eps).map(Some)
}

/// 受认证秩处的产物与自证证书：低秩项、显式残差、逐位往返与 [`Note`] 都出自同一次分解。
fn assemble(
    w: &[f32],
    d: &svd::Svd,
    (m, n): (usize, usize),
    rank: usize,
    eps: f64,
) -> Result<Certified> {
    let (left, right) = d.factors(rank);
    let low64 = svd::factor_low(&left, &right, (m, n), rank);
    // 低秩项一旦落盘就只剩 `f32` 精度，所以显式残差也必须在 `f32` 世界里取差：拿 `f64` 的差去
    // 存，残差里装的是分解的舍入噪声（每个元素一个非零 ulp），整张残差因此不可压，因子化的
    // 收益被它整块吃掉。取 `f32` 的差则「能精确表示的矩阵残差恰好为零」。
    let low: Vec<f32> = low64.iter().map(|&l| l as f32).collect();
    let rest: Vec<f32> = w.iter().zip(&low).map(|(orig, l)| orig - l).collect();
    // 无损判据与 rank 轴同源：`low + rest` 必须逐位返回原矩阵，然后才谈得上证书。
    let equal = w
        .iter()
        .zip(&low)
        .zip(&rest)
        .all(|((x, low), rest)| (low + rest).to_bits() == x.to_bits());
    let gap = d.sigma[rank - 1] - d.sigma[rank];
    let note = Note {
        rank: rank as u32,
        used: true,
        mode: Mode::Residual,
        stable: stable(gap, eps),
        method: EpsMethod::Residual,
        bound: Bound::Diagnostic,
        gap,
        eps,
        decomp: svd::residual32(w, &low),
        // 残差 `C` 以 `f32` 原样落盘，补偿侧没有再编码一次，所以它这里的误差精确为零。
        comp: 0.0,
        equal,
        lossless: equal,
    };
    check_note(&note)?;
    Ok(Certified {
        rank,
        low,
        rest,
        left,
        right,
        note,
    })
}

/// 形状判据：`m`、`n` 非零且元素数与乘积一致；返回奇异值个数 `min(m,n)`。
fn shape(len: usize, m: usize, n: usize) -> Result<usize> {
    if m == 0 || n == 0 || m.checked_mul(n) != Some(len) {
        return Err(bad(format!("形状不合法 m={m} n={n} len={len}")));
    }
    Ok(m.min(n))
}

// —— 证书编解码与自证（原 `note` 平铺进 `ledger`）：闭集字段怎么落字节、怎么逐条把关 ——

impl Note {
    /// 强类型 LEB128 编码。
    pub(crate) fn encode(&self, out: &mut impl Sink) {
        leb::write(out, self.rank);
        out.put(&[u8::from(self.used)]);
        out.put(&[self.mode as u8]);
        out.put(&[self.stable as u8]);
        out.put(&[self.method as u8]);
        out.put(&[self.bound as u8]);
        out.put(&self.gap.to_le_bytes());
        out.put(&self.eps.to_le_bytes());
        out.put(&self.decomp.to_le_bytes());
        out.put(&self.comp.to_le_bytes());
        out.put(&[u8::from(self.equal)]);
        out.put(&[u8::from(self.lossless)]);
    }

    /// 强类型解码；字段不符显式报错。逐段按 `Note::encode` 的顺序读，段内报错点与逐字段读一致。
    pub fn decode(cur: &mut Cursor<'_>) -> Result<Self> {
        let (rank, used, mode, stable, method, bound) = head(cur)?;
        let ([gap, eps, decomp, comp], [equal, lossless]) = tail(cur)?;
        Ok(Self {
            rank,
            used,
            mode,
            stable,
            method,
            bound,
            gap,
            eps,
            decomp,
            comp,
            equal,
            lossless,
        })
    }

    /// `ε_synth = E_decomp + E_comp`：合成误差的唯一算式。
    #[must_use]
    pub fn synth(&self) -> f64 {
        self.decomp + self.comp
    }

    /// 序列化为字节。
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode(&mut out);
        out
    }
}

/// 标识段的原始字：秩与五个字节（`used` | 模式 | 稳定 | 方法 | 界）。
fn words(cur: &mut Cursor<'_>) -> Result<(u32, [u8; 5])> {
    let rank = cur.read::<u32>()?;
    let flags = cur.take(5)?;
    Ok((rank, [flags[0], flags[1], flags[2], flags[3], flags[4]]))
}

/// 把四个码位解成闭集值；码位不认识就是坏载荷，不猜近似项。
fn signs(flags: &[u8; 5]) -> Result<(Mode, Stable, EpsMethod, Bound)> {
    let mode = Mode::from_code(flags[1])?;
    let stable = Stable::from_code(flags[2])?;
    let method = EpsMethod::from_code(flags[3])?;
    let bound = Bound::from_code(flags[4])?;
    Ok((mode, stable, method, bound))
}

/// 标识段：秩、是否使用秩选择、重构模式码、稳定性码、`ε_alg` 方法码、界类型码。
fn head(cur: &mut Cursor<'_>) -> Result<(u32, bool, Mode, Stable, EpsMethod, Bound)> {
    let (rank, flags) = words(cur)?;
    let (mode, stable, method, bound) = signs(&flags)?;
    Ok((rank, flags[0] != 0, mode, stable, method, bound))
}

/// 实数与标志段：四个 f64（谱隙、总扰动、截断误差、补偿误差）小端连排，其后是两条布尔（非 0 即真）。
fn tail(cur: &mut Cursor<'_>) -> Result<([f64; 4], [bool; 2])> {
    let mut reals = [0.0f64; 4];
    for value in &mut reals {
        *value = read_f64(cur)?;
    }
    let mut flags = [false; 2];
    for flag in &mut flags {
        *flag = cur.read_u8()? != 0;
    }
    Ok((reals, flags))
}

/// 不变量校验：证书自己能查的矛盾，全部在这里挡住。
pub fn check_note(n: &Note) -> Result<()> {
    encoded(n)?;
    measured(n)?;
    compensated(n)?;
    consistent(n)
}

/// 载荷写法：三个误差与谱隙必须是有限非负实数。
fn encoded(n: &Note) -> Result<()> {
    rules(&[
        (
            [n.gap, n.eps, n.decomp, n.comp]
                .iter()
                .all(|v| v.is_finite()),
            "存在非有限值",
        ),
        (n.comp >= 0.0, "补偿误差不能为负"),
        (n.gap >= 0.0, "谱隙不能为负"),
    ])
}

/// `ε` 的口径：候选筛查方法签不出 Pass，L0' 候选界签不出无损，用了秩选择就得说明这张界是谁。
fn measured(n: &Note) -> Result<()> {
    rules(&[
        (
            n.method != EpsMethod::Candidate || n.stable != Stable::Pass,
            "候选筛查方法（幂迭代等）不能支撑 Pass 证书",
        ),
        (
            n.bound != Bound::Candidate || !n.lossless,
            "L0' 候选界是研究口径，不能签发无损证书",
        ),
        (
            !n.used || n.bound != Bound::None,
            "用了秩选择就必须声明这张界是 L0 诊断还是 L0' 候选",
        ),
    ])
}

/// 补偿项的归属：它只属于残差模式，且逐位往返时必为零（否则就是同一份误差记了两遍）。
fn compensated(n: &Note) -> Result<()> {
    rules(&[
        (
            n.comp == 0.0 || n.mode == Mode::Residual,
            "补偿误差只存在于残差补偿模式",
        ),
        (
            !n.equal || n.comp == 0.0,
            "逐位往返说明补偿项没有引入误差，不得再计一遍",
        ),
    ])
}

/// 无损判定与谱隙阈值的自洽：`Pass` 要求 `γ_r > 2ε` 严格成立。
fn consistent(n: &Note) -> Result<()> {
    rules(&[
        (
            n.lossless
                == match n.mode {
                    Mode::Trunc => n.decomp == 0.0,
                    Mode::Residual => n.equal,
                },
            "无损判定与模式/逐位/误差不一致",
        ),
        (
            !(n.used && n.stable == Stable::Pass && n.gap <= 2.0 * n.eps),
            "Pass 但谱隙不满足严格阈值",
        ),
    ])
}

/// 规则表的唯一消费点：第一条不成立的就是答案。
fn rules(table: &[(bool, &str)]) -> Result<()> {
    match table.iter().find(|(holds, _)| !*holds) {
        Some((_, why)) => Err(bad(why)),
        None => Ok(()),
    }
}

/// 读取侧复核：证书里凡是能从载荷自身重算的，都要重算一遍。
pub fn audit(note: &Note, m: usize, n: usize, low: &[f32], rest: &[f32]) -> Result<()> {
    check_note(note)?;
    let rank = note.rank as usize;
    let facts = [
        (note.used, "证书没记录秩选择，却出现在秩载荷里"),
        (
            note.mode == Mode::Residual,
            "显式残差载荷不能声明纯截断模式",
        ),
        (
            note.stable == Stable::Pass,
            "证书没过 L0 门，却被当作载荷落盘",
        ),
        (
            note.method == EpsMethod::Residual,
            "落盘证书的 ε 必须来自实测残差，不能是候选筛查或别人的反向界",
        ),
        (
            note.bound == Bound::Diagnostic,
            "落盘证书的界必须是 L0 诊断：L0' 候选或未声明都签不出可证压缩",
        ),
        (
            note.comp == 0.0,
            "显式残差以 f32 原样落盘，补偿项误差应当为零",
        ),
        (
            (1..m.min(n)).contains(&rank),
            "受认证秩越出 1 <= r < min(m,n)",
        ),
        (low.len() == rest.len(), "两段长度不等，截断误差无从重算"),
    ];
    if let Some((_, why)) = facts.iter().find(|(holds, _)| !*holds) {
        return Err(bad(why));
    }
    let truncation = svd::norm(rest.iter().map(|&v| f64::from(v)));
    let weight = svd::norm(
        low.iter()
            .zip(rest)
            .map(|(&l, &r)| f64::from(l) + f64::from(r)),
    );
    if (truncation - note.decomp).abs() > 1e-6 * (note.decomp.abs() + weight) {
        return Err(bad(format!(
            "截断误差与载荷不符：证书 {}，载荷 {truncation}",
            note.decomp
        )));
    }
    Ok(())
}

fn read_f64(cur: &mut Cursor<'_>) -> Result<f64> {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(cur.take(8)?);
    Ok(f64::from_le_bytes(bytes))
}

#[cfg(test)]
#[path = "../tests/ledger.rs"]
mod tests;
