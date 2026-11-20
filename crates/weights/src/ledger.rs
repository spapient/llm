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

use crate::leb::{self, Cursor};
use crate::sink::Sink;
use crate::svd;

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
    fn from_code(code: u8) -> Result<Self> {
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
    fn from_code(code: u8) -> Result<Self> {
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
    fn from_code(code: u8) -> Result<Self> {
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
    fn from_code(code: u8) -> Result<Self> {
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
    /// 这次选择的证书。
    pub note: Note,
}

/// L0 门：压前判定，通过则一次给出受认证的秩与两项产物。
///
/// 返回 `Ok(None)` 表示**拒压**：形状退化（`min(m,n) < 2`）或没有满足 `γ_r > 2ε` 的秩（无谱隙，
/// 谱选择不稳定）。调用方据此回退到别的轴——这正是 §4.6「证书不过就拒压回退」的判据。
/// `Err` 只留给实现层失败（形状与长度不符、分解不收敛）。
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
    let a: Vec<f64> = w.iter().map(|&x| x as f64).collect();
    let d = svd::compute(&a, m, n)?;
    // ε 只有算法项：后验残差是本次分解的实测值，不是机器精度的估计。
    let eps = residual(&a, &d.reconstruct(k));
    let Some(rank) = certified_rank(&d.sigma, eps) else {
        return Ok(None);
    };
    assemble(w, &a, &d, rank, eps).map(Some)
}

/// 受认证秩处的产物与自证证书：低秩项、显式残差、逐位往返与 [`Note`] 都出自同一次分解。
fn assemble(w: &[f32], a: &[f64], d: &svd::Svd, rank: usize, eps: f64) -> Result<Certified> {
    let low64 = d.reconstruct(rank);
    let low: Vec<f32> = low64.iter().map(|&l| l as f32).collect();
    let rest: Vec<f32> = a
        .iter()
        .zip(&low64)
        .map(|(orig, l)| (orig - l) as f32)
        .collect();
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
        decomp: residual(a, &low64),
        // 残差 `C` 以 `f32` 原样落盘，补偿侧没有再编码一次，所以它这里的误差精确为零。
        comp: 0.0,
        equal,
        lossless: equal,
    };
    check(&note)?;
    Ok(Certified {
        rank,
        low,
        rest,
        note,
    })
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
fn certified_rank(sigma: &[f64], eps: f64) -> Option<usize> {
    (1..sigma.len())
        .rev()
        .find(|&r| stable(sigma[r - 1] - sigma[r], eps) == Stable::Pass)
}

/// 后验残差 `‖A − B‖_F`：`ε_alg` 的实测值。
fn residual(a: &[f64], b: &[f64]) -> f64 {
    norm(a.iter().zip(b).map(|(x, y)| x - y))
}

/// Frobenius 范数：`ε`、截断误差与它在读端的复核都走这一份定义。
fn norm(values: impl Iterator<Item = f64>) -> f64 {
    values.map(|v| v * v).sum::<f64>().sqrt()
}

/// 形状判据：`m`、`n` 非零且元素数与乘积一致；返回奇异值个数 `min(m,n)`。
fn shape(len: usize, m: usize, n: usize) -> Result<usize> {
    if m == 0 || n == 0 || m.checked_mul(n) != Some(len) {
        return Err(bad(&format!("形状不合法 m={m} n={n} len={len}")));
    }
    Ok(m.min(n))
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
    ///
    /// 两项来自**不同产物**（低秩项的截断 vs 补偿项自己的编码），所以相加不是重复计入；
    /// 反过来，把同一份误差同时填进两个字段才是——[`check`] 用「逐位往返则 `comp = 0`」
    /// 把这种填法挡住。
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
///
/// 五组规则各答一个问题——载荷写法、`ε` 的口径、补偿项的归属、无损判定的自洽、谱隙阈值——
/// 于是读一段报错就知道该找哪一类，不必把整张表从头读到尾。
pub fn check(n: &Note) -> Result<()> {
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
///
/// 写端把 [`Note`] 随载荷落盘（`axes` 的 `rank` 段），读端就有义务把它当**断言**而不是注释来查：
/// [`check`] 管内部一致性，这里接三件只有「证书 + 载荷」两方对质才能问的事——
///
/// - 这条载荷存在，就说明门确实放行了（`used`、`Residual`、`Pass` 三条同时成立）；
/// - 受认证秩落在可判域 `1 <= r < min(m,n)`；
/// - 截断误差等于载荷里 `low` 与 `rest` 算出来的那个值。
///
/// 任一条不符都是 `Invalid`：一张与载荷不符的证书比没有证书更糟，它会把「可证压缩」变成空话。
///
/// 容差按 `f32` 往返的上界给：`rest` 是 `A − L` 逐元素舍入到 `f32` 的结果，误差在 `2^24`
/// 分之一量级，所以取 `1e-6 · (|decomp| + ‖W‖_F)`——两个尺度同增同减，退化到极小矩阵时也不会
/// 把舍入当成篡改。`equal` / `lossless` 不在复核范围：它们说的是「与原始矩阵逐位一致」，
/// 而读取侧没有原始矩阵，那两条只能由写端的逐位验证承担。
pub fn audit(note: &Note, m: usize, n: usize, low: &[f32], rest: &[f32]) -> Result<()> {
    check(note)?;
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
    let truncation = norm(rest.iter().map(|&v| f64::from(v)));
    let weight = norm(
        low.iter()
            .zip(rest)
            .map(|(&l, &r)| f64::from(l) + f64::from(r)),
    );
    if (truncation - note.decomp).abs() > 1e-6 * (note.decomp.abs() + weight) {
        return Err(bad(&format!(
            "截断误差与载荷不符：证书 {}，载荷 {truncation}",
            note.decomp
        )));
    }
    Ok(())
}

fn read_f64(cur: &mut Cursor<'_>) -> Result<f64> {
    let mut bytes = [0u8; 8];
    for slot in &mut bytes {
        *slot = cur.read_u8()?;
    }
    Ok(f64::from_le_bytes(bytes))
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("ledger")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// diag(3, 2, 1, 0.5)：谱隙 1、1、0.5 都远离机器精度，判定结果可逐点预算。
    fn diag4() -> Vec<f32> {
        let mut w = vec![0.0f32; 16];
        for (i, d) in [3.0f32, 2.0, 1.0, 0.5].iter().enumerate() {
            w[i * 4 + i] = *d;
        }
        w
    }

    /// 6×6 精确秩 2：两行线性无关，其余行全零，因此 σ3 = … = σ6 = 0。
    ///
    /// 行取整数是为了让 f64 → f32 的重构精确：夹具自己不能引入误差，否则测的就不是证书。
    fn rank_two() -> Vec<f32> {
        let mut w = vec![0.0f32; 36];
        w[..6].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        w[6..12].copy_from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        w
    }

    #[test]
    fn stability_threshold_is_strict() {
        assert_eq!(stable(1.0, 0.1), Stable::Pass);
        assert_eq!(stable(0.2, 0.1), Stable::Fail);
        assert_eq!(stable(0.2, 0.099), Stable::Pass);
        assert_eq!(stable(0.2, 0.1), Stable::Fail, "等号不算严格大于");
    }

    #[test]
    fn certified_rank_stops_at_the_spectral_gap() {
        let sigma = [10.0, 5.0, 1.0, 0.0];
        assert_eq!(certified_rank(&sigma, 0.1), Some(3), "三处谱隙都过 2ε");
        assert_eq!(
            certified_rank(&sigma, 0.6),
            Some(2),
            "1 − 0 = 1 不大于 2·0.6，只好退回 r = 2"
        );
        assert_eq!(
            certified_rank(&[1.0, 1.0], 0.1),
            None,
            "没有谱隙就没有可判秩，不许瞎选一个"
        );
    }

    #[test]
    fn the_payload_audit_rejects_a_certificate_that_names_another_measurement() {
        let cert = certify(&diag4(), 4, 4).expect("可认证").expect("有谱隙");
        audit(&cert.note, 4, 4, &cert.low, &cert.rest).expect("真实载荷必须自洽");

        // 三条读端复核各自独立：改一条就得被这条载荷自己拒掉。
        for tampered in [
            Note {
                method: EpsMethod::Candidate,
                ..cert.note.clone()
            },
            Note {
                bound: Bound::Candidate,
                stable: Stable::Open,
                equal: false,
                lossless: false,
                ..cert.note.clone()
            },
            Note {
                comp: 1e-9,
                ..cert.note.clone()
            },
        ] {
            assert_eq!(
                audit(&tampered, 4, 4, &cert.low, &cert.rest)
                    .unwrap_err()
                    .code(),
                Code::Invalid,
                "{tampered:?}"
            );
        }
    }

    #[test]
    fn certify_reports_the_gated_rank_and_a_lossless_note() {
        let cert = certify(&diag4(), 4, 4).expect("可认证").expect("有谱隙");
        assert_eq!(cert.rank, 3, "diag(3,2,1,0.5) 的谱隙在这里断开");
        assert_eq!(cert.note.rank, 3);
        assert_eq!(cert.note.stable, Stable::Pass);
        assert_eq!(cert.note.mode, Mode::Residual);
        assert!(
            cert.note.equal && cert.note.lossless,
            "low + rest 必须逐位还原"
        );
        assert!((cert.note.gap - 0.5).abs() < 1e-9);
        // 显式残差补偿不计 σ_{r+1}：E_decomp 只是诊断，无损由逐位往返判定。
        assert!((cert.note.decomp - 0.5).abs() < 1e-9);
        assert!(
            cert.note.eps < 1e-12,
            "后验残差应在机器精度量级：{}",
            cert.note.eps
        );
        assert_eq!(cert.low.len(), 16);
        assert_eq!(cert.rest.len(), 16);
    }

    /// 精确秩 2 的矩阵：认证出的秩就是 2，残差到机器精度、往返逐位。
    #[test]
    fn a_low_rank_matrix_certifies_at_its_exact_rank() {
        let cert = certify(&rank_two(), 6, 6).expect("可认证").expect("有谱隙");
        assert_eq!(
            cert.rank, 2,
            "σ3 = σ4 = 0 没有谱隙，秩必须停在 2，而不是溜进噪声底"
        );
        assert!(cert.note.lossless && cert.note.equal);
        assert!(
            cert.note.decomp < 1e-10,
            "秩 2 处残差应到机器精度：{}",
            cert.note.decomp
        );
        assert!(
            cert.rest.iter().all(|x| x.abs() < 1e-5),
            "残差全是数值噪声，不该有可见量"
        );
    }

    #[test]
    fn a_matrix_without_a_gap_is_rejected() {
        // 全零：所有奇异值为 0，谱隙 0 不大于 2ε = 0 → 拒压，而不是假装无损。
        assert!(certify(&[0.0f32; 16], 4, 4).expect("形状合法").is_none());
        // min(m,n) = 1：`1 <= r < 1` 无解，秩选择不适用。
        assert!(certify(&[1.0, 2.0, 3.0], 1, 3).expect("形状合法").is_none());
    }

    #[test]
    fn a_bad_shape_is_an_error() {
        assert_eq!(
            certify(&[1.0, 2.0, 3.0], 2, 2).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(certify(&[], 0, 4).unwrap_err().code(), Code::Invalid);
        assert_eq!(
            certify(&[1.0], usize::MAX, 2).unwrap_err().code(),
            Code::Invalid
        );
    }

    /// 一张自洽的证书：L0 诊断界、残差方法、无补偿误差。
    fn valid() -> Note {
        Note {
            rank: 2,
            used: true,
            mode: Mode::Residual,
            stable: Stable::Pass,
            method: EpsMethod::Residual,
            bound: Bound::Diagnostic,
            gap: 1.0,
            eps: 0.1,
            decomp: 0.0,
            comp: 0.0,
            equal: true,
            lossless: true,
        }
    }

    #[test]
    fn note_roundtrips_through_bytes() {
        let note = valid();
        assert!(check(&note).is_ok());
        let bytes = note.to_bytes();
        let mut cur = Cursor::new(&bytes);
        let back = Note::decode(&mut cur).expect("解码");
        assert_eq!(back, note);
        assert!(cur.remaining().is_empty(), "编码不许留尾巴");
    }

    #[test]
    fn an_unknown_method_or_bound_code_is_rejected() {
        // 秩 2 只占 1 字节，于是布局是 rank|used|mode|stable|method|bound|…：方法码在 4、界码在 5。
        let bytes = valid().to_bytes();
        assert_eq!((bytes[4], bytes[5]), (1, 0), "残差方法 + L0 诊断界");
        for (offset, code) in [(4, 3u8), (5, 3u8)] {
            let mut broken = bytes.clone();
            broken[offset] = code;
            let mut cur = Cursor::new(&broken);
            assert!(Note::decode(&mut cur).is_err(), "offset {offset}");
        }
    }

    #[test]
    fn the_synthesis_error_adds_the_two_disjoint_terms() {
        // §4.8：`E_decomp` 与 `E_comp` 来自不同产物，相加就是 `ε_synth`，不重复计入。
        let note = Note {
            mode: Mode::Residual,
            decomp: 1e-3,
            comp: 2e-3,
            // 补偿项非零就不可能逐位往返。
            equal: false,
            lossless: false,
            ..valid()
        };
        assert!(check(&note).is_ok());
        assert!((note.synth() - 3e-3).abs() < 1e-18);
    }

    #[test]
    fn a_candidate_only_measurement_cannot_sign_a_pass() {
        // §4.2.1：幂迭代这类只够排序的 `ε` 不能当证书。
        let note = Note {
            method: EpsMethod::Candidate,
            ..valid()
        };
        assert!(check(&note).is_err());
        // 退到 Open 就不矛盾：它本来就没说“稳定”。
        let open = Note {
            stable: Stable::Open,
            ..note
        };
        assert!(check(&open).is_ok());
    }

    #[test]
    fn an_l0_prime_candidate_bound_cannot_sign_lossless() {
        let note = Note {
            bound: Bound::Candidate,
            ..valid()
        };
        assert!(check(&note).is_err());
        let open = Note {
            stable: Stable::Open,
            equal: false,
            lossless: false,
            ..note
        };
        assert!(check(&open).is_ok());
    }

    #[test]
    fn a_rank_choice_must_name_its_bound() {
        let note = Note {
            bound: Bound::None,
            ..valid()
        };
        assert!(check(&note).is_err());
        // 没用秩选择时，“未声明界”只是如实记录。
        let unused = Note {
            used: false,
            decomp: 0.0,
            ..note
        };
        assert!(check(&unused).is_ok());
    }

    #[test]
    fn a_compensation_error_only_belongs_to_the_residual_mode() {
        // 纯截断只丢低秩项，没有补偿产物，就没有补偿误差。
        let trunc = Note {
            mode: Mode::Trunc,
            decomp: 0.0,
            comp: 1e-9,
            ..valid()
        };
        assert!(check(&trunc).is_err());
    }

    #[test]
    fn a_bitwise_roundtrip_forbids_counting_the_compensation_twice() {
        let note = Note {
            comp: 1e-9,
            ..valid()
        };
        assert!(check(&note).is_err());
    }

    #[test]
    fn every_name_is_stable_and_distinct() {
        let methods: Vec<&str> = [
            EpsMethod::Backward,
            EpsMethod::Residual,
            EpsMethod::Candidate,
        ]
        .into_iter()
        .map(EpsMethod::as_str)
        .collect();
        assert_eq!(
            methods,
            [
                "verified_backward_error",
                "verified_residual",
                "candidate_only"
            ]
        );
        let bounds: Vec<&str> = [Bound::Diagnostic, Bound::Candidate, Bound::None]
            .into_iter()
            .map(Bound::as_str)
            .collect();
        assert_eq!(bounds, ["l0_diagnostic", "l0prime_candidate_open", "none"]);
    }

    #[test]
    fn invariant_violations_are_rejected() {
        let base = Note {
            gap: 0.1, // Pass 但 gap 不 > 2eps
            ..valid()
        };
        assert!(check(&base).is_err());
        let contradictory = Note {
            lossless: true,
            decomp: 1.0,
            equal: false,
            ..base
        };
        assert!(check(&contradictory).is_err());
    }
}
