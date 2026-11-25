//! 谱隙证书账本：L0 判定、秩选择与证书同源，只有一个入口 [`certify`]。

use crate::bad;
use diagnostic::{Code, Error, Result};
use storage::leb::Cursor;
use storage::sink::Sink;

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
    let s = Spectral::open(w, m, n)?;
    // ε 只有算法项：后验残差是本次分解的实测值，不是机器精度的估计。
    let eps = residual(&s.a, &s.svd.reconstruct(k));
    let Some(rank) = certified_rank(&s.svd.sigma, eps) else {
        return Ok(None);
    };
    assemble(w, &s.svd, (m, n), rank, eps).map(Some)
}

/// 受认证秩处的产物与自证证书：低秩项、显式残差、逐位往返与 [`Note`] 都出自同一次分解。
fn assemble(
    w: &[f32],
    d: &Svd,
    (m, n): (usize, usize),
    rank: usize,
    eps: f64,
) -> Result<Certified> {
    let (left, right) = d.factors(rank);
    let low64 = factor_low(&left, &right, (m, n), rank);
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
        decomp: residual32(w, &low),
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
        self.encode_fields(out);
    }

    /// 强类型解码；字段不符显式报错。字段顺序与 [`Note::encode`] 同源。
    pub fn decode(cur: &mut Cursor<'_>) -> Result<Self> {
        let (rank, used, mode, stable, method, bound, gap, eps, decomp, comp, equal, lossless) =
            Self::decode_fields(cur)?;
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

// 证书线上字段的唯一声明：秩、五个标志/闭集码、四个小端 f64、两个布尔。
wire_fields! {
    Note {
        rank: leb_u32,
        used: bool,
        mode: enum_u8<Mode>,
        stable: enum_u8<Stable>,
        method: enum_u8<EpsMethod>,
        bound: enum_u8<Bound>,
        gap: f64,
        eps: f64,
        decomp: f64,
        comp: f64,
        equal: bool,
        lossless: bool,
    }
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
    let truncation = norm(rest.iter().map(|&v| f64::from(v)));
    let weight = norm(
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

// —— 零依赖 f64 奇异值分解：单边 Jacobi SVD（one-sided Jacobi） ——

/// 最大扫描轮数：典型矩阵几轮即收敛，硬上限防止病态输入死循环。
const MAX_SWEEPS: usize = 100;

/// 收敛阈值：列内积相对范数小于此即视为已正交。
const TOL: f64 = 1e-14;

/// SVD 结果：奇异值按降序，配套左右奇异向量。
#[derive(Clone, Debug)]
pub struct Svd {
    /// 奇异值 `σ_1 ≥ σ_2 ≥ …`。
    pub sigma: Vec<f64>,
    /// 左奇异向量，行主序 `m × k`（第 j 列是对应 `sigma[j]` 的左向量）。
    pub u: Vec<f64>,
    /// 右奇异向量，行主序 `n × k`。
    pub v: Vec<f64>,
    m: usize,
    n: usize,
}

impl Svd {
    /// 左向量矩阵列数（= 非零/全部奇异值个数，取 n 与 m 的较小者由调用语义保证）。
    #[must_use]
    pub fn rank(&self) -> usize {
        self.sigma.len()
    }

    /// 矩阵形状 `(m, n)`：因子化存储要按它算 `UΣ` 与 `Vᵀ` 的行宽。
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    /// 秩-r 截断重构（行主序 `m×n`）：`U[:, :r] Σ[:r] V[:, :r]ᵀ`。
    #[must_use]
    pub fn reconstruct(&self, r: usize) -> Vec<f64> {
        let r = r.min(self.sigma.len());
        let mut out = vec![0.0f64; self.m * self.n];
        for i in 0..self.m {
            for j in 0..self.n {
                let mut sum = 0.0;
                for k in 0..r {
                    // U(i,k)·σ_k·V(j,k)
                    sum += self.u[i * self.sigma.len() + k]
                        * self.sigma[k]
                        * self.v[j * self.sigma.len() + k];
                }
                out[i * self.n + j] = sum;
            }
        }
        out
    }

    /// 秩-`r` 的两个因子：`left[i,j] = u[i,j]·σ_j`（`m×r`）、`right[j,c] = v[c,j]`（`r×n`）。
    pub(crate) fn factors(&self, rank: usize) -> (Vec<f64>, Vec<f64>) {
        let ((m, n), k) = (self.shape(), self.sigma.len());
        let mut left = Vec::with_capacity(m * rank);
        for i in 0..m {
            left.extend((0..rank).map(|j| self.u[i * k + j] * self.sigma[j]));
        }
        let mut right = Vec::with_capacity(rank * n);
        for j in 0..rank {
            right.extend((0..n).map(|c| self.v[c * k + j]));
        }
        (left, right)
    }
}

/// 信号秩选择的相对谱隙阈值。
pub const REL_GAP: f64 = 0.2;

/// 有损补偿侧的信号秩：返回**噪声底之前**的最大秩 `r`，即 `(σ_r − σ_{r+1}) / σ_1 > REL_GAP`
/// 的最后一点。
#[must_use]
pub fn signal_rank(sigma: &[f64]) -> Option<usize> {
    let floor = REL_GAP * sigma.first()?;
    (1..sigma.len())
        .rev()
        .find(|&r| sigma[r - 1] - sigma[r] > floor)
}

/// 对行主序 `a[m*n]` 做 SVD。
pub fn compute_svd(a: &[f64], m: usize, n: usize) -> Result<Svd> {
    if !fits(a.len(), m, n) {
        return Err(Error::new(
            Code::Invalid,
            format!("SVD: 形状不合法 m={m} n={n} len={}", a.len()),
        )
        .with("svd::compute"));
    }
    let k = m.min(n);
    let (mut u, mut v) = column_major(a, m, n);
    orthogonalize(&mut u, &mut v, m, n);
    let sigma = singular_values(&mut u, m, n);
    let (sigma, u, v) = sorted_columns(&sigma, &u, &v, m, n, k);
    Ok(Svd { sigma, u, v, m, n })
}

/// 形状自洽：零维与长度不符是同一类调用错误，分开写只会让 `compute` 多两条分支。
fn fits(len: usize, m: usize, n: usize) -> bool {
    m != 0 && n != 0 && len == m * n
}

/// 工作矩阵 `u` 转列主序、`v` 置单位阵：Jacobi 全程都在列对上做点积/旋转，列连续存放才能
/// 命中缓存（行主序下 n=1024 的列访问步长 8KiB，每次都是缓存未命中）。输出仍按约定行主序。
fn column_major(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut u = vec![0.0f64; m * n];
    for j in 0..n {
        for i in 0..m {
            u[j * m + i] = a[i * n + j];
        }
    }
    let mut v = vec![0.0f64; n * n];
    for j in 0..n {
        v[j * n + j] = 1.0;
    }
    (u, v)
}

/// 反复扫描直到收敛；硬上限防止病态输入死循环。
fn orthogonalize(u: &mut [f64], v: &mut [f64], m: usize, n: usize) {
    for _ in 0..MAX_SWEEPS {
        if sweep(u, v, m, n) {
            break;
        }
    }
}

/// 一轮扫描：逐个列对旋转，返回本轮是否全部已正交。
fn sweep(u: &mut [f64], v: &mut [f64], m: usize, n: usize) -> bool {
    let mut converged = true;
    for p in 0..n {
        for q in (p + 1)..n {
            if rotate_pair(u, v, m, n, p, q) {
                converged = false;
            }
        }
    }
    converged
}

/// 一对列：已正交返回 `false`，做了旋转返回 `true`。`u`/`v` 均为列主序。
fn rotate_pair(u: &mut [f64], v: &mut [f64], m: usize, n: usize, p: usize, q: usize) -> bool {
    let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
    let up = p * m;
    let uq = q * m;
    for row in 0..m {
        let x = u[up + row];
        let y = u[uq + row];
        alpha += x * x;
        beta += y * y;
        gamma += x * y;
    }
    if gamma.abs() <= TOL * (alpha * beta).sqrt() {
        return false;
    }
    let (c, s) = rotation(alpha, beta, gamma);
    rotate_columns(u, m, p, q, c, s);
    rotate_columns(v, n, p, q, c, s);
    true
}

/// 对列主序 `mat` 的第 p、q 列施加旋转 `(c, s)`；`stride` 是每列长度（行数）。
fn rotate_columns(mat: &mut [f64], stride: usize, p: usize, q: usize, c: f64, s: f64) {
    let bp = p * stride;
    let bq = q * stride;
    for row in 0..stride {
        let x = mat[bp + row];
        let y = mat[bq + row];
        mat[bp + row] = c * x - s * y;
        mat[bq + row] = s * x + c * y;
    }
}

/// 列长度即奇异值；随后把 U 的列归一化。`u` 为列主序。
fn singular_values(u: &mut [f64], m: usize, n: usize) -> Vec<f64> {
    let mut sigma = vec![0.0f64; n];
    for (j, value) in sigma.iter_mut().enumerate() {
        *value = column_norm(u, m, j);
    }
    for (j, &sigma) in sigma.iter().enumerate() {
        normalize_column(u, m, j, sigma);
    }
    sigma
}

/// 一列的欧氏长度。
fn column_norm(u: &[f64], m: usize, j: usize) -> f64 {
    let base = j * m;
    let mut norm = 0.0;
    for i in 0..m {
        let x = u[base + i];
        norm += x * x;
    }
    norm.sqrt()
}

/// 列除以它的长度；零列保持为零。
fn normalize_column(u: &mut [f64], m: usize, j: usize, sigma: f64) {
    if sigma <= 0.0 {
        return;
    }
    let base = j * m;
    for i in 0..m {
        u[base + i] /= sigma;
    }
}

/// 按奇异值降序重排 U、V 的列，只保留前 k 列。输入 `u`/`v` 为列主序，输出转回行主序。
fn sorted_columns(
    sigma: &[f64],
    u: &[f64],
    v: &[f64],
    m: usize,
    n: usize,
    k: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| sigma[b].total_cmp(&sigma[a]));
    let sorted_sigma = order.iter().map(|&j| sigma[j]).take(k).collect();
    let mut sorted_u = vec![0.0f64; m * k];
    let mut sorted_v = vec![0.0f64; n * k];
    for (new_col, &old_col) in order.iter().take(k).enumerate() {
        for i in 0..m {
            sorted_u[i * k + new_col] = u[old_col * m + i];
        }
        for i in 0..n {
            sorted_v[i * k + new_col] = v[old_col * n + i];
        }
    }
    (sorted_sigma, sorted_u, sorted_v)
}

/// 由列 Gram 参数算 Jacobi 旋转系数 `(c, s)`。
fn rotation(alpha: f64, beta: f64, gamma: f64) -> (f64, f64) {
    let tau = (beta - alpha) / (2.0 * gamma);
    let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
    let c = 1.0 / (1.0 + t * t).sqrt();
    (c, c * t)
}

// —— 谱证书共享数学：范数、残差、因子化低秩项与谱隙，账本与补偿侧同出一门 ——

/// Frobenius 范数：`ε`、截断误差与它在读端的复核都走这一份定义。
pub(crate) fn norm(values: impl Iterator<Item = f64>) -> f64 {
    values.map(|v| v * v).sum::<f64>().sqrt()
}

/// 后验残差 `‖A − B‖_F`（f64）：`ε_alg` 的实测值。
pub(crate) fn residual(a: &[f64], b: &[f64]) -> f64 {
    norm(a.iter().zip(b).map(|(x, y)| x - y))
}

/// 落盘精度上的差范数（f32）：`decomp` 记的是 f32 低秩项与原矩阵之间还剩多少，与 `rest` 同源。
pub(crate) fn residual32(a: &[f32], b: &[f32]) -> f64 {
    norm(a.iter().zip(b).map(|(&x, &y)| f64::from(x) - f64::from(y)))
}

/// 截断处上方那道缝：`σ_r − σ_{r+1}`；已到最后一个奇异值时退化成 `σ_r`（下面没有值可供相减）。
pub(crate) fn spectral_gap(sigma: &[f64], r: usize) -> f64 {
    match sigma.get(r) {
        Some(&below) => sigma[r - 1] - below,
        None => sigma[r - 1],
    }
}

/// `left · right` 的 `m×n` 重构（行主序 f64）：写端与读端共用，因子化低秩项只算一份。
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

/// 一次谱分解的共享产物：f64 工作矩阵、SVD 与整体范数同出一门；秩选择与残差精度由调用方定。
pub(crate) struct Spectral {
    /// 原矩阵升到 f64 的工作副本。
    pub(crate) a: Vec<f64>,
    /// 这次分解。
    pub(crate) svd: Svd,
    /// 整体 Frobenius 范数 `‖W‖_F`。
    pub(crate) norm: f64,
}

impl Spectral {
    /// f32 → f64 → 一次 SVD → 整体范数：账本与补偿侧共用的分解入口，转换与分解只写一份。
    pub(crate) fn open(w: &[f32], m: usize, n: usize) -> Result<Self> {
        let a: Vec<f64> = w.iter().map(|&x| f64::from(x)).collect();
        let svd = compute_svd(&a, m, n)?;
        let total = norm(a.iter().copied());
        Ok(Self {
            a,
            svd,
            norm: total,
        })
    }
}

#[cfg(test)]
#[path = "../tests/ptq_svd.rs"]
mod svd_tests;

#[cfg(test)]
#[path = "../tests/ledger.rs"]
mod tests;
