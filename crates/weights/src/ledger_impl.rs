//! [`ledger`](crate::ledger) 的机制层：同一次 f64 SVD 分解出受认证的秩、低秩项、显式残差与自证
//! [`Note`]，再加上码流的强类型编解码与校验规则。公开面只有 [`certify`]、[`check`]、[`audit`] 三个
//! 入口，由 [`ledger`](crate::ledger) 原样再导出。

use diagnostic::Result;

use crate::leb::{self, Cursor};
use crate::sink::Sink;
use crate::svd;

use crate::ledger::{
    bad, certified_rank, factor_low, stable, Bound, Certified, EpsMethod, Mode, Note, Stable,
};

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
    assemble(w, &d, (m, n), rank, eps).map(Some)
}

/// 受认证秩处的产物与自证证书：低秩项、显式残差、逐位往返与 [`Note`] 都出自同一次分解。
fn assemble(
    w: &[f32],
    d: &svd::Svd,
    (m, n): (usize, usize),
    rank: usize,
    eps: f64,
) -> Result<Certified> {
    let (left, right) = factors(d, rank);
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
    check(&note)?;
    Ok(Certified {
        rank,
        low,
        rest,
        left,
        right,
        note,
    })
}

/// 秩-`rank` 的两个因子：`left[i,j] = u[i,j]·σ_j`（`m×r`）、`right[j,c] = v[c,j]`（`r×n`）。
///
/// 这是 `UΣ` 与 `Vᵀ` 的显式形式，落盘只需 `(m+n)·r` 个 `f64`；重构交给 [`factor_low`]，
/// 写端与读端因此只有一份乘积算式。
fn factors(d: &svd::Svd, rank: usize) -> (Vec<f64>, Vec<f64>) {
    let ((m, n), k) = (d.shape(), d.sigma.len());
    let mut left = Vec::with_capacity(m * rank);
    for i in 0..m {
        left.extend((0..rank).map(|j| d.u[i * k + j] * d.sigma[j]));
    }
    let mut right = Vec::with_capacity(rank * n);
    for j in 0..rank {
        right.extend((0..n).map(|c| d.v[c * k + j]));
    }
    (left, right)
}

/// 后验残差 `‖A − B‖_F`：`ε_alg` 的实测值。
fn residual(a: &[f64], b: &[f64]) -> f64 {
    norm(a.iter().zip(b).map(|(x, y)| x - y))
}

/// 落盘精度上的差范数：`decomp` 记的是「`f32` 低秩项与原矩阵之间还剩多少」，与 `rest` 同源。
fn residual32(a: &[f32], b: &[f32]) -> f64 {
    norm(a.iter().zip(b).map(|(&x, &y)| f64::from(x) - f64::from(y)))
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
