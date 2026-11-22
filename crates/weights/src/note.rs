//! 账本 [`Note`] 的编解码与自证：五个闭集字段怎么落字节、[`check`] 怎么逐条把关。

use diagnostic::Result;

use crate::bad;
use crate::ledger::{Bound, EpsMethod, Mode, Note, Stable, norm};
use wire::leb::{self, Cursor};
use wire::sink::Sink;

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

fn read_f64(cur: &mut Cursor<'_>) -> Result<f64> {
    let mut bytes = [0u8; 8];
    for slot in &mut bytes {
        *slot = cur.read_u8()?;
    }
    Ok(f64::from_le_bytes(bytes))
}
