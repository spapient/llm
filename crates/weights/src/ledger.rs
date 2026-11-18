//! 谱隙证书账本：基于自研 [`svd`] 的秩选择稳定性判定与 Eckart-Young 截断。
//!
//! 旧版用 serde 承载 60+ 个字段，其中大量是 `Option` 占位、无真实消费方。这里收敛为数学上
//! 必要的闭集字段，用 LEB128 强类型二进制编解码（替代 serde，零第三方依赖），并保留六项
//! 不变量校验。谱隙判定严格：`γ_r > 2ε` 才 Pass，等号一律 Fail；ε_alg 用实际重构残差作证，
//! 不用 `ε_mach·σ_1` 冒充。

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

/// SVD 谱隙分析：σ_r、σ_{r+1}、谱隙 γ_r 与后验残差 ε_alg。
#[derive(Clone, Copy, Debug)]
pub struct Spectral {
    /// σ_r：目标秩 r 处的奇异值。
    pub sigr: f64,
    /// σ_{r+1}：下一个奇异值。
    pub sig1: f64,
    /// 谱隙 γ_r = σ_r − σ_{r+1}。
    pub gap: f64,
    /// 后验残差 ε_alg = ‖A − UΣVᵀ‖_F。
    pub alg: f64,
}

/// 对行主序权重 `w[m*n]` 做 f64 SVD，返回谱隙与后验残差。
///
/// `r` 为 1-based 目标秩，要求 `1 <= r < min(m,n)` 以保证 σ_{r+1} 存在。
pub fn spectral(w: &[f32], m: usize, n: usize, r: usize) -> Result<Spectral> {
    check_rank(w.len(), m, n, r)?;
    let a: Vec<f64> = w.iter().map(|&x| x as f64).collect();
    let decomposed = svd::compute(&a, m, n)?;
    let k = m.min(n);
    let sigr = decomposed.sigma[r - 1];
    let sig1 = decomposed.sigma[r];
    let full = decomposed.reconstruct(k);
    let alg = a
        .iter()
        .zip(&full)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt();
    Ok(Spectral {
        sigr,
        sig1,
        gap: sigr - sig1,
        alg,
    })
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

/// Eckart-Young 秩-r 截断结果。
pub struct Trunc {
    /// 低秩项 L（行主序 `m*n` f32）。
    pub low: Vec<f32>,
    /// 残差 C = W − L（行主序 `m*n` f32）。
    pub rest: Vec<f32>,
    /// 截断误差 E_decomp = ‖W − L‖_F。
    pub decomp: f64,
}

/// 秩-r 截断：`L = U[:, :r] Σ[:r] V[:, :r]ᵀ`，残差 `C = W − L`。
pub fn truncate(w: &[f32], m: usize, n: usize, r: usize) -> Result<Trunc> {
    check_rank(w.len(), m, n, r)?;
    let a: Vec<f64> = w.iter().map(|&x| x as f64).collect();
    let decomposed = svd::compute(&a, m, n)?;
    let low64 = decomposed.reconstruct(r);
    let low: Vec<f32> = low64.iter().map(|&l| l as f32).collect();
    let rest: Vec<f32> = a
        .iter()
        .zip(&low64)
        .map(|(orig, l)| (orig - l) as f32)
        .collect();
    let decomp = a
        .iter()
        .zip(&low64)
        .map(|(orig, low)| (orig - low) * (orig - low))
        .sum::<f64>()
        .sqrt();
    Ok(Trunc { low, rest, decomp })
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
    /// 谱隙 γ_r。
    pub gap: f64,
    /// 总扰动 ε = ε_obs + ε_alg。
    pub eps: f64,
    /// 截断误差 E_decomp。
    pub decomp: f64,
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
        out.put(&self.gap.to_le_bytes());
        out.put(&self.eps.to_le_bytes());
        out.put(&self.decomp.to_le_bytes());
        out.put(&[u8::from(self.equal)]);
        out.put(&[u8::from(self.lossless)]);
    }

    /// 强类型解码；字段不符显式报错。逐段按 `Note::encode` 的顺序读，段内报错点与逐字段读一致。
    pub fn decode(cur: &mut Cursor<'_>) -> Result<Self> {
        let (rank, used, mode, stable) = head(cur)?;
        let ([gap, eps, decomp], [equal, lossless]) = tail(cur)?;
        Ok(Self {
            rank,
            used,
            mode,
            stable,
            gap,
            eps,
            decomp,
            equal,
            lossless,
        })
    }

    /// 序列化为字节。
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode(&mut out);
        out
    }
}

/// 标识段：秩、是否使用秩选择、重构模式码、稳定性判定码。
fn head(cur: &mut Cursor<'_>) -> Result<(u32, bool, Mode, Stable)> {
    let rank = cur.read::<u32>()?;
    let flags = cur.take(3)?;
    let mode = Mode::from_code(flags[1])?;
    let stable = Stable::from_code(flags[2])?;
    Ok((rank, flags[0] != 0, mode, stable))
}

/// 实数与标志段：三个 f64（谱隙、总扰动、截断误差）小端连排，其后是两条布尔（非 0 即真）。
fn tail(cur: &mut Cursor<'_>) -> Result<([f64; 3], [bool; 2])> {
    let mut reals = [0.0f64; 3];
    for value in &mut reals {
        *value = read_f64(cur)?;
    }
    let mut flags = [false; 2];
    for flag in &mut flags {
        *flag = cur.read_u8()? != 0;
    }
    Ok((reals, flags))
}

/// 六项不变量校验（收敛为可执行的四条核心约束，逐条对应旧版 F1–F4/A2/D2）。
pub fn check(n: &Note) -> Result<()> {
    let rules = [
        (
            [n.gap, n.eps, n.decomp].iter().all(|v| v.is_finite()),
            "存在非有限值",
        ),
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
        (n.gap >= 0.0, "谱隙不能为负"),
    ];
    match rules.iter().find(|(holds, _)| !*holds) {
        Some((_, why)) => Err(bad(why)),
        None => Ok(()),
    }
}

/// 由谱隙分析与秩截断合成一张证书：`eps` 是总扰动（观测 + 算法），`used` 是否采用秩选择，
/// `mode` 冻结重构模式（§4.4 第 3 步）。`equal` 判据与五轴 rank 轴一致（`low + rest` 与原始
/// 逐位相等），无损判据随模式取不同误差项：纯截断看 `E_decomp`，残差补偿看逐位往返，随后过
/// [`check`] 自证。
pub fn note(
    w: &[f32],
    m: usize,
    n: usize,
    r: usize,
    eps: f64,
    used: bool,
    mode: Mode,
) -> Result<Note> {
    let s = spectral(w, m, n, r)?;
    let t = truncate(w, m, n, r)?;
    let equal = w
        .iter()
        .zip(&t.low)
        .zip(&t.rest)
        .all(|((x, low), rest)| (low + rest).to_bits() == x.to_bits());
    let note = Note {
        rank: r as u32,
        used,
        mode,
        stable: stable(s.gap, eps),
        gap: s.gap,
        eps,
        decomp: t.decomp,
        equal,
        lossless: match mode {
            Mode::Trunc => t.decomp == 0.0,
            Mode::Residual => equal,
        },
    };
    check(&note)?;
    Ok(note)
}

fn check_rank(len: usize, m: usize, n: usize, r: usize) -> Result<()> {
    let fits = [
        m != 0,
        n != 0,
        m.checked_mul(n) == Some(len),
        (1..m.min(n)).contains(&r),
    ];
    if fits.iter().all(|ok| *ok) {
        return Ok(());
    }
    Err(bad(&format!(
        "要求 1 <= r < min(m,n)，实际 m={m} n={n} r={r}"
    )))
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

    fn diag4() -> Vec<f32> {
        let mut w = vec![0.0f32; 16];
        for (i, d) in [3.0f32, 2.0, 1.0, 0.5].iter().enumerate() {
            w[i * 4 + i] = *d;
        }
        w
    }

    #[test]
    fn spectral_matches_singular_values() {
        let s = spectral(&diag4(), 4, 4, 2).expect("谱分析");
        assert!((s.sigr - 2.0).abs() < 1e-9);
        assert!((s.sig1 - 1.0).abs() < 1e-9);
        assert!((s.gap - 1.0).abs() < 1e-9);
        assert!(s.alg < 1e-6, "后验残差应接近 0，实际 {}", s.alg);
    }

    #[test]
    fn stability_threshold_is_strict() {
        assert_eq!(stable(1.0, 0.1), Stable::Pass);
        assert_eq!(stable(0.2, 0.1), Stable::Fail);
        assert_eq!(stable(0.2, 0.099), Stable::Pass);
    }

    #[test]
    fn truncate_r2_tail() {
        let t = truncate(&diag4(), 4, 4, 2).expect("截断");
        assert!((t.low[0] - 3.0).abs() < 1e-6);
        assert!((t.low[5] - 2.0).abs() < 1e-6);
        assert!(t.low[10].abs() < 1e-6);
        assert!((t.rest[10] - 1.0).abs() < 1e-6);
        assert!((t.decomp - 1.25f64.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn note_roundtrips_through_bytes() {
        let note = Note {
            rank: 2,
            used: true,
            mode: Mode::Residual,
            stable: Stable::Pass,
            gap: 1.0,
            eps: 0.1,
            decomp: 0.0,
            equal: true,
            lossless: true,
        };
        assert!(check(&note).is_ok());
        let bytes = note.to_bytes();
        let mut cur = Cursor::new(&bytes);
        let back = Note::decode(&mut cur).expect("解码");
        assert_eq!(back, note);
    }

    #[test]
    fn invariant_violations_are_rejected() {
        let base = Note {
            rank: 2,
            used: true,
            mode: Mode::Residual,
            stable: Stable::Pass,
            gap: 0.1, // Pass 但 gap 不 > 2eps
            eps: 0.1,
            decomp: 0.0,
            equal: true,
            lossless: true,
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

    #[test]
    fn bad_rank_is_an_error() {
        assert!(spectral(&diag4(), 4, 4, 4).is_err());
        assert!(truncate(&diag4(), 4, 4, 0).is_err());
    }

    #[test]
    fn note_synthesizes_and_self_checks() {
        let residual = note(&diag4(), 4, 4, 2, 0.1, true, Mode::Residual).expect("证书");
        assert_eq!(residual.rank, 2);
        assert!(residual.used);
        assert_eq!(residual.mode, Mode::Residual);
        assert_eq!(residual.stable, Stable::Pass); // gap = 2 - 1 = 1 > 2 * 0.1
        assert!((residual.gap - 1.0).abs() < 1e-9);
        assert!((residual.decomp - 1.25f64.sqrt()).abs() < 1e-5);
        // 显式残差补偿：low + rest 逐位还原 diag(3,2,1,0.5)，E_comp=0 ⇒ 无损（不重复计 σ_{r+1}）。
        assert!(residual.equal);
        assert!(residual.lossless);

        let trunc = note(&diag4(), 4, 4, 2, 0.1, true, Mode::Trunc).expect("证书");
        assert_eq!(trunc.mode, Mode::Trunc);
        // 纯截断：E_decomp = ‖diag(0,0,1,0.5)‖_F > 0 ⇒ 有损。
        assert!(!trunc.lossless);
    }

    /// 矩阵本就秩 ≤ r 时，纯截断无残差，判无损。
    #[test]
    fn trunc_mode_is_lossless_when_rank_is_exact() {
        let mut w = vec![0.0f32; 16];
        w[0] = 3.0;
        w[5] = 2.0;
        let n = note(&w, 4, 4, 2, 0.1, true, Mode::Trunc).expect("证书");
        assert!(n.lossless);
    }
}
