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
        gap,
        eps,
        decomp: residual(a, &low64),
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
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
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

/// 不变量校验（可执行的四条核心约束，逐条对应旧版 F1–F4/A2/D2）。
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
        assert!(cur.remaining().is_empty(), "编码不许留尾巴");
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
}
