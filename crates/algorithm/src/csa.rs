//! V4 CSA/HCA 超长注意力（§5.8.3）：序列分块压缩 + 稀疏/稠密注意力。
//!
//! CSA：每 `m=4` token 压成 1 条，再 DSA top-k（Flash 512 / Pro 1024）；HCA：每 `m'=128` token 压成
//! 1 条，稠密注意。有效长 `n/m`、`n/m'`（数学验证 41）。这是 α 通道的**序列压缩注意力**，与 MLA
//! 潜在维压缩分列，禁止写成 `O(n)` 扫描（Mamba-3 轴）。
//!
//! 纯数学函数统一走 [`op::scan::Scalar`] 泛型（与 [`crate::mix`] / [`crate::policy`] / [`crate::mix::pulse`]
//! 同一闭集）：`f32` 供前向、`f64` 供验证对拍，不因精度复制两份实现。
//!
//! [`BlockTable`] 是这套公式的**执行侧**：压缩 K/V 进表，再按因果口径寻址（整块 + 滑窗），
//! 一步就能算出注意力输出。没有它，压缩与 top-k 只是没人调用的纯函数。

use core::cmp::Ordering;
use diagnostic::{Code, Error, Result};
use op::scan::Scalar;

/// CSA 分块因子。
pub const CSA_M: usize = 4;
/// HCA 分块因子（`m' ≫ m`）。
pub const HCA_M: usize = 128;
/// DSA top-k：Flash 档。
pub const CSA_TOPK_FLASH: usize = 512;
/// DSA top-k：Pro 档。
pub const CSA_TOPK_PRO: usize = 1024;
/// 滑窗宽（补块内因果，与 `m'` 同宽）。
pub const WIN: usize = 128;

/// 有效序列长 `n/m`（CSA）或 `n/m'`（HCA）。
pub const fn effective_len(n: usize, m: usize) -> usize {
    n.div_ceil(m)
}

/// 分块压缩：每 `m` 行均值池化成 1 行（行主序 `[n, d] → [⌈n/m⌉, d]`）。
pub fn compress<S: Scalar>(seq: &[S], n: usize, d: usize, m: usize) -> Vec<S> {
    assert_eq!(seq.len(), n * d, "seq 长度与形状不符");
    assert!(m > 0, "分块因子必须非零");
    let blocks = n.div_ceil(m);
    let mut out = vec![S::ZERO; blocks * d];
    for b in 0..blocks {
        let start = b * m;
        let end = (start + m).min(n);
        let count = S::from_f64((end - start) as f64);
        for j in 0..d {
            let sum = (start..end).fold(S::ZERO, |acc, i| acc.add(seq[i * d + j]));
            out[b * d + j] = sum.div(count);
        }
    }
    out
}

/// DSA top-k：每行只保留最大的 `k` 个分数，其余置 `−∞`（softmax 后为 0，实现稀疏注意力）。
pub fn dsa_topk<S: Scalar>(scores: &mut [S], rows: usize, cols: usize, k: usize) {
    debug_assert_eq!(scores.len(), rows * cols);
    debug_assert!(k <= cols);
    for r in 0..rows {
        let row = &mut scores[r * cols..(r + 1) * cols];
        if k >= cols {
            continue;
        }
        // 降序选择：index k−1 是第 k 大，`[0..k]` 为 top-k，`[k..]` 为弃项。
        row.select_nth_unstable_by(k - 1, |a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        for v in &mut row[k..] {
            *v = S::NEG_INFINITY;
        }
    }
}

/// 稠密注意力：每行 softmax 归一化（HCA 的稠密路径，作为 DSA 的对照）。
pub fn dense_softmax<S: Scalar>(scores: &mut [S], rows: usize, cols: usize) {
    debug_assert_eq!(scores.len(), rows * cols);
    for r in 0..rows {
        softmax_row(&mut scores[r * cols..(r + 1) * cols]);
    }
}

/// 单行 softmax：先减最大值再指数，和为零时保持原值（全 `-inf` 的退化行）。
///
/// `pub(crate)` 而不是私有：[`crate::mla`] 的潜在注意力要在同一口径上归一化，
/// 再写一份就等于给 softmax 两个定义。
pub(crate) fn softmax_row<S: Scalar>(row: &mut [S]) {
    let max = row
        .iter()
        .copied()
        .fold(S::NEG_INFINITY, |a, b| if b > a { b } else { a });
    let mut sum = S::ZERO;
    for v in row.iter_mut() {
        *v = v.sub(max).exp();
        sum = sum.add(*v);
    }
    if sum > S::ZERO {
        for v in row.iter_mut() {
            *v = v.div(sum);
        }
    }
}

/// Sinkhorn 双随机归一化（方阵）：行/列交替缩放，收敛后行和、列和均 ≈ 1，且 `‖B‖₂ ≤ 1`
/// （数学验证 41）。返回是否在 `iters` 轮内收敛到 `tol`。
pub fn sinkhorn<S: Scalar>(mat: &mut [S], n: usize, iters: usize, tol: S) -> bool {
    debug_assert_eq!(mat.len(), n * n);
    for _ in 0..iters {
        normalize_rows(mat, n);
        normalize_cols(mat, n);
    }
    is_doubly_stochastic(mat, n, tol)
}

/// 行归一：每行除以行和（行和为零时保持原值）。
fn normalize_rows<S: Scalar>(mat: &mut [S], n: usize) {
    for r in 0..n {
        let sum = mat[r * n..(r + 1) * n]
            .iter()
            .copied()
            .fold(S::ZERO, |acc, v| acc.add(v));
        if sum > S::ZERO {
            for v in &mut mat[r * n..(r + 1) * n] {
                *v = v.div(sum);
            }
        }
    }
}

/// 列归一：每列除以列和（列和为零时保持原值）。
fn normalize_cols<S: Scalar>(mat: &mut [S], n: usize) {
    for c in 0..n {
        let sum = (0..n).fold(S::ZERO, |acc, r| acc.add(mat[r * n + c]));
        if sum > S::ZERO {
            for r in 0..n {
                mat[r * n + c] = mat[r * n + c].div(sum);
            }
        }
    }
}

/// 是否（近似）双随机：行和、列和都 ≈ 1。
pub fn is_doubly_stochastic<S: Scalar>(mat: &[S], n: usize, tol: S) -> bool {
    debug_assert_eq!(mat.len(), n * n);
    (0..n).all(|r| {
        let row = mat[r * n..(r + 1) * n]
            .iter()
            .copied()
            .fold(S::ZERO, |acc, v| acc.add(v));
        let col = (0..n).fold(S::ZERO, |acc, i| acc.add(mat[i * n + r]));
        row.sub(S::ONE).abs() <= tol && col.sub(S::ONE).abs() <= tol
    })
}

/// 块表：一段 K/V 按块因子压缩后的键值表，以及它的因果寻址口径。
///
/// [`compress`] 只回答"怎么压"，这张表回答"压完之后谁来用"——§5.8.3 里 α 通道的检索面。
/// 原始行与压缩行同时留着，同一份数据因此能给出两种注意力：压缩块（检索）与滑窗原始行
/// （补块内因果）。寻址两级缺一不可：
///
/// - **整块**：只取完全落在查询位置之前的块（`(b+1)·m ≤ t`）。块里混了未来行，整块可见就等于
///   提前看见未来；
/// - **滑窗**：再补 `[t+1−win, t]` 的原始行。块压缩把块内的相对顺序抹平，这一窗把它补回来。
///
/// `m = 1` 且 `win = 1` 时可见槽恰好是 `0..=t`，输出与稠密因果注意力一致（单测钉住）。`m` 取
/// [`CSA_M`]、`win` 取 [`WIN`]、`topk = Some(CSA_TOPK_FLASH)` 即 CSA；`m` 取 [`HCA_M`] 且
/// `topk = None` 即 HCA 的稠密块。
#[derive(Debug, Clone, PartialEq)]
pub struct BlockTable<S: Scalar = f32> {
    dim: usize,
    /// 块因子 `m`。
    m: usize,
    /// 滑窗宽 `win`。
    win: usize,
    seq: usize,
    /// 原始行，行主序 `[seq, dim]`：滑窗从这一份里取。
    keys: Vec<S>,
    values: Vec<S>,
    /// 压缩行，行主序 `[⌈seq/m⌉, dim]`：整块从这一份里取。
    block_keys: Vec<S>,
    block_values: Vec<S>,
}

/// 块表报错的出处：几何与查询越界都在构造/寻址期失败，不留到数值阶段。
const SITE: &str = "algorithm::csa::BlockTable";

/// 可寻址宽度 `seq × dim`：任一几何为零时算不出宽度，乘法溢出同样拒绝。
///
/// 单独一函数是因为它是「表能不能建」的第一道判据：宽度是后面长度校验与压缩切块的共同依据，
/// 与「K/V 给对没有」是两件事。
fn geometry(seq: usize, dim: usize, m: usize, win: usize) -> Result<usize> {
    let width = seq.checked_mul(dim).ok_or_else(|| {
        Error::new(Code::Invalid, format!("seq × dim 溢出（{seq} × {dim}）")).with(SITE)
    })?;
    if dim.min(m).min(win) == 0 {
        return Err(Error::new(
            Code::Invalid,
            format!("块表几何必须非零（dim = {dim}，m = {m}，win = {win}）"),
        )
        .with(SITE));
    }
    Ok(width)
}

/// K/V 两段都必须正好是 `width` 个元素：长度不符时压缩会按错几何切块。
fn lengths<S>(keys: &[S], values: &[S], width: usize) -> Result<()> {
    if [keys.len(), values.len()] == [width; 2] {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!("K/V 长度必须都是 seq × dim = {width}"),
    )
    .with(SITE))
}

impl<S: Scalar> BlockTable<S> {
    /// 由 `seq` 行 K/V 建表：`keys` / `values` 都是行主序 `[seq, dim]`，压缩按 [`compress`]。
    ///
    /// `dim`、`m`、`win` 必须非零，K/V 长度必须是 `seq · dim`；任何一条不成立都是
    /// `Code::Invalid`——块表带着错的几何活下去，之后每一次寻址都错。
    pub fn new(
        keys: &[S],
        values: &[S],
        seq: usize,
        dim: usize,
        m: usize,
        win: usize,
    ) -> Result<Self> {
        let width = geometry(seq, dim, m, win)?;
        lengths(keys, values, width)?;
        Ok(Self {
            dim,
            m,
            win,
            seq,
            block_keys: compress(keys, seq, dim, m),
            block_values: compress(values, seq, dim, m),
            keys: keys.to_vec(),
            values: values.to_vec(),
        })
    }

    /// 序列长。
    pub fn seq(&self) -> usize {
        self.seq
    }

    /// 单行宽度。
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 块因子 `m`。
    pub fn factor(&self) -> usize {
        self.m
    }

    /// 滑窗宽。
    pub fn window(&self) -> usize {
        self.win
    }

    /// 块数 `⌈seq/m⌉`，与 [`effective_len`] 同式。
    pub fn blocks(&self) -> usize {
        self.block_keys.len() / self.dim
    }

    /// 压缩后的表 `(K, V)`，行主序 `[blocks, dim]`。
    pub fn compressed(&self) -> (&[S], &[S]) {
        (&self.block_keys, &self.block_values)
    }

    /// 查询位置 `t` 的可见槽：完全落在 `t` 之前的整块数，与滑窗原始行的起点。
    ///
    /// 两者共同决定槽序（先整块、后滑窗行），也是因果性的全部依据：`t` 之后的行既不在整块里
    /// （块必须整块在 `t` 之前），也不在窗口里。
    pub fn visible(&self, t: usize) -> (usize, usize) {
        let whole = (t / self.m).min(self.blocks());
        (whole, t.saturating_sub(self.win - 1))
    }

    /// 因果注意力：查询 `q` 在位置 `t` 的输出。
    ///
    /// 槽序先压缩块、后滑窗行；`topk` 只保留分数最大的 `k` 个槽（CSA 取
    /// [`CSA_TOPK_FLASH`]），`None`、`Some(0)` 或 `k ≥ 槽数` 都是稠密（没有可丢的槽）。
    pub fn attend(&self, q: &[S], t: usize, topk: Option<usize>) -> Result<Vec<S>> {
        if t >= self.seq || q.len() != self.dim {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "位置 {t} 或查询宽度 {} 越界（seq = {}，dim = {}）",
                    q.len(),
                    self.seq,
                    self.dim
                ),
            )
            .with(SITE));
        }
        let (whole, start) = self.visible(t);
        let mut scores = self.scores(q, whole, start, t);
        if let Some(k) = topk.filter(|&k| k > 0 && k < scores.len()) {
            keep_topk(&mut scores, k);
        }
        softmax_row(&mut scores);
        Ok(self.combine(&scores, whole, start, t))
    }

    /// 槽分数 `q·k/√dim`：先压缩块、再滑窗原始行。
    fn scores(&self, q: &[S], whole: usize, start: usize, t: usize) -> Vec<S> {
        let scale = S::ONE.div(S::from_f64((self.dim as f64).sqrt()));
        let mut out = Vec::with_capacity(whole + t + 1 - start);
        for b in 0..whole {
            let key = &self.block_keys[b * self.dim..(b + 1) * self.dim];
            out.push(dot(q, key).mul(scale));
        }
        for i in start..=t {
            let key = &self.keys[i * self.dim..(i + 1) * self.dim];
            out.push(dot(q, key).mul(scale));
        }
        out
    }

    /// 按 softmax 权重对 V 加权求和，权重与槽一一对应。
    fn combine(&self, weights: &[S], whole: usize, start: usize, t: usize) -> Vec<S> {
        let mut out = vec![S::ZERO; self.dim];
        for (b, &w) in weights[..whole].iter().enumerate() {
            let value = &self.block_values[b * self.dim..(b + 1) * self.dim];
            accumulate(&mut out, value, w);
        }
        for (offset, i) in (start..=t).enumerate() {
            let value = &self.values[i * self.dim..(i + 1) * self.dim];
            accumulate(&mut out, value, weights[whole + offset]);
        }
        out
    }
}

/// 只留分数最大的 `k` 个槽，其余置 `−∞`。
///
/// 与 [`dsa_topk`] 的区别是**不改槽序**：这里的分支随后要拿权重乘 `V`，而 `V` 不会跟着重排，
/// 任何换位都会让分数与值错配。因此只取第 `k` 大的值做阈值，并列者按原序保留。
fn keep_topk<S: Scalar>(scores: &mut [S], k: usize) {
    let mut ranked = scores.to_vec();
    ranked.select_nth_unstable_by(k - 1, |a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    let threshold = ranked[k - 1];
    for score in scores.iter_mut() {
        if *score < threshold {
            *score = S::NEG_INFINITY;
        }
    }
}

/// 是否有限：`NaN` 与 `±∞` 都让 `v − v` 不等于零。
///
/// `pub(crate)` 而不是私有：MoE 的路由分布、mHC 的混合矩阵、byte-latent 的分数与 KDA 的门都要在入口
/// 挡非有限值，再各写一份判据就等于给「有限」两个定义。
pub(crate) fn finite<S: Scalar>(v: S) -> bool {
    v.sub(v) == S::ZERO
}

/// 点积 `a·b`（等长，由调用方保证），[`crate::mla`] 的分数也走这一份。
pub(crate) fn dot<S: Scalar>(a: &[S], b: &[S]) -> S {
    a.iter()
        .zip(b)
        .fold(S::ZERO, |acc, (&x, &y)| acc.add(x.mul(y)))
}

/// `acc += weight · v`，[`crate::mla`] 的加权求和也走这一份。
pub(crate) fn accumulate<S: Scalar>(acc: &mut [S], v: &[S], weight: S) {
    for (slot, &value) in acc.iter_mut().zip(v) {
        *slot = slot.add(weight.mul(value));
    }
}

/// 幂迭代估计谱范数 `‖B‖₂`（最大奇异值）。
pub fn spectral_norm<S: Scalar>(mat: &[S], n: usize, iters: usize) -> S {
    debug_assert_eq!(mat.len(), n * n);
    let mut v = vec![S::ONE.div(S::from_f64(n as f64).sqrt()); n];
    let mut sigma = S::ZERO;
    for _ in 0..iters {
        // u = B v。
        let u: Vec<S> = (0..n)
            .map(|i| (0..n).fold(S::ZERO, |acc, j| acc.add(mat[i * n + j].mul(v[j]))))
            .collect();
        sigma = u
            .iter()
            .copied()
            .fold(S::ZERO, |acc, x| acc.add(x.mul(x)))
            .sqrt();
        if sigma <= S::ZERO {
            return S::ZERO;
        }
        // v = Bᵀ u / ‖Bᵀ u‖。
        let w: Vec<S> = (0..n)
            .map(|j| (0..n).fold(S::ZERO, |acc, i| acc.add(mat[i * n + j].mul(u[i]))))
            .collect();
        let norm = w
            .iter()
            .copied()
            .fold(S::ZERO, |acc, x| acc.add(x.mul(x)))
            .sqrt();
        if norm <= S::ZERO {
            return S::ZERO;
        }
        for (x, &y) in v.iter_mut().zip(&w) {
            *x = y.div(norm);
        }
    }
    sigma
}

#[cfg(test)]
#[path = "../tests/csa_tests.rs"]
mod tests;
