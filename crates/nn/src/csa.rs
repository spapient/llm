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
const SITE: &str = "nn::csa::BlockTable";

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
mod tests {
    use super::*;

    /// 分块因子钉死：m' ≫ m。
    #[test]
    fn frozen_compress_factors() {
        assert_eq!(CSA_M, 4);
        assert_eq!(HCA_M, 128);
        const { assert!(HCA_M > CSA_M) };
    }

    /// 有效长 = ⌈n/m⌉。
    #[test]
    fn effective_len_ceil_div() {
        assert_eq!(effective_len(100, CSA_M), 25);
        assert_eq!(effective_len(100, HCA_M), 1);
        assert_eq!(effective_len(7, 4), 2);
    }

    /// 分块压缩按 m 均值池化，末块不足 m 也正确。
    #[test]
    fn compress_mean_pools_blocks() {
        // n=4, d=2, m=2 → 2 块。
        let seq = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let got = compress(&seq, 4, 2, 2);
        assert_eq!(got, vec![2.0, 3.0, 6.0, 7.0]);
        // 末块不足：n=3, m=2 → 2 块，末块单行。
        let tail = compress(&[1.0, 10.0, 3.0, 30.0, 5.0, 50.0], 3, 2, 2);
        assert_eq!(tail, vec![2.0, 20.0, 5.0, 50.0]);
    }

    /// 分块压缩 f64 同式（泛型复用：与 f32 不复制实现）。
    #[test]
    fn compress_f64_agrees() {
        let seq = vec![1.0f64, 2.0, 3.0, 4.0];
        assert_eq!(compress(&seq, 4, 1, 2), vec![1.5, 3.5]);
    }

    /// DSA 每行只保留 top-k。
    #[test]
    fn dsa_topk_keeps_largest_k() {
        let mut s = vec![1.0f32, 5.0, 3.0, 4.0, 2.0];
        dsa_topk(&mut s, 1, 5, 2);
        let finite: Vec<f32> = s.iter().copied().filter(|x| x.is_finite()).collect();
        assert_eq!(finite.len(), 2);
        assert!(finite.contains(&5.0) && finite.contains(&4.0));
    }

    /// 稠密 softmax 行和为 1。
    #[test]
    fn dense_softmax_sums_to_one() {
        let mut s = vec![1.0f32, 2.0, 3.0];
        dense_softmax(&mut s, 1, 3);
        assert!((s.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    /// Sinkhorn 收敛到双随机，且 ‖B‖₂ ≤ 1。
    #[test]
    fn sinkhorn_reaches_doubly_stochastic() {
        let mut b = vec![1.0f32, 3.0, 4.0, 2.0];
        let ok = sinkhorn(&mut b, 2, 200, 1e-5);
        assert!(ok, "Sinkhorn 应收敛到双随机");
        assert!(spectral_norm(&b, 2, 100) <= 1.0 + 1e-3);
    }

    /// 幂迭代对单位阵给出谱范数 1。
    #[test]
    fn spectral_norm_of_identity_is_one() {
        let mut id = vec![0.0f32; 9];
        for i in 0..3 {
            id[i * 3 + i] = 1.0;
        }
        assert!((spectral_norm(&id, 3, 100) - 1.0).abs() < 1e-4);
    }

    /// 稠密因果注意力（对拍基准）：位置 `t` 看 `0..=t` 的全部原始行。
    fn dense(q: &[f32], k: &[f32], v: &[f32], dim: usize, t: usize) -> Vec<f32> {
        let scale = 1.0 / (dim as f32).sqrt();
        let mut scores: Vec<f32> = (0..=t)
            .map(|i| (0..dim).map(|j| q[j] * k[i * dim + j]).sum::<f32>() * scale)
            .collect();
        dense_softmax(&mut scores, 1, t + 1);
        (0..dim)
            .map(|j| (0..=t).map(|i| scores[i] * v[i * dim + j]).sum())
            .collect()
    }

    /// 一行确定性的 K/V 夹具（不用随机数：对拍要可复跑）。
    fn fixture(seq: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
        let key = (0..seq * dim)
            .map(|i| ((i * 7 + 3) % 11) as f32 * 0.1)
            .collect();
        let value = (0..seq * dim)
            .map(|i| ((i * 5 + 1) % 13) as f32 * 0.2)
            .collect();
        (key, value)
    }

    /// 块数 = ⌈seq/m⌉ = `effective_len`，压缩表长度与块数一致。
    #[test]
    fn a_block_table_holds_ceil_seq_over_m_blocks() {
        let (key, _) = fixture(100, 1);
        let table = BlockTable::new(&key, &key, 100, 1, CSA_M, WIN).expect("合法表");
        assert_eq!(table.blocks(), effective_len(100, CSA_M));
        assert_eq!(table.compressed().0.len(), 100 / CSA_M);
        assert_eq!(table.compressed().1.len(), 100 / CSA_M);
        let hca = BlockTable::new(&key, &key, 100, 1, HCA_M, WIN).expect("合法表");
        assert_eq!(hca.blocks(), 1, "100 行按 128 压成 1 块");
    }

    /// m = win = 1 时可见槽恰好是 `0..=t`，输出与稠密因果注意力一致。
    #[test]
    fn tight_geometry_matches_dense_causal_attention() {
        let (seq, dim) = (6, 3);
        let (key, value) = fixture(seq, dim);
        let table = BlockTable::new(&key, &value, seq, dim, 1, 1).expect("m = win = 1");
        for t in 0..seq {
            let q: Vec<f32> = (0..dim).map(|j| ((t + 3 * j) % 5) as f32 * 0.3).collect();
            let got = table.attend(&q, t, None).expect("合法查询");
            let want = dense(&q, &key, &value, dim, t);
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() < 1e-5, "t = {t}：{got:?} ≠ {want:?}");
            }
        }
    }

    /// 因果：`t` 之后的行怎么改都不影响 `t` 的输出；`t` 之前的一改就影响。
    #[test]
    fn future_tokens_cannot_reach_the_past() {
        let (seq, dim) = (8, 2);
        let (key, value) = fixture(seq, dim);
        let (mut tail_key, mut tail_value) = (key.clone(), value.clone());
        for x in tail_key[5 * dim..]
            .iter_mut()
            .chain(tail_value[5 * dim..].iter_mut())
        {
            *x *= 100.0;
        }
        let before = BlockTable::new(&key, &value, seq, dim, CSA_M, 2).expect("合法表");
        let after = BlockTable::new(&tail_key, &tail_value, seq, dim, CSA_M, 2).expect("合法表");
        let q = [1.0f32, -0.5];
        for t in 0..5 {
            assert_eq!(
                before.attend(&q, t, None).expect("合法查询"),
                after.attend(&q, t, None).expect("合法查询"),
                "t = {t} 看见了未来"
            );
        }
        let mut past = key.clone();
        past[0] = 42.0;
        let shifted = BlockTable::new(&past, &value, seq, dim, CSA_M, 2).expect("合法表");
        assert_ne!(
            before.attend(&q, 4, None).expect("合法查询"),
            shifted.attend(&q, 4, None).expect("合法查询"),
            "改第一行必须改变 t = 4 的输出，否则上一条是空断言"
        );
    }

    /// top-k 只留分数最大的槽：k = 1 的输出就是最优槽的 V。
    #[test]
    fn topk_keeps_only_the_best_slot() {
        let k = [1.0f32, 2.0, 3.0, 4.0];
        let v = [10.0f32, 20.0, 30.0, 40.0];
        let table = BlockTable::new(&k, &v, 4, 1, 1, 1).expect("合法表");
        assert_eq!(table.attend(&[1.0], 3, Some(1)).expect("合法查询"), [40.0]);
        let dense_out = table.attend(&[1.0], 3, None).expect("合法查询");
        assert!(
            dense_out[0] > 30.0 && dense_out[0] < 40.0,
            "稠密是真加权：{dense_out:?}"
        );
    }

    /// 几何不合法与查询越界都在入口失败。
    #[test]
    fn a_block_table_rejects_bad_geometry_and_queries() {
        let k = [1.0f32, 2.0, 3.0, 4.0];
        for (dim, m, win) in [(0, 1, 1), (2, 0, 1), (2, 1, 0)] {
            assert_eq!(
                BlockTable::new(&k, &k, 2, dim, m, win).unwrap_err().code(),
                Code::Invalid,
                "dim = {dim}、m = {m}、win = {win} 必须在构造期被拒"
            );
        }
        assert_eq!(
            BlockTable::new(&k, &k, 3, 2, 1, 1).unwrap_err().code(),
            Code::Invalid,
            "K/V 长度必须是 seq × dim"
        );
        let table = BlockTable::new(&k, &k, 2, 2, 1, 1).expect("合法表");
        assert_eq!(
            table.attend(&[1.0], 0, None).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(
            table.attend(&[1.0, 2.0], 2, None).unwrap_err().code(),
            Code::Invalid,
            "位置 2 已经越过 seq = 2"
        );
    }
}
