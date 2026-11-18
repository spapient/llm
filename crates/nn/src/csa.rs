//! V4 CSA/HCA 超长注意力（§5.8.3）：序列分块压缩 + 稀疏/稠密注意力。
//!
//! CSA：每 `m=4` token 压成 1 条，再 DSA top-k（Flash 512 / Pro 1024）；HCA：每 `m'=128` token 压成
//! 1 条，稠密注意。有效长 `n/m`、`n/m'`（数学验证 41）。这是 α 通道的**序列压缩注意力**，与 MLA
//! 潜在维压缩分列，禁止写成 `O(n)` 扫描（Mamba-3 轴）。

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
pub fn compress(seq: &[f32], n: usize, d: usize, m: usize) -> Vec<f32> {
    assert_eq!(seq.len(), n * d, "seq 长度与形状不符");
    assert!(m > 0, "分块因子必须非零");
    let blocks = n.div_ceil(m);
    let mut out = vec![0.0f32; blocks * d];
    for b in 0..blocks {
        let start = b * m;
        let end = (start + m).min(n);
        let count = (end - start) as f32;
        for j in 0..d {
            let sum: f32 = (start..end).map(|i| seq[i * d + j]).sum();
            out[b * d + j] = sum / count;
        }
    }
    out
}

/// DSA top-k：每行只保留最大的 `k` 个分数，其余置 `−∞`（softmax 后为 0，实现稀疏注意力）。
pub fn dsa_topk(scores: &mut [f32], rows: usize, cols: usize, k: usize) {
    debug_assert_eq!(scores.len(), rows * cols);
    debug_assert!(k <= cols);
    for r in 0..rows {
        let row = &mut scores[r * cols..(r + 1) * cols];
        if k >= cols {
            continue;
        }
        // 降序选择：index k−1 是第 k 大，`[0..k]` 为 top-k，`[k..]` 为弃项。
        row.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
        for v in &mut row[k..] {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// 稠密注意力：每行 softmax 归一化（HCA 的稠密路径，作为 DSA 的对照）。
pub fn dense_softmax(scores: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(scores.len(), rows * cols);
    for r in 0..rows {
        let row = &mut scores[r * cols..(r + 1) * cols];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
}

/// Sinkhorn 双随机归一化（方阵）：行/列交替缩放，收敛后行和、列和均 ≈ 1，且 `‖B‖₂ ≤ 1`
/// （数学验证 41）。返回是否在 `iters` 轮内收敛到 `tol`。
pub fn sinkhorn(mat: &mut [f32], n: usize, iters: usize, tol: f32) -> bool {
    debug_assert_eq!(mat.len(), n * n);
    for _ in 0..iters {
        // 行归一。
        for r in 0..n {
            let sum: f32 = mat[r * n..(r + 1) * n].iter().sum();
            if sum > 0.0 {
                for v in &mut mat[r * n..(r + 1) * n] {
                    *v /= sum;
                }
            }
        }
        // 列归一。
        for c in 0..n {
            let sum: f32 = (0..n).map(|r| mat[r * n + c]).sum();
            if sum > 0.0 {
                for r in 0..n {
                    mat[r * n + c] /= sum;
                }
            }
        }
    }
    is_doubly_stochastic(mat, n, tol)
}

/// 是否（近似）双随机：行和、列和都 ≈ 1。
pub fn is_doubly_stochastic(mat: &[f32], n: usize, tol: f32) -> bool {
    debug_assert_eq!(mat.len(), n * n);
    (0..n).all(|r| {
        let row: f32 = mat[r * n..(r + 1) * n].iter().sum();
        let col: f32 = (0..n).map(|i| mat[i * n + r]).sum();
        (row - 1.0).abs() <= tol && (col - 1.0).abs() <= tol
    })
}

/// 幂迭代估计谱范数 `‖B‖₂`（最大奇异值）。
pub fn spectral_norm(mat: &[f32], n: usize, iters: usize) -> f32 {
    debug_assert_eq!(mat.len(), n * n);
    let mut v = vec![1.0f32 / (n as f32).sqrt(); n];
    let mut sigma = 0.0f32;
    for _ in 0..iters {
        // u = B v。
        let u: Vec<f32> = (0..n)
            .map(|i| (0..n).map(|j| mat[i * n + j] * v[j]).sum())
            .collect();
        sigma = u.iter().map(|x| x * x).sum::<f32>().sqrt();
        if sigma <= 0.0 {
            return 0.0;
        }
        // v = Bᵀ u / ‖Bᵀ u‖。
        let w: Vec<f32> = (0..n)
            .map(|j| (0..n).map(|i| mat[i * n + j] * u[i]).sum())
            .collect();
        let norm = w.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm <= 0.0 {
            return 0.0;
        }
        for (x, &y) in v.iter_mut().zip(&w) {
            *x = y / norm;
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
        assert!(HCA_M > CSA_M);
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
}
