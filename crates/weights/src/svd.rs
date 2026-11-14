//! 零依赖 f64 奇异值分解：单边 Jacobi SVD（one-sided Jacobi）。
//!
//! 不先形成 `AᵀA`（那会把条件数平方、损失小奇异值精度），而是直接在工作矩阵的列对之间做
//! Jacobi 旋转，逐步把列正交化：收敛后列长度即奇异值，归一化列即左奇异向量，累积的旋转给出
//! 右奇异向量。Jacobi 旋转对小奇异值精度高、实现仅用 f64 算术，适合替代旧版的 faer 依赖。
//!
//! 物理下界维度（浮点机器精度）只承诺残差不劣于 `O(ε_mach·σ_1)`，不宣称精确零。

use diagnostic::{Code, Error, Result};

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
}

/// 对行主序 `a[m*n]` 做 SVD。
pub fn compute(a: &[f64], m: usize, n: usize) -> Result<Svd> {
    if m == 0 || n == 0 || a.len() != m * n {
        return Err(Error::new(
            Code::Invalid,
            format!("SVD: 形状不合法 m={m} n={n} len={}", a.len()),
        )
        .with("svd::compute"));
    }
    let k = m.min(n);
    let mut u = a.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    orthogonalize(&mut u, &mut v, m, n);
    let sigma = singular_values(&mut u, m, n);
    let (sigma, u, v) = sorted_columns(&sigma, &u, &v, m, n, k);
    Ok(Svd { sigma, u, v, m, n })
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

/// 一对列：已正交返回 `false`，做了旋转返回 `true`。
fn rotate_pair(u: &mut [f64], v: &mut [f64], m: usize, n: usize, p: usize, q: usize) -> bool {
    let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
    for row in 0..m {
        let x = u[row * n + p];
        let y = u[row * n + q];
        alpha += x * x;
        beta += y * y;
        gamma += x * y;
    }
    if gamma.abs() <= TOL * (alpha * beta).sqrt() {
        return false;
    }
    let (c, s) = rotation(alpha, beta, gamma);
    rotate_columns(u, m, n, p, q, c, s);
    rotate_columns(v, n, n, p, q, c, s);
    true
}

/// 对 `mat` 的第 p、q 列施加旋转 `(c, s)`；`cols` 是行跨度。
fn rotate_columns(mat: &mut [f64], rows: usize, cols: usize, p: usize, q: usize, c: f64, s: f64) {
    for row in 0..rows {
        let x = mat[row * cols + p];
        let y = mat[row * cols + q];
        mat[row * cols + p] = c * x - s * y;
        mat[row * cols + q] = s * x + c * y;
    }
}

/// 列长度即奇异值；随后把 U 的列归一化。
fn singular_values(u: &mut [f64], m: usize, n: usize) -> Vec<f64> {
    let mut sigma = vec![0.0f64; n];
    for (j, value) in sigma.iter_mut().enumerate() {
        *value = column_norm(u, m, n, j);
    }
    for (j, &sigma) in sigma.iter().enumerate() {
        normalize_column(u, m, n, j, sigma);
    }
    sigma
}

/// 一列的欧氏长度。
fn column_norm(u: &[f64], m: usize, n: usize, j: usize) -> f64 {
    let mut norm = 0.0;
    for row in 0..m {
        norm += u[row * n + j].powi(2);
    }
    norm.sqrt()
}

/// 列除以它的长度；零列保持为零。
fn normalize_column(u: &mut [f64], m: usize, n: usize, j: usize, sigma: f64) {
    if sigma <= 0.0 {
        return;
    }
    for row in 0..m {
        u[row * n + j] /= sigma;
    }
}

/// 按奇异值降序重排 U、V 的列，只保留前 k 列。
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
        copy_column(u, m, n, &mut sorted_u, k, old_col, new_col);
        copy_column(v, n, n, &mut sorted_v, k, old_col, new_col);
    }
    (sorted_sigma, sorted_u, sorted_v)
}

/// 把 `src` 的第 `from` 列拷进 `dst` 的第 `to` 列。
fn copy_column(
    src: &[f64],
    rows: usize,
    src_stride: usize,
    dst: &mut [f64],
    dst_stride: usize,
    from: usize,
    to: usize,
) {
    for row in 0..rows {
        dst[row * dst_stride + to] = src[row * src_stride + from];
    }
}

/// 由列 Gram 参数算 Jacobi 旋转系数 `(c, s)`。
fn rotation(alpha: f64, beta: f64, gamma: f64) -> (f64, f64) {
    let tau = (beta - alpha) / (2.0 * gamma);
    let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
    let c = 1.0 / (1.0 + t * t).sqrt();
    (c, c * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frob_resid(a: &[f64], back: &[f64]) -> f64 {
        a.iter()
            .zip(back)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn diagonal_singular_values_are_exact() {
        // diag(3,2,1,0.5) 行主序。
        let mut a = vec![0.0f64; 16];
        let diag = [3.0, 2.0, 1.0, 0.5];
        for (i, d) in diag.iter().enumerate() {
            a[i * 4 + i] = *d;
        }
        let svd = compute(&a, 4, 4).expect("svd");
        for (got, want) in svd.sigma.iter().zip(diag.iter()) {
            assert!((got - want).abs() < 1e-10, "got {got} want {want}");
        }
    }

    #[test]
    fn full_reconstruction_residual_is_machine_order() {
        let mut rng = 0x1234_5678u64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 40) as f64 / (1u64 << 24) as f64 - 0.5
        };
        for (m, n) in [(8, 8), (12, 6), (6, 12), (20, 10)] {
            let a: Vec<f64> = (0..m * n).map(|_| next()).collect();
            let svd = compute(&a, m, n).expect("svd");
            let k = m.min(n);
            assert_eq!(svd.sigma.len(), k);
            let back = svd.reconstruct(k);
            let scale = svd.sigma[0];
            assert!(
                frob_resid(&a, &back) < 1e-10 * scale * (m as f64).sqrt(),
                "残差应在机器精度量级 m={m} n={n}"
            );
        }
    }

    #[test]
    fn truncated_reconstruction_is_monotonic() {
        let mut a = vec![0.0f64; 36];
        for i in 0..6 {
            a[i * 6 + i] = 6.0 - i as f64;
        }
        let svd = compute(&a, 6, 6).expect("svd");
        let mut prev = f64::INFINITY;
        for r in 1..=6 {
            let err = frob_resid(&a, &svd.reconstruct(r));
            assert!(err <= prev, "增加秩不应让误差变大");
            prev = err;
        }
    }

    #[test]
    fn singular_values_are_descending_and_nonnegative() {
        let mut rng = 99u64;
        let a: Vec<f64> = (0..64)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng >> 40) as f64 / (1u64 << 24) as f64
            })
            .collect();
        let svd = compute(&a, 8, 8).expect("svd");
        for pair in svd.sigma.windows(2) {
            assert!(pair[0] >= pair[1]);
            assert!(pair[0] >= 0.0);
        }
    }

    #[test]
    fn bad_shape_is_an_error() {
        assert!(compute(&[1.0], 2, 2).is_err());
        assert!(compute(&[], 0, 4).is_err());
    }
}
