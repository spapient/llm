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
}

/// 信号秩选择的相对谱隙阈值。
///
/// 依据（本机实测）：满秩高斯矩阵相邻奇异值的最大相对谱隙 ≈0.03；结构化近似低秩矩阵
/// （秩-10 信号 + 高斯噪声）在信号/噪声边界处的相对谱隙 ≈0.49。0.2 落在这两个实测值之间，
/// 与两侧都保持 >2× 的分离，用于把「有信号可压」与「纯噪声不可压」分开。
pub const REL_GAP: f64 = 0.2;

/// 有损补偿侧的信号秩：返回**噪声底之前**的最大秩 `r`，即 `(σ_r − σ_{r+1}) / σ_1 > REL_GAP`
/// 的最后一点。
///
/// 与 [`crate::ledger`] 的 `certified_rank` 分工：后者用算法残差 `ε`（机器精度量级）做绝对
/// 阈值，服务**无损**压缩的精确秩判定（满秩矩阵返回满秩是正确语义）；本函数用相对阈值，服务
/// **有损**量化补偿（满秩噪声矩阵无信号谱隙 → 返回 `None` = 拒压回退）。两者不可互换。
#[must_use]
pub fn signal_rank(sigma: &[f64]) -> Option<usize> {
    let floor = REL_GAP * sigma.first()?;
    (1..sigma.len())
        .rev()
        .find(|&r| sigma[r - 1] - sigma[r] > floor)
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
    // 工作矩阵用列主序：Jacobi 全程都在列对上做点积/旋转，列连续存放才能命中缓存
    // （行主序下 n=1024 的列访问步长 8KiB，每次都是缓存未命中）。输出仍按约定行主序。
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

#[cfg(test)]
#[path = "../tests/ptq_svd.rs"]
mod tests;
