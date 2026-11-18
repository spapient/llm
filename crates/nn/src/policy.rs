//! `π_t` / `γ_t` 因果决策律（§1.6，`O(1)`，I/II）。
//!
//! 特征只许用 `t` 及以前的量（与 M11 同一因果）：`ρ_t` 最近 `W` 步块表未命中率、`ν_t = n_t/n_ref`、
//! `σ_t` 输入稀疏度。门控 `π_t = clip(ρ_t + (1+ν_t)⁻¹)`、`γ_t = clip(σ_t)`。短上下文或未命中抬 π
//! 走 α 检索；输入稀疏抬 γ 走事件臂语义（IEEE 上编码仍恒等）。

use op::scan::Scalar;

/// `ρ_t` 窗口宽度（冻结常数，数学验证 38）。
pub const W: usize = 64;
/// `ν_t` 的归一化参考长。
pub const N_REF: f64 = 4096.0;
/// `τ = 0.05 · RMS(x)` 的比例。
pub const TAU_SCALE: f64 = 0.05;

/// `ν_t = n_t / n_ref`。
pub fn nu<S: Scalar>(n_t: S) -> S {
    n_t.div(S::from_f64(N_REF))
}

/// α/β 门控：`π_t = clip(ρ_t + (1+ν_t)⁻¹, 0, 1)`。
pub fn pi<S: Scalar>(rho: S, nu: S) -> S {
    clip(rho.add(S::ONE.div(S::ONE.add(nu))))
}

/// 连续/事件门控：`γ_t = clip(σ_t, 0, 1)`。
pub fn gamma<S: Scalar>(sigma: S) -> S {
    clip(sigma)
}

/// `RMS(x) = sqrt(mean(xᵢ²))`。
pub fn rms<S: Scalar>(x: &[S]) -> S {
    if x.is_empty() {
        return S::ZERO;
    }
    let sum_sq = x.iter().fold(S::ZERO, |acc, &v| acc.add(v.mul(v)));
    sum_sq.div(S::from_f64(x.len() as f64)).sqrt()
}

/// `τ = 0.05 · RMS(x)`。
pub fn tau<S: Scalar>(x: &[S]) -> S {
    rms(x).mul(S::from_f64(TAU_SCALE))
}

/// 输入稀疏度 `σ_t = |{i: |xᵢ| < τ}| / |x|`。
pub fn sparsity<S: Scalar>(x: &[S], tau: S) -> S {
    if x.is_empty() {
        return S::ZERO;
    }
    let below = x.iter().filter(|&&v| v.abs() < tau).count();
    S::from_f64(below as f64 / x.len() as f64)
}

/// 把值夹到 `[0, 1]`。
fn clip<S: Scalar>(v: S) -> S {
    if v < S::ZERO {
        S::ZERO
    } else if v > S::ONE {
        S::ONE
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    /// 门控始终落在 `[0,1]`。
    #[test]
    fn gates_are_clipped() {
        assert_eq!(pi(-1.0f64, 0.0), 0.0);
        assert_eq!(pi(2.0, 0.0), 1.0);
        assert_eq!(gamma(-0.5f64), 0.0);
        assert_eq!(gamma(1.5), 1.0);
    }

    /// 短上下文（ν 小）抬 π，长上下文（ν 大）压 π。
    #[test]
    fn short_context_raises_pi() {
        assert!(pi(0.0, 0.0) > pi(0.0, 4.0));
    }

    /// RMS 与 τ 的比例关系。
    #[test]
    fn tau_is_scaled_rms() {
        let x = vec![1.0f64, 2.0, 3.0, 4.0];
        let expected = rms(&x) * TAU_SCALE;
        assert!((tau(&x) - expected).abs() <= TOL);
    }

    /// 稀疏度统计 `|xᵢ| < τ` 的占比。
    #[test]
    fn sparsity_counts_below_tau() {
        let x = vec![0.1f64, 0.2, 5.0, 6.0];
        assert!((sparsity(&x, 1.0) - 0.5).abs() <= TOL);
    }

    /// 空输入退化为零。
    #[test]
    fn empty_input_is_zero() {
        assert_eq!(rms::<f64>(&[]), 0.0);
        assert_eq!(sparsity::<f64>(&[], 1.0), 0.0);
    }
}
