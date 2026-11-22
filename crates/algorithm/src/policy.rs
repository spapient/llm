//! `π_t` / `γ_t` 因果决策律（§1.6，`O(1)`，I/II）。

use std::collections::VecDeque;

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

/// 通道闭集：α 检索、β 记忆、γ 事件。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// α：压缩块检索（[`crate::BlockTable`]）。
    Alpha,
    /// β：条件记忆（[`nn::Bypass`]）。
    Beta,
    /// γ：事件臂（IEEE 上 [`crate::encode_id`] 恒等）。
    Gamma,
}

impl Channel {
    /// 短名：诊断与报告里的稳定标签。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Alpha => "alpha",
            Self::Beta => "beta",
            Self::Gamma => "gamma",
        }
    }
}

/// 一步的门控结果：合成系数与主导通道。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gating<S: Scalar = f32> {
    pi: S,
    gamma: S,
    channel: Channel,
}

impl<S: Scalar> Gating<S> {
    /// 由两步门控系数构造，主导通道取系数最大者（等值按 α、β、γ 的声明次序）。
    pub fn new(pi: S, gamma: S) -> Self {
        let mut gating = Self {
            pi,
            gamma,
            channel: Channel::Alpha,
        };
        gating.channel = dominant(gating.coefficients());
        gating
    }

    /// α/β 门控 `π_t`。
    pub fn pi(&self) -> S {
        self.pi
    }

    /// 连续/事件门控 `γ_t`。
    pub fn gamma(&self) -> S {
        self.gamma
    }

    /// 主导通道。
    pub fn channel(&self) -> Channel {
        self.channel
    }

    /// 单位剖分系数，次序 α、β、γ：三项非负且和为 1。
    pub fn coefficients(&self) -> [S; 3] {
        let om = S::ONE.sub(self.gamma);
        [om.mul(self.pi), om.mul(S::ONE.sub(self.pi)), self.gamma]
    }

    /// 按 §1.5 合成律混合三通道输出（`mix::synthesize` 是唯一入口，这里不另写一份加权和）。
    pub fn mix(&self, alpha: &[S], beta: &[S], event: &[S]) -> Vec<S> {
        crate::mix::synthesize(alpha, beta, event, self.pi, self.gamma)
    }
}

/// 主导通道：系数最大者；等值按声明次序（确定性，不随比较写法漂移）。
fn dominant<S: Scalar>(weights: [S; 3]) -> Channel {
    let mut best = 0;
    for (i, &w) in weights.iter().enumerate().skip(1) {
        if w > weights[best] {
            best = i;
        }
    }
    [Channel::Alpha, Channel::Beta, Channel::Gamma][best]
}

/// 在线未命中率：最近 [`W`] 步块表未命中的因果窗口（`ρ_t`）。
#[derive(Debug, Clone)]
pub struct MissRate {
    misses: VecDeque<bool>,
}

impl MissRate {
    /// 空窗口（容量 [`W`]）。
    pub fn new() -> Self {
        Self {
            misses: VecDeque::with_capacity(W),
        }
    }

    /// 记一步：`miss = true` 表示这一步没命中块表；窗口满后丢最旧的一步。
    pub fn observe(&mut self, miss: bool) {
        if self.misses.len() == W {
            self.misses.pop_front();
        }
        self.misses.push_back(miss);
    }

    /// `ρ_t`：窗口内未命中率；窗口为空时为 0。
    pub fn rho(&self) -> f64 {
        if self.misses.is_empty() {
            return 0.0;
        }
        let misses = self.misses.iter().filter(|&&miss| miss).count();
        misses as f64 / self.misses.len() as f64
    }

    /// 一步门控：`ρ_t` 与 `ν_t` 给 `π_t`，输入稀疏度给 `γ_t`，再按系数选出主导通道。
    pub fn route<S: Scalar>(&self, hidden: &[S], n_t: usize) -> Gating<S> {
        let rho = S::from_f64(self.rho());
        let nu_t = nu(S::from_f64(n_t as f64));
        Gating::new(pi(rho, nu_t), gamma(sparsity(hidden, tau(hidden))))
    }
}

impl Default for MissRate {
    fn default() -> Self {
        Self::new()
    }
}

mod decision {
    //! `DecisionHead`：唯一门控实现（§1.11）。

    use op::scan::Scalar;

    /// `DecisionHead` 的逐 token 默认层深（§1.11：`head_layers = 0`）。
    pub const HEAD_LAYERS_TOKEN: usize = 0;

    /// 校准名闭集（score → 分布）。
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Calib {
        /// 球面（默认）：非负截断后 L2 归一化，严格适当评分族。
        Spherical,
        /// 标准 softmax。
        Softmax,
        /// 温度 softmax（`τ` 为温度，须为正）。
        Temperature(f64),
        /// Brier：先 softmax 得分布，再用 [`brier`] 评分（同一端口注册）。
        Brier,
    }

    impl Calib {
        /// 按名解析；未注册名字返回 `None`（启动失败，§1.11）。
        pub fn from_name(name: &str) -> Option<Self> {
            match name {
                "spherical" => Some(Self::Spherical),
                "softmax" => Some(Self::Softmax),
                "temperature" => Some(Self::Temperature(1.0)),
                "brier" => Some(Self::Brier),
                _ => None,
            }
        }

        /// 分数 → 候选集合上的分布。
        pub fn to_dist<S: Scalar>(self, scores: &[S]) -> Vec<S> {
            match self {
                Self::Spherical => spherical(scores),
                Self::Softmax => softmax(scores),
                Self::Temperature(t) => {
                    debug_assert!(t > 0.0, "温度必须为正");
                    let scaled: Vec<S> = scores.iter().map(|&s| s.div(S::from_f64(t))).collect();
                    softmax(&scaled)
                }
                Self::Brier => softmax(scores),
            }
        }
    }

    /// 稳定 softmax（先减最大值防溢出，输出和为 1）。
    fn softmax<S: Scalar>(scores: &[S]) -> Vec<S> {
        if scores.is_empty() {
            return Vec::new();
        }
        let max = scores
            .iter()
            .copied()
            .fold(scores[0], |m, x| if x > m { x } else { m });
        let exps: Vec<S> = scores.iter().map(|&s| s.sub(max).exp()).collect();
        let sum = exps.iter().copied().fold(S::ZERO, |a, b| a.add(b));
        exps.into_iter().map(|e| e.div(sum)).collect()
    }

    /// 球面投影：非负截断后 L2 归一化；零范数退化为均匀分布。
    fn spherical<S: Scalar>(scores: &[S]) -> Vec<S> {
        let pos: Vec<S> = scores
            .iter()
            .map(|&s| if s < S::ZERO { S::ZERO } else { s })
            .collect();
        let norm = pos
            .iter()
            .copied()
            .fold(S::ZERO, |a, b| a.add(b.mul(b)))
            .sqrt();
        if norm == S::ZERO {
            let u = S::ONE.div(S::from_f64(pos.len() as f64));
            return vec![u; pos.len()];
        }
        pos.into_iter().map(|p| p.div(norm)).collect()
    }

    /// Brier 严格适当评分：`B = (1/K) Σ (pᵢ − yᵢ)²`，`y` 为目标 one-hot（`target` 为命中下标）。
    pub fn brier<S: Scalar>(p: &[S], target: usize) -> S {
        debug_assert!(target < p.len());
        if p.is_empty() {
            return S::ZERO;
        }
        let mut sum = S::ZERO;
        for (i, &pi) in p.iter().enumerate() {
            let y = if i == target { S::ONE } else { S::ZERO };
            let d = pi.sub(y);
            sum = sum.add(d.mul(d));
        }
        sum.div(S::from_f64(p.len() as f64))
    }
}

pub use decision::{Calib, HEAD_LAYERS_TOKEN, brier};

#[cfg(test)]
#[path = "../tests/policy_tests.rs"]
mod tests;
