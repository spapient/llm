//! `π_t` / `γ_t` 因果决策律（§1.6，`O(1)`，I/II）。
//!
//! 特征只许用 `t` 及以前的量（与 M11 同一因果）：`ρ_t` 最近 `W` 步块表未命中率、`ν_t = n_t/n_ref`、
//! `σ_t` 输入稀疏度。门控 `π_t = clip(ρ_t + (1+ν_t)⁻¹)`、`γ_t = clip(σ_t)`。短上下文或未命中抬 π
//! 走 α 检索；输入稀疏抬 γ 走事件臂语义（IEEE 上编码仍恒等）。

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
    /// β：条件记忆（[`crate::Bypass`]）。
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
///
/// 系数 `((1−γ)π, (1−γ)(1−π), γ)` 是 §1.5 的单位剖分；通道只是它的 argmax，回答"这一步走哪条
/// 路"，不替代合成——前向永远走 [`Gating::mix`] 的全合成，通道决定的是**该准备哪条路的输入**
/// （是否查块表、是否查记忆、是否切事件臂）。
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
///
/// §1.6 只给了 `ρ_t` 的定义（最近 `W` 步的未命中率），没给更新律；这里就是那条更新律：每步报告
/// 一次"这一步命没命中块表"，窗口滑一格，超过 `W` 丢最旧的一步。窗口空时 `ρ_t = 0`——没有证据
/// 就不因未命中抬 `π`，`π` 只由 `ν_t` 决定。
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
    //!
    //! 输入是候选上的分数，输出是该集合上的分布。MoE 专家、题目选项、层间路由都是这个集合；随后的
    //! TopK 只调用 L0（不在本模块）。默认校准名 `spherical`；`softmax` / `temperature` / `brier` 只在
    //! 同一端口注册后才能被配置点名，未注册名字启动失败。§1.6 的 clip 公式不是第二个模块，而是
    //! `head_layers = 0`（逐 token 默认）时的特征组合。

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
    ///
    /// 与 `spherical` 同属严格适当评分族，注册在 `DecisionHead` 的同一端口（§1.11）。
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

    #[cfg(test)]
    mod tests {
        use super::*;

        const TOL: f64 = 1e-9;

        /// 已注册名解析成功，未注册名失败。
        #[test]
        fn from_name_accepts_only_registered() {
            assert_eq!(Calib::from_name("spherical"), Some(Calib::Spherical));
            assert_eq!(Calib::from_name("softmax"), Some(Calib::Softmax));
            assert_eq!(
                Calib::from_name("temperature"),
                Some(Calib::Temperature(1.0))
            );
            assert_eq!(Calib::from_name("brier"), Some(Calib::Brier));
            assert_eq!(Calib::from_name("argmax"), None, "未注册名字启动失败");
        }

        /// softmax 输出和为 1。
        #[test]
        fn softmax_sums_to_one() {
            let d = Calib::Softmax.to_dist(&[1.0f64, 2.0, 3.0]);
            assert!((d.iter().sum::<f64>() - 1.0).abs() <= TOL);
        }

        /// 球面校准输出 L2 范数为 1（非负分数）。
        #[test]
        fn spherical_has_unit_l2_norm() {
            let d = Calib::Spherical.to_dist(&[3.0f64, 4.0]);
            let norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() <= TOL);
        }

        /// 温度缩放：τ 越小分布越尖锐。
        #[test]
        fn temperature_sharpens_distribution() {
            let cold = Calib::Temperature(0.5).to_dist(&[1.0f64, 2.0]);
            let hot = Calib::Temperature(2.0).to_dist(&[1.0f64, 2.0]);
            assert!(cold[1] / cold[0] > hot[1] / hot[0]);
        }

        /// Brier 评分：命中时为 0（one-hot），未命中为正，值域 `[0,1]`。
        #[test]
        fn brier_scores_perfect_as_zero() {
            let p = vec![0.0f64, 1.0, 0.0];
            assert!((brier(&p, 1)).abs() <= TOL);
            let wrong = brier(&p, 0);
            assert!(wrong > 0.0 && wrong <= 1.0);
        }
    }
}

pub use decision::{Calib, HEAD_LAYERS_TOKEN, brier};

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

    /// 系数是单位剖分：三项非负且和为 1。
    #[test]
    fn gating_coefficients_partition_unity() {
        for (pi_t, gamma_t) in [(0.0f64, 0.0), (0.25, 0.5), (1.0, 0.0), (0.7, 1.0)] {
            let c = Gating::new(pi_t, gamma_t).coefficients();
            assert!(c.iter().all(|&w| w >= 0.0), "{c:?}");
            assert!((c.iter().sum::<f64>() - 1.0).abs() <= TOL, "{c:?}");
        }
    }

    /// 主导通道 = 系数最大者，等值按 α、β、γ 的声明次序。
    #[test]
    fn the_largest_coefficient_picks_the_channel() {
        assert_eq!(Gating::new(1.0f64, 0.0).channel(), Channel::Alpha);
        assert_eq!(Gating::new(0.0f64, 0.0).channel(), Channel::Beta);
        assert_eq!(Gating::new(1.0f64, 0.9).channel(), Channel::Gamma);
        assert_eq!(Gating::new(0.5f64, 0.0).channel(), Channel::Alpha);
    }

    /// 在线窗口只保留最近 `W` 步，比率随之滑动。
    #[test]
    fn the_window_keeps_the_last_w_observations() {
        let mut rate = MissRate::new();
        assert_eq!(rate.rho(), 0.0, "空窗口没有证据");
        for _ in 0..W {
            rate.observe(true);
        }
        assert_eq!(rate.rho(), 1.0);
        rate.observe(false);
        assert!((rate.rho() - (W - 1) as f64 / W as f64).abs() <= TOL);
        for _ in 0..W {
            rate.observe(false);
        }
        assert_eq!(rate.rho(), 0.0, "旧的未命中被挤出窗口");
    }

    /// 一步门控吃的是这一步的特征：短上下文抬 π、稀疏输入抬 γ。
    #[test]
    fn routing_reads_the_step_it_is_given() {
        let rate = MissRate::default();
        let dense = vec![1.0f64; 8];
        let sparse = [1.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let short = rate.route(&dense, 16);
        let long = rate.route(&dense, 1 << 20);
        assert!(short.pi() > long.pi(), "短上下文更该走 α 检索");
        assert_eq!(short.channel(), Channel::Alpha);
        assert!(
            rate.route(&sparse, 16).gamma() > short.gamma(),
            "稀疏输入更该走事件臂"
        );
    }

    /// 通道选择不替代合成：前向走的就是 §1.5 的全合成。
    #[test]
    fn mix_is_the_full_synthesis() {
        let gating = Gating::new(0.25f64, 0.5);
        let (a, b, e) = (vec![1.0f64, 2.0], vec![3.0f64, 4.0], vec![5.0f64, 6.0]);
        assert_eq!(
            gating.mix(&a, &b, &e),
            crate::mix::synthesize(&a, &b, &e, 0.25, 0.5)
        );
        assert_eq!(
            Gating::new(0.5f64, 1.0).mix(&a, &b, &e),
            e,
            "γ = 1 只剩事件臂"
        );
        assert_eq!(
            Gating::new(1.0f64, 0.0).mix(&a, &b, &e),
            a,
            "π = 1、γ = 0 只剩 α"
        );
    }

    /// 短名稳定（诊断、快照与报告按它对齐）。
    #[test]
    fn channel_short_names_are_stable() {
        assert_eq!(Channel::Alpha.as_str(), "alpha");
        assert_eq!(Channel::Beta.as_str(), "beta");
        assert_eq!(Channel::Gamma.as_str(), "gamma");
    }
}
