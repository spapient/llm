//! `T_θ` 三通道合成律（§1.5，数学验证 31）。
//!
//! `y_cont = π·y_α + (1−π)·y_β`；`y_γ = Encode_event(y_cont)`；`y_t = (1−γ)·y_cont + γ·y_γ`。
//! 系数 `((1−γ)π, (1−γ)(1−π), γ)` 构成单位剖分（和为 1、各项 ≥ 0），故 `y_t` 落在三通道输出的
//! 凸包内。事件编码 `y_γ` 由调用方（[`crate::mix::pulse`]）算出，本模块只做纯合成，无状态。

use op::scan::Scalar;

/// α/β 凸组合：`y_cont = π·y_α + (1−π)·y_β`。
pub fn cont<S: Scalar>(ya: &[S], yb: &[S], pi: S) -> Vec<S> {
    debug_assert_eq!(ya.len(), yb.len());
    let om = S::ONE.sub(pi);
    ya.iter()
        .zip(yb)
        .map(|(&a, &b)| a.mul(pi).add(b.mul(om)))
        .collect()
}

/// 连续/事件门控：`y_t = (1−γ)·y_cont + γ·y_γ`。
pub fn gate<S: Scalar>(yc: &[S], yg: &[S], gamma: S) -> Vec<S> {
    debug_assert_eq!(yc.len(), yg.len());
    let om = S::ONE.sub(gamma);
    yc.iter()
        .zip(yg)
        .map(|(&c, &g)| c.mul(om).add(g.mul(gamma)))
        .collect()
}

/// 全合成：`y_t = (1−γ)(π·y_α + (1−π)·y_β) + γ·y_γ`。
pub fn synthesize<S: Scalar>(ya: &[S], yb: &[S], yg: &[S], pi: S, gamma: S) -> Vec<S> {
    gate(&cont(ya, yb, pi), yg, gamma)
}
mod pulse {
    //! γ 脉冲通道：事件编码 `Encode_event`（§1.5）。
    //!
    //! IEEE 上 `Encode_event ≡ Id`（Lipschitz 常数 `L_e = 1`）；FR-HW 走 LIF 发放 + 事件驱动整数
    //! 累加。γ 通道不是可选装饰，而是跨平台统一契约的必要组成（§1.5 第 6 点）：它让同一模型可跨
    //! IEEE/FR-HW 迁移，并允许运行时按输入稀疏度切换事件臂。

    use op::scan::Scalar;

    /// LIF 神经元（FR-HW 事件臂）：膜电位积分 + 泄漏 + 阈值发放 + 重置。
    ///
    /// 状态是膜电位 [`Lif::mem`]；每步 [`Lif::step`] 返回本步发放（`ONE`/`ZERO`），事件驱动的整数
    /// 累加由调用方把返回值累加（§1.4 第 5 类）。
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Lif<S> {
        threshold: S,
        leak: S,
        reset: S,
        mem: S,
    }

    impl<S: Scalar> Lif<S> {
        /// 构造：发放阈值 `threshold`、每步泄漏比例 `leak ∈ [0,1]`（`0` = 纯 IF）、发放后重置 `reset`。
        pub fn new(threshold: S, leak: S, reset: S) -> Self {
            Self {
                threshold,
                leak,
                reset,
                mem: S::ZERO,
            }
        }

        /// 推进一步：`mem = mem·(1−leak) + u`，超阈值发放并重置。
        ///
        /// 返回本步发放（`ONE` = 发放，`ZERO` = 未发放）。
        pub fn step(&mut self, u: S) -> S {
            self.mem = self.mem.mul(S::ONE.sub(self.leak)).add(u);
            let spike = if self.mem >= self.threshold {
                S::ONE
            } else {
                S::ZERO
            };
            if spike == S::ONE {
                self.mem = self.reset;
            }
            spike
        }

        /// 当前膜电位（观测用）。
        pub fn mem(&self) -> S {
            self.mem
        }
    }

    /// IEEE 事件编码恒等：`Encode_event(y) = y`（`L_e = 1`，§1.5）。
    ///
    /// 逐元素恒等，因此 IEEE 六平台上 γ 混合与 `γ≡0/1` 输出一致（F0-2 门控同构）。
    pub fn encode_id<S: Scalar>(y: &[S]) -> Vec<S> {
        y.to_vec()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// 阈值以下积分不发放，膜电位累加。
        #[test]
        fn integrates_without_spike_below_threshold() {
            let mut lif = Lif::new(1.0f64, 0.0, 0.0);
            assert_eq!(lif.step(0.3), 0.0);
            assert_eq!(lif.step(0.3), 0.0);
            assert!((lif.mem() - 0.6).abs() < 1e-12);
        }

        /// 达到阈值发放并重置。
        #[test]
        fn spikes_and_resets_at_threshold() {
            let mut lif = Lif::new(1.0f64, 0.0, 0.0);
            assert_eq!(lif.step(0.6), 0.0);
            assert_eq!(lif.step(0.6), 1.0, "超过阈值发放");
            assert!((lif.mem() - 0.0).abs() < 1e-12, "发放后重置");
        }

        /// 泄漏使膜电位随时间衰减。
        #[test]
        fn leak_decays_membrane() {
            let mut lif = Lif::new(1.0f64, 0.5, 0.0);
            lif.step(0.8);
            let before = lif.mem();
            lif.step(0.0);
            assert!(lif.mem() < before, "泄漏后膜电位下降");
        }

        /// IEEE 恒等编码逐元素不变。
        #[test]
        fn encode_id_is_identity() {
            let y = vec![1.0f64, -2.0, 3.0];
            assert_eq!(encode_id(&y), y);
        }
    }
}

pub use pulse::{Lif, encode_id};

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= TOL, "{x} vs {y}");
        }
    }

    /// 系数构成单位剖分：和为 1。
    #[test]
    fn coefficients_partition_unity() {
        for pi in [0.0, 0.25, 0.5, 1.0] {
            for gamma in [0.0, 0.25, 0.5, 1.0] {
                let c_alpha = (1.0 - gamma) * pi;
                let c_beta = (1.0 - gamma) * (1.0 - pi);
                let c_gamma = gamma;
                assert!((c_alpha + c_beta + c_gamma - 1.0).abs() <= TOL);
            }
        }
    }

    /// γ=0 时输出就是连续合成。
    #[test]
    fn gamma_zero_is_continuous() {
        let (ya, yb) = (vec![1.0, 2.0], vec![3.0, 4.0]);
        let yc = cont(&ya, &yb, 0.25);
        close(&synthesize(&ya, &yb, &ya, 0.25, 0.0), &yc);
    }

    /// γ=1 时输出就是事件通道。
    #[test]
    fn gamma_one_is_event() {
        let (ya, yb, yg) = (vec![1.0, 2.0], vec![3.0, 4.0], vec![0.0, 1.0]);
        close(&synthesize(&ya, &yb, &yg, 0.25, 1.0), &yg);
    }

    /// IEEE 恒等编码下（y_γ = y_cont），γ 混合与纯连续一致（F0-2）。
    #[test]
    fn ieee_gamma_mix_is_continuous() {
        let (ya, yb) = (vec![1.0, -2.0], vec![3.0, 4.0]);
        let yc = cont(&ya, &yb, 0.4);
        for gamma in [0.0, 0.3, 0.7, 1.0] {
            close(&synthesize(&ya, &yb, &yc, 0.4, gamma), &yc);
        }
    }

    /// 输出落在各通道值的凸包（min/max）内。
    #[test]
    fn output_in_convex_hull() {
        let ya: Vec<f64> = vec![1.0, 5.0];
        let yb: Vec<f64> = vec![3.0, 2.0];
        let yg: Vec<f64> = vec![9.0, -1.0];
        for pi in [0.0, 0.5, 1.0] {
            for gamma in [0.0, 0.5, 1.0] {
                let y = synthesize(&ya, &yb, &yg, pi, gamma);
                for (i, &v) in y.iter().enumerate() {
                    let lo = ya[i].min(yb[i]).min(yg[i]);
                    let hi = ya[i].max(yb[i]).max(yg[i]);
                    assert!(v >= lo && v <= hi, "v={v} 不在 [{lo},{hi}]");
                }
            }
        }
    }
}
