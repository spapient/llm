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
