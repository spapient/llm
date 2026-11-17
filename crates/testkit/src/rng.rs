//! 确定性伪随机数：测试共享一个不带 `rand` 依赖的 xorshift32，让失败可复现。

/// 确定性 PRNG（xorshift32）：测试共享，不给依赖表加 `rand`，同时让失败可复现。
pub struct Rng {
    state: u32,
}

impl Rng {
    /// 固定种子。
    #[must_use]
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// 取 `[0, bound)` 内的确定性值。
    pub fn below(&mut self, bound: u32) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state % bound.max(1)
    }

    /// `±[0.5, 2]` 的确定性值：远离 0，避免平方类算子的差分为 0 造成无谓抵消。
    pub fn value(&mut self) -> f64 {
        let magnitude = 0.5 + 1.5 * f64::from(self.below(1_000)) / 1_000.0;
        match self.below(2) {
            0 => magnitude,
            _ => -magnitude,
        }
    }

    /// `[0, 1)` 内的确定性 f32。
    pub fn next_f32(&mut self) -> f32 {
        self.below(1_000_000) as f32 / 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_bounded() {
        let mut left = Rng::new(1234);
        let mut right = Rng::new(1234);
        for _ in 0..16 {
            let value = left.below(10);
            assert_eq!(value, right.below(10), "同种子序列一致");
            assert!(value < 10, "结果落在界内");
        }
        assert_eq!(Rng::new(7).below(1), 0, "bound=1 恒为 0");
    }

    #[test]
    fn rng_value_stays_away_from_zero_and_is_signable() {
        let mut rng = Rng::new(42);
        for _ in 0..16 {
            let value = rng.value();
            assert!(value.abs() >= 0.5 && value.abs() <= 2.0);
        }
    }

    #[test]
    fn rng_next_f32_is_in_unit_interval() {
        let mut rng = Rng::new(7);
        for _ in 0..16 {
            let value = rng.next_f32();
            assert!((0.0..1.0).contains(&value));
        }
    }
}
