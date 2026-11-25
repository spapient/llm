//! 确定性伪随机与确定性样本：测试共享一个不带 `rand` 依赖的 xorshift32，让失败可复现。

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

    /// `[-1, 1)` 内的确定性 f32：千分网格上取值，有符号夹具（权重、输入）共用一份。
    pub fn signed_unit(&mut self) -> f32 {
        self.below(2_001) as f32 / 1_000.0 - 1.0
    }
}

/// 确定性样本序列：基础 PRNG 公式（`i·2654435761 + salt·40503`）只在这里写一次，
/// 各测试的取值口径（移位、模、缩放、偏移、元素类型）通过 `map` 闭包**配置化**给出。
/// 同一 `(len, salt)` 永远同一批值；`T` 由映射决定（f32 / i8 / 其他标量均可）。
#[must_use]
pub fn sample_with<T>(len: usize, salt: u32, map: impl Fn(u32) -> T) -> Vec<T> {
    (0..len)
        .map(|i| {
            let z = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503));
            map(z)
        })
        .collect()
}

/// 确定性测试样本：`sample(len, salt)` 生成 `[-1, 1)` 的 f32，同一 `(len, salt)` 永远同一批值。
#[must_use]
pub fn sample(len: usize, salt: u32) -> Vec<f32> {
    sample_with(len, salt, |z| ((z >> 8) % 2048) as f32 / 1024.0 - 1.0)
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

    #[test]
    fn rng_signed_unit_covers_both_signs_inside_the_band() {
        let mut rng = Rng::new(9);
        let draws: Vec<f32> = (0..64).map(|_| rng.signed_unit()).collect();
        assert!(draws.iter().all(|value| (-1.0..1.0).contains(value)));
        assert!(draws.iter().any(|value| *value < 0.0));
        assert!(draws.iter().any(|value| *value >= 0.0));
    }

    #[test]
    fn sample_is_reproducible_and_inside_the_band() {
        assert_eq!(sample(9, 3), sample(9, 3), "同 (len, salt) 必须同一批值");
        assert_ne!(sample(9, 3), sample(9, 4), "salt 必须真的参与计算");
        assert!(
            sample(64, 1)
                .iter()
                .all(|value| (-1.0..1.0).contains(value))
        );
        assert_eq!(sample(0, 7), Vec::<f32>::new(), "零长批是空切片");
    }

    #[test]
    fn sample_with_shares_the_base_prng_and_supports_custom_mappings() {
        // `sample` 就是 `sample_with` 的 [-1,1) 投影：同一条基础序列，不多一份公式。
        assert_eq!(
            sample(16, 5),
            sample_with(16, 5, |z| ((z >> 8) % 2048) as f32 / 1024.0 - 1.0)
        );
        // 可配置映射：缩放、偏移、元素类型各取所需，基础 PRNG 公式只留一份。
        let scaled = sample_with(8, 1, |z| ((z >> 8) as f32) / 256.0 - 32.0);
        assert_eq!(
            scaled,
            sample_with(8, 1, |z| ((z >> 8) as f32) / 256.0 - 32.0)
        );
        let as_i8 = sample_with(8, 1, |z| ((z % 251) as i32 - 125) as i8);
        assert!(as_i8.iter().all(|v| (-125..=125).contains(v)));
        assert_eq!(sample_with::<f32>(0, 1, |_| 0.0), Vec::<f32>::new());
    }
}
