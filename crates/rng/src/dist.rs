//! 分布闭集：参数在构造期一次性校验，采样是 `u ∈ [0,1) → 元素` 的纯映射。
//!
//! 中间计算一律在 f32 完成：均匀缩放、Box-Muller 的对数与开方都不在半精度上累积；
//! 输出到 F16/BF16 的舍入属于 dtype 转换职责，不在本层。分布参数按 f32 解析并要求有限。

use std::cmp::Ordering;

use diagnostic::{Code, Error, Result};

use crate::philox::Rng;

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("rng::dist")
}

/// 均匀分布 `[lo, hi)`。
pub struct Uniform {
    lo: f32,
    hi: f32,
}

impl Uniform {
    /// 校验端点有限且 `lo < hi`；相等、空区间、NaN 或无穷都拒绝。
    pub fn new(lo: f32, hi: f32) -> Result<Self> {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(bad("均匀区间端点必须有限"));
        }
        if lo.partial_cmp(&hi) != Some(Ordering::Less) {
            return Err(bad("均匀区间必须满足 lo < hi"));
        }
        Ok(Self { lo, hi })
    }

    /// 抽一个 `[lo, hi)` 内的 f32。
    pub fn sample(&self, rng: &mut Rng) -> f32 {
        self.lo + (self.hi - self.lo) * rng.next_f32()
    }
}

/// 伯努利分布：以概率 `p` 返回 `true`。
pub struct Bernoulli {
    p: f32,
}

impl Bernoulli {
    /// 校验概率落在 `[0, 1]` 内且有限。
    pub fn new(p: f32) -> Result<Self> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(bad("伯努利概率必须是 [0,1] 内的有限值"));
        }
        Ok(Self { p })
    }

    /// 抽一次试验结果。
    pub fn sample(&self, rng: &mut Rng) -> bool {
        rng.next_f32() < self.p
    }
}

/// 正态分布 `N(mean, std²)`。
pub struct Normal {
    mean: f32,
    std: f32,
}

impl Normal {
    /// 校验均值有限、标准差为正且有限。
    pub fn new(mean: f32, std: f32) -> Result<Self> {
        if !mean.is_finite() || !std.is_finite() {
            return Err(bad("正态参数必须有限"));
        }
        if std.partial_cmp(&0.0) != Some(Ordering::Greater) {
            return Err(bad("标准差必须为正数"));
        }
        Ok(Self { mean, std })
    }

    /// Box-Muller 变换抽一个标准正态样本。
    ///
    /// 每次取两个均匀量生成一对样本，本轮只返回其一；第二值不缓存，避免把分布状态
    /// 藏进采样器。尾部质量要求更高的场景应在调用侧升级到 f64 中间计算后再回 f32。
    pub fn sample(&self, rng: &mut Rng) -> f32 {
        let u1 = rng.next_f32().max(f32::MIN_POSITIVE);
        let u2 = rng.next_f32();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        self.mean + self.std * radius * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_respects_endpoints_and_is_reproducible() {
        let distribution = Uniform::new(-3.0, 5.0).expect("区间合法");
        let mut left = Rng::new(2024);
        let mut right = Rng::new(2024);
        for _ in 0..16 {
            let value = distribution.sample(&mut left);
            assert!((-3.0..5.0).contains(&value), "落在 [lo,hi)");
            assert_eq!(distribution.sample(&mut right), value, "同种子可复现");
        }
    }

    #[test]
    fn uniform_rejects_empty_or_non_finite_range() {
        assert!(Uniform::new(1.0, 1.0).is_err());
        assert!(Uniform::new(1.0, f32::NAN).is_err());
        assert!(Uniform::new(f32::INFINITY, 2.0).is_err());
        assert!(Uniform::new(2.0, 1.0).is_err());
    }

    #[test]
    fn bernoulli_probability_is_bounded() {
        assert!(Bernoulli::new(0.0).is_ok());
        assert!(Bernoulli::new(1.0).is_ok());
        assert!(Bernoulli::new(-0.1).is_err());
        assert!(Bernoulli::new(1.2).is_err());
        assert!(Bernoulli::new(f32::NAN).is_err());

        let always = Bernoulli::new(1.0).expect("p=1 合法");
        let never = Bernoulli::new(0.0).expect("p=0 合法");
        let mut rng = Rng::new(1);
        assert!(always.sample(&mut rng));
        assert!(!never.sample(&mut rng));
    }

    #[test]
    fn normal_samples_are_finite_and_reproducible() {
        let distribution = Normal::new(1.0, 0.25).expect("参数合法");
        let mut left = Rng::new(11);
        let mut right = Rng::new(11);
        for _ in 0..16 {
            let value = distribution.sample(&mut left);
            assert!(value.is_finite(), "正态样本必须有限");
            assert_eq!(distribution.sample(&mut right), value, "同种子可复现");
        }
    }

    #[test]
    fn normal_rejects_non_positive_or_non_finite_std() {
        assert!(Normal::new(0.0, 0.0).is_err());
        assert!(Normal::new(0.0, -1.0).is_err());
        assert!(Normal::new(f32::NAN, 1.0).is_err());
        assert!(Normal::new(0.0, f32::INFINITY).is_err());
    }
}
