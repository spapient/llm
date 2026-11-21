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

/// 校验谓词不成立时返回 [`bad`]；成立为 `Ok`。分布参数校验共用这一条出口。
fn check(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(bad(message)) }
}

/// 均匀分布 `[lo, hi)`。
pub struct Uniform {
    lo: f32,
    hi: f32,
}

impl Uniform {
    /// 校验端点有限且 `lo < hi`；相等、空区间、NaN 或无穷都拒绝。
    pub fn new(lo: f32, hi: f32) -> Result<Self> {
        check(lo.is_finite() && hi.is_finite(), "均匀区间端点必须有限")?;
        check(
            lo.partial_cmp(&hi) == Some(Ordering::Less),
            "均匀区间必须满足 lo < hi",
        )?;
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
        check(
            p.is_finite() && (0.0..=1.0).contains(&p),
            "伯努利概率必须是 [0,1] 内的有限值",
        )?;
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
        check(mean.is_finite() && std.is_finite(), "正态参数必须有限")?;
        check(
            std.partial_cmp(&0.0) == Some(Ordering::Greater),
            "标准差必须为正数",
        )?;
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
#[path = "../tests/dist_tests.rs"]
mod tests;
