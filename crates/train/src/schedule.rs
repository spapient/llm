//! 学习率调度：warmup 升到峰值，之后按 [`Anneal`] 曲线降到下限比例。
//!
//! 曲线是纯数据闭集，[`lr`](Self::lr) 是纯函数：同一份 [`Schedule`] 每次算出同一条曲线，
//! 便于对拍、缓存与落盘复现。下限比例 `min_ratio` 只压住线性/余弦的谷底，恒定曲线不受其压。

use diagnostic::{Code, Error, Result};

/// 衰减曲线闭集：warmup 段对所有策略一致，差别只在 warmup 之后怎么降到学习率下限。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Anneal {
    /// 线性衰减：与峰值成比例匀速下行，只有两个参数、`lr()` 是纯函数，可解释性优先。
    #[default]
    Linear,
    /// 余弦衰减：warmup 后按 `½(1 + cos(π·p))` 平滑下行，是 LLM 训练主流的末段收敛形态。
    Cosine,
    /// 恒定：warmup 后保持峰值，下限不参与下行。
    Constant,
}

impl Anneal {
    /// 曲线短名：诊断与报告里的稳定标签。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Cosine => "cosine",
            Self::Constant => "constant",
        }
    }

    /// 归一化形状：`1.0` 为峰值、`0.0` 为谷底；谷底比例由 [`factor`](Self::factor) 统一插值。
    fn shaped(self, progress: f32) -> f32 {
        match self {
            Self::Constant => 1.0,
            Self::Linear => 1.0 - progress,
            Self::Cosine => 0.5 * (1.0 + (core::f32::consts::PI * progress).cos()),
        }
    }

    /// 曲线在进度 `progress ∈ [0, 1]` 处的因子：`min_ratio` 为谷底，`1.0` 为峰值。
    fn factor(self, progress: f32, min_ratio: f32) -> f32 {
        let progress = progress.clamp(0.0, 1.0);
        min_ratio + (1.0 - min_ratio) * self.shaped(progress)
    }
}

/// 学习率调度：warmup 升到峰值，之后按 [`Anneal`] 曲线降到下限比例。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Schedule {
    warmup: u64,
    total: u64,
    anneal: Anneal,
    min_ratio: f32,
}

impl Schedule {
    /// 线性衰减到 0：`total` 必须非零，`warmup` 不得超过 `total`。
    pub fn new(warmup: u64, total: u64) -> Result<Self> {
        Self::with(warmup, total, Anneal::Linear, 0.0)
    }

    /// 全参构造；`min_ratio` 是谷底占峰值的比例，必须落在 `[0, 1]`。
    pub fn with(warmup: u64, total: u64, anneal: Anneal, min_ratio: f32) -> Result<Self> {
        if total == 0 {
            return Err(Error::new(Code::Invalid, "总步数不能为 0").with("train::Schedule::with"));
        }
        if warmup > total {
            return Err(
                Error::new(Code::Invalid, format!("warmup {warmup} 超过总步数 {total}"))
                    .with("train::Schedule::with"),
            );
        }
        if !(0.0..=1.0).contains(&min_ratio) {
            return Err(
                Error::new(Code::Invalid, format!("下限比例 {min_ratio} 不在 [0, 1]"))
                    .with("train::Schedule::with"),
            );
        }
        Ok(Self {
            warmup,
            total,
            anneal,
            min_ratio,
        })
    }

    /// warmup 步数。
    pub fn warmup(&self) -> u64 {
        self.warmup
    }

    /// 总步数。
    pub fn total(&self) -> u64 {
        self.total
    }

    /// 衰减曲线。
    pub fn anneal(&self) -> Anneal {
        self.anneal
    }

    /// 谷底占峰值的比例。
    pub fn min_ratio(&self) -> f32 {
        self.min_ratio
    }

    /// 第 `step` 步的学习率：`peak` 是 warmup 结束时的值，超过 `total` 后为 0。
    pub fn lr(&self, peak: f32, step: u64) -> f32 {
        if step >= self.total {
            return 0.0;
        }
        if self.warmup > 0 && step < self.warmup {
            // 第 0 步已经有非零学习率，否则前若干步等于白跑。
            return peak * ((step + 1) as f32) / (self.warmup as f32);
        }
        let span = self.total - self.warmup;
        if span == 0 {
            return peak;
        }
        let progress = (step - self.warmup) as f32 / span as f32;
        peak * self.anneal.factor(progress, self.min_ratio)
    }
}

#[cfg(test)]
mod tests {
use super::*;
use diagnostic::Code;

#[test]
fn the_schedule_warms_up_then_decays_to_zero() {
    let schedule = Schedule::new(10, 100).expect("合法");
    assert_eq!(schedule.lr(1.0, 0), 0.1);
    assert_eq!(schedule.lr(1.0, 9), 1.0);
    assert!(
        (schedule.lr(1.0, 55) - 0.5).abs() < 1e-6,
        "warmup 后线性衰减"
    );
    assert_eq!(schedule.lr(1.0, 100), 0.0);
    assert_eq!(schedule.lr(1.0, 1_000), 0.0);
}

#[test]
fn a_schedule_without_warmup_starts_at_the_peak() {
    let schedule = Schedule::new(0, 10).expect("合法");
    assert_eq!(schedule.lr(0.5, 0), 0.5);
    assert!(
        (schedule.lr(0.5, 9) - 0.05).abs() < 1e-6,
        "无 warmup 时直接线性衰减"
    );
}

#[test]
fn an_impossible_schedule_is_rejected() {
    assert_eq!(Schedule::new(0, 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Schedule::new(11, 10).unwrap_err().code(), Code::Invalid);
    assert_eq!(
        Schedule::with(0, 10, Anneal::Cosine, 1.5)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
}

#[test]
fn cosine_decays_from_the_peak_to_the_floor_ratio() {
    let schedule = Schedule::with(0, 100, Anneal::Cosine, 0.1).expect("合法");
    assert_eq!(schedule.lr(1.0, 0), 1.0, "进度起点在峰值");
    let floor = schedule.lr(1.0, 99);
    assert!((floor - 0.1).abs() < 1e-3, "末段落到下限比例：{floor}");
    assert_eq!(schedule.lr(1.0, 100), 0.0, "超出总步数为 0");
}

#[test]
fn a_constant_schedule_holds_the_peak_after_warmup() {
    let schedule = Schedule::with(10, 100, Anneal::Constant, 0.1).expect("合法");
    assert_eq!(schedule.lr(1.0, 9), 1.0, "warmup 结束到峰值");
    assert_eq!(schedule.lr(1.0, 50), 1.0, "恒定曲线不向下限压");
}

#[test]
fn linear_decay_respects_the_floor_ratio() {
    let schedule = Schedule::with(0, 100, Anneal::Linear, 0.2).expect("合法");
    let floor = schedule.lr(1.0, 99);
    assert!(floor >= 0.2 - 1e-6, "线性谷底不低于下限：{floor}");
    assert!(floor <= 1.0, "学习率不超过峰值：{floor}");
}
}

