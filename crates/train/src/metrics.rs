//! 训练指标看板：按步累积 loss 与 accuracy，给出加权均值、最优 loss 与可读行。
//!
//! 看板只持有已记账的评估量，不做测量也不写性能结论；数值由编排器从前向结果算出后传入。
//! 聚合按**样本数**加权：一批样本多的指标对均值贡献更大，步均值会被 micro-batch 尺寸带偏。
//! 本模块不持久化——序列化接口属后续里程碑，当前只在内存累积并渲染人可读行。

use diagnostic::{Code, Error, Result};

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::metrics";

/// 一条步指标：步号、本步样本数、平均 loss、命中条数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricPoint {
    /// 优化器步数（未步进的累积批不计步号）。
    pub step: u64,
    /// 本步参与评估的样本条数。
    pub samples: usize,
    /// 本步平均交叉熵。
    pub loss: f32,
    /// 本步命中条数，闭区间 `0..=samples`。
    pub correct: u64,
}

impl MetricPoint {
    /// 命中率 `correct / samples`，闭区间 `[0, 1]`。
    pub fn accuracy(&self) -> f32 {
        self.correct as f32 / self.samples as f32
    }
}

/// 指标累积表：逐点入账 + 加权聚合计数。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetricsBoard {
    points: Vec<MetricPoint>,
    eval_points: Vec<MetricPoint>,
    samples: u64,
    loss_sum: f64,
    correct: u64,
    best_loss: Option<f32>,
    eval_samples: u64,
    eval_loss_sum: f64,
    eval_correct: u64,
    best_eval_loss: Option<f32>,
}

impl MetricsBoard {
    /// 空看板。
    pub fn new() -> Self {
        Self::default()
    }

    /// 记一条训练步指标；累计样本、加权 loss 与命中条数，并刷新最优 loss。
    ///
    /// # Errors
    ///
    /// 样本数为 0、loss 非有限数，或命中条数超过样本数时返回 [`Code::Invalid`]。
    pub fn record(&mut self, point: MetricPoint) -> Result<()> {
        validate(&point)?;
        let samples = point.samples as u64;
        self.samples += samples;
        self.loss_sum += point.loss as f64 * samples as f64;
        self.correct += point.correct;
        self.best_loss = Some(match self.best_loss {
            Some(best) => best.min(point.loss),
            None => point.loss,
        });
        self.points.push(point);
        Ok(())
    }

    /// 记一条评估步指标：独立账表，不并入训练序列的样本数、加权均值与最优 loss。
    ///
    /// # Errors
    ///
    /// 样本数为 0、loss 非有限数，或命中条数超过样本数时返回 [`Code::Invalid`]。
    pub fn record_eval(&mut self, point: MetricPoint) -> Result<()> {
        validate(&point)?;
        let samples = point.samples as u64;
        self.eval_samples += samples;
        self.eval_loss_sum += point.loss as f64 * samples as f64;
        self.eval_correct += point.correct;
        self.best_eval_loss = Some(match self.best_eval_loss {
            Some(best) => best.min(point.loss),
            None => point.loss,
        });
        self.eval_points.push(point);
        Ok(())
    }

    /// 已记账的训练指标点（按入账顺序）。
    pub fn points(&self) -> &[MetricPoint] {
        &self.points
    }

    /// 已记账的评估指标点（与训练序列分开）。
    pub fn eval_points(&self) -> &[MetricPoint] {
        &self.eval_points
    }

    /// 已记账的点数。
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// 看板是否还没有任何指标。
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// 最近一条指标；空看板为 `None`。
    pub fn latest(&self) -> Option<&MetricPoint> {
        self.points.last()
    }

    /// 样本加权平均 loss；无样本为 `None`。
    pub fn mean_loss(&self) -> Option<f32> {
        self.nonzero_samples()
            .map(|samples| (self.loss_sum / samples as f64) as f32)
    }

    /// 样本加权命中率；无样本为 `None`。
    pub fn mean_accuracy(&self) -> Option<f32> {
        self.nonzero_samples()
            .map(|samples| self.correct as f32 / samples as f32)
    }

    /// 各步最低训练 loss；空看板为 `None`。
    pub fn best_loss(&self) -> Option<f32> {
        self.best_loss
    }

    /// 评估账表样本加权平均 loss；无评估样本为 `None`。
    pub fn mean_eval_loss(&self) -> Option<f32> {
        (self.eval_samples > 0).then_some((self.eval_loss_sum / self.eval_samples as f64) as f32)
    }

    /// 评估账表样本加权命中率；无评估样本为 `None`。
    pub fn mean_eval_accuracy(&self) -> Option<f32> {
        (self.eval_samples > 0).then_some(self.eval_correct as f32 / self.eval_samples as f32)
    }

    /// 各步最低评估 loss；空看板为 `None`。
    pub fn best_eval_loss(&self) -> Option<f32> {
        self.best_eval_loss
    }

    /// 累计样本数（大于 0 时才参与加权均值）。
    fn nonzero_samples(&self) -> Option<u64> {
        (self.samples > 0).then_some(self.samples)
    }

    /// 渲染最近一条为可读行；空看板返回占位说明。
    pub fn render(&self) -> String {
        match self.latest() {
            Some(point) => format!(
                "step {:>4} | loss {:.4} | acc {:5.1}% | n={}",
                point.step,
                point.loss,
                point.accuracy() * 100.0,
                point.samples
            ),
            None => "step    - | loss     - | acc     -% | n=0".to_string(),
        }
    }
}

/// 一条指标的入参校验：样本非零、loss 有限、命中条数不超过样本数。
fn validate(point: &MetricPoint) -> Result<()> {
    if point.samples == 0 {
        return Err(Error::new(Code::Invalid, "指标样本数不能为 0").with(SITE));
    }
    if !point.loss.is_finite() {
        let message = format!("loss {} 不是有限数（NaN / Infinity 不进看板）", point.loss);
        return Err(Error::new(Code::Invalid, message).with(SITE));
    }
    if point.correct > point.samples as u64 {
        let message = format!("命中条数 {} 超过样本数 {}", point.correct, point.samples);
        return Err(Error::new(Code::Invalid, message).with(SITE));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/metrics_tests.rs"]
mod tests;
