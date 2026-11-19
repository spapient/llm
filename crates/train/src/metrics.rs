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
mod tests {
use crate::{MetricPoint, MetricsBoard};
use diagnostic::{Code, Result};

/// 加权聚合：loss 与命中率按样本数加权，best 取各步最低。
#[test]
fn aggregates_weighted_by_sample_count() -> Result<()> {
    let mut board = MetricsBoard::new();
    board.record(MetricPoint {
        step: 0,
        samples: 2,
        loss: 1.0,
        correct: 2,
    })?;
    board.record(MetricPoint {
        step: 1,
        samples: 2,
        loss: 3.0,
        correct: 0,
    })?;

    assert_eq!(board.len(), 2);
    assert!(!board.is_empty());
    assert_eq!(board.latest().map(|point| point.step), Some(1));

    let mean_loss = board.mean_loss().expect("有样本");
    assert!((mean_loss - 2.0).abs() < 1e-6, "样本加权均值 loss 应为 2.0");
    let mean_acc = board.mean_accuracy().expect("有样本");
    assert!((mean_acc - 0.5).abs() < 1e-6, "样本加权命中率应为 0.5");
    assert_eq!(board.best_loss(), Some(1.0));
    Ok(())
}

/// 空看板：聚合量与最新点都是 `None`，渲染占位行。
#[test]
fn empty_board_reports_nothing() {
    let board = MetricsBoard::new();
    assert!(board.is_empty());
    assert!(board.latest().is_none());
    assert!(board.mean_loss().is_none());
    assert!(board.mean_accuracy().is_none());
    assert!(board.best_loss().is_none());
    assert!(board.render().contains("n=0"));
}

/// 非法点入账前拒绝：样本为 0、loss 非有限、命中条数超过样本数。
#[test]
fn rejects_invalid_points() {
    let mut board = MetricsBoard::new();

    let zero_samples = MetricPoint {
        step: 0,
        samples: 0,
        loss: 1.0,
        correct: 0,
    };
    assert_eq!(
        board.record(zero_samples).expect_err("样本为 0").code(),
        Code::Invalid
    );

    let nan_loss = MetricPoint {
        step: 0,
        samples: 1,
        loss: f32::NAN,
        correct: 0,
    };
    assert_eq!(
        board.record(nan_loss).expect_err("NaN loss").code(),
        Code::Invalid
    );

    let too_many_correct = MetricPoint {
        step: 0,
        samples: 1,
        loss: 1.0,
        correct: 2,
    };
    assert_eq!(
        board
            .record(too_many_correct)
            .expect_err("命中条数超过样本数")
            .code(),
        Code::Invalid
    );
    assert!(board.is_empty(), "拒绝前不写入任何点");
}

/// 渲染行带出最新步的 loss、命中率与样本数。
#[test]
fn render_shows_latest_metrics() -> Result<()> {
    let mut board = MetricsBoard::new();
    board.record(MetricPoint {
        step: 3,
        samples: 4,
        loss: 0.75,
        correct: 1,
    })?;
    let line = board.render();
    assert!(line.contains("3"), "行内带步号");
    assert!(line.contains("0.7500"), "行内带 loss");
    assert!(line.contains("25.0%"), "行内带命中率");
    assert!(line.contains("n=4"), "行内带样本数");
    Ok(())
}

/// 样本数不等时，大批次对均值的贡献更大：micro 加权而非步均值。
#[test]
fn weights_by_larger_batch_not_by_step_count() -> Result<()> {
    let mut board = MetricsBoard::new();
    // 小批 2 样本：loss 1.0、全对；大批 4 样本：loss 3.0、全错。
    board.record(MetricPoint {
        step: 0,
        samples: 2,
        loss: 1.0,
        correct: 2,
    })?;
    board.record(MetricPoint {
        step: 1,
        samples: 4,
        loss: 3.0,
        correct: 0,
    })?;

    let mean_loss = board.mean_loss().expect("有样本");
    let expect_loss = (2.0 + 4.0 * 3.0) / 6.0;
    assert!(
        (mean_loss - expect_loss).abs() < 1e-6,
        "样本加权 loss 应为 {expect_loss}，得到 {mean_loss}"
    );
    let mean_acc = board.mean_accuracy().expect("有样本");
    assert!(
        (mean_acc - 2.0 / 6.0).abs() < 1e-6,
        "样本加权命中率应为 1/3，得到 {mean_acc}"
    );
    Ok(())
}

/// 指标点按入账顺序留存，`points` 不重排。
#[test]
fn points_preserve_record_order() -> Result<()> {
    let mut board = MetricsBoard::new();
    board.record(MetricPoint {
        step: 5,
        samples: 1,
        loss: 0.5,
        correct: 1,
    })?;
    board.record(MetricPoint {
        step: 2,
        samples: 1,
        loss: 0.8,
        correct: 0,
    })?;
    let steps: Vec<u64> = board.points().iter().map(|point| point.step).collect();
    assert_eq!(steps, [5, 2], "按入账顺序保留，不按步号重排");
    Ok(())
}
}

