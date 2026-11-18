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
        accuracy: 1.0,
    })?;
    board.record(MetricPoint {
        step: 1,
        samples: 2,
        loss: 3.0,
        accuracy: 0.0,
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

/// 非法点入账前拒绝：样本为 0、loss 非有限、命中率越界。
#[test]
fn rejects_invalid_points() {
    let mut board = MetricsBoard::new();

    let zero_samples = MetricPoint {
        step: 0,
        samples: 0,
        loss: 1.0,
        accuracy: 0.5,
    };
    assert_eq!(
        board.record(zero_samples).expect_err("样本为 0").code(),
        Code::Invalid
    );

    let nan_loss = MetricPoint {
        step: 0,
        samples: 1,
        loss: f32::NAN,
        accuracy: 0.5,
    };
    assert_eq!(
        board.record(nan_loss).expect_err("NaN loss").code(),
        Code::Invalid
    );

    let too_high = MetricPoint {
        step: 0,
        samples: 1,
        loss: 1.0,
        accuracy: 1.5,
    };
    assert_eq!(
        board.record(too_high).expect_err("命中率 > 1").code(),
        Code::Invalid
    );

    let too_low = MetricPoint {
        step: 0,
        samples: 1,
        loss: 1.0,
        accuracy: -0.2,
    };
    assert_eq!(
        board.record(too_low).expect_err("命中率 < 0").code(),
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
        accuracy: 0.25,
    })?;
    let line = board.render();
    assert!(line.contains("3"), "行内带步号");
    assert!(line.contains("0.7500"), "行内带 loss");
    assert!(line.contains("25.0%"), "行内带命中率");
    assert!(line.contains("n=4"), "行内带样本数");
    Ok(())
}
