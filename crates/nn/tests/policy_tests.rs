use super::*;

const TOL: f64 = 1e-9;

/// 已注册名解析成功，未注册名失败。
#[test]
fn from_name_accepts_only_registered() {
    assert_eq!(Calib::from_name("spherical"), Some(Calib::Spherical));
    assert_eq!(Calib::from_name("softmax"), Some(Calib::Softmax));
    assert_eq!(
        Calib::from_name("temperature"),
        Some(Calib::Temperature(1.0))
    );
    assert_eq!(Calib::from_name("brier"), Some(Calib::Brier));
    assert_eq!(Calib::from_name("argmax"), None, "未注册名字启动失败");
}

/// softmax 输出和为 1。
#[test]
fn softmax_sums_to_one() {
    let d = Calib::Softmax.to_dist(&[1.0f64, 2.0, 3.0]);
    assert!((d.iter().sum::<f64>() - 1.0).abs() <= TOL);
}

/// 球面校准输出 L2 范数为 1（非负分数）。
#[test]
fn spherical_has_unit_l2_norm() {
    let d = Calib::Spherical.to_dist(&[3.0f64, 4.0]);
    let norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() <= TOL);
}

/// 温度缩放：τ 越小分布越尖锐。
#[test]
fn temperature_sharpens_distribution() {
    let cold = Calib::Temperature(0.5).to_dist(&[1.0f64, 2.0]);
    let hot = Calib::Temperature(2.0).to_dist(&[1.0f64, 2.0]);
    assert!(cold[1] / cold[0] > hot[1] / hot[0]);
}

/// Brier 评分：命中时为 0（one-hot），未命中为正，值域 `[0,1]`。
#[test]
fn brier_scores_perfect_as_zero() {
    let p = vec![0.0f64, 1.0, 0.0];
    assert!((brier(&p, 1)).abs() <= TOL);
    let wrong = brier(&p, 0);
    assert!(wrong > 0.0 && wrong <= 1.0);
}
