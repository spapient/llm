//! 原先内联在 `src/policy.rs` 的测试。
//!
//! 用 `#[path]` 挂回原模块，模块树位置不变，故 `super::`/`crate::` 路径照旧。

use super::*;

const TOL: f64 = 1e-12;

testkit::test_cases! {
/// 门控始终落在 `[0,1]`。
fn gates_are_clipped() {
    assert_eq!(pi(-1.0f64, 0.0), 0.0);
    assert_eq!(pi(2.0, 0.0), 1.0);
    assert_eq!(gamma(-0.5f64), 0.0);
    assert_eq!(gamma(1.5), 1.0);
}

/// 短上下文（ν 小）抬 π，长上下文（ν 大）压 π。
fn short_ctx_raises_pi() {
    assert!(pi(0.0, 0.0) > pi(0.0, 4.0));
}

/// RMS 与 τ 的比例关系。
fn tau_is_scaled_rms() {
    let x = vec![1.0f64, 2.0, 3.0, 4.0];
    let expected = rms(&x) * TAU_SCALE;
    assert!((tau(&x) - expected).abs() <= TOL);
}

/// 稀疏度统计 `|xᵢ| < τ` 的占比。
fn sparsity_counts_below_tau() {
    let x = vec![0.1f64, 0.2, 5.0, 6.0];
    assert!((sparsity(&x, 1.0) - 0.5).abs() <= TOL);
}

/// 空输入退化为零。
fn empty_input_is_zero() {
    assert_eq!(rms::<f64>(&[]), 0.0);
    assert_eq!(sparsity::<f64>(&[], 1.0), 0.0);
}

/// 系数是单位剖分：三项非负且和为 1。
fn gating_coefficients_partition_unity() {
    for (pi_t, gamma_t) in [(0.0f64, 0.0), (0.25, 0.5), (1.0, 0.0), (0.7, 1.0)] {
        let c = Gating::new(pi_t, gamma_t).coefficients();
        assert!(c.iter().all(|&w| w >= 0.0), "{c:?}");
        assert!((c.iter().sum::<f64>() - 1.0).abs() <= TOL, "{c:?}");
    }
}

/// 主导通道 = 系数最大者，等值按 α、β、γ 的声明次序。
fn the_largest_coefficient_picks_the_channel() {
    assert_eq!(Gating::new(1.0f64, 0.0).channel(), Channel::Alpha);
    assert_eq!(Gating::new(0.0f64, 0.0).channel(), Channel::Beta);
    assert_eq!(Gating::new(1.0f64, 0.9).channel(), Channel::Gamma);
    assert_eq!(Gating::new(0.5f64, 0.0).channel(), Channel::Alpha);
}

/// 在线窗口只保留最近 `W` 步，比率随之滑动。
fn the_window_keeps_the_last_w_observations() {
    let mut rate = MissRate::new();
    assert_eq!(rate.rho(), 0.0, "空窗口没有证据");
    for _ in 0..W {
        rate.observe(true);
    }
    assert_eq!(rate.rho(), 1.0);
    rate.observe(false);
    assert!((rate.rho() - (W - 1) as f64 / W as f64).abs() <= TOL);
    for _ in 0..W {
        rate.observe(false);
    }
    assert_eq!(rate.rho(), 0.0, "旧的未命中被挤出窗口");
}

/// 一步门控吃的是这一步的特征：短上下文抬 π、稀疏输入抬 γ。
fn routing_reads_the_step_it_is_given() {
    let rate = MissRate::default();
    let dense = vec![1.0f64; 8];
    let sparse = [1.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let short = rate.route(&dense, 16);
    let long = rate.route(&dense, 1 << 20);
    assert!(short.pi() > long.pi(), "短上下文更该走 α 检索");
    assert_eq!(short.channel(), Channel::Alpha);
    assert!(
        rate.route(&sparse, 16).gamma() > short.gamma(),
        "稀疏输入更该走事件臂"
    );
}

/// 通道选择不替代合成：前向走的就是 §1.5 的全合成。
fn mix_is_the_full_synthesis() {
    let gating = Gating::new(0.25f64, 0.5);
    let (a, b, e) = (vec![1.0f64, 2.0], vec![3.0f64, 4.0], vec![5.0f64, 6.0]);
    assert_eq!(
        gating.mix(&a, &b, &e),
        crate::mix::synthesize(&a, &b, &e, 0.25, 0.5)
    );
    assert_eq!(
        Gating::new(0.5f64, 1.0).mix(&a, &b, &e),
        e,
        "γ = 1 只剩事件臂"
    );
    assert_eq!(
        Gating::new(1.0f64, 0.0).mix(&a, &b, &e),
        a,
        "π = 1、γ = 0 只剩 α"
    );
}

/// 短名稳定（诊断、快照与报告按它对齐）。
fn channel_short_names_are_stable() {
    assert_eq!(Channel::Alpha.as_str(), "alpha");
    assert_eq!(Channel::Beta.as_str(), "beta");
    assert_eq!(Channel::Gamma.as_str(), "gamma");
}
}

// ---- 标定口径（原 `tests/policy_tests.rs`）：注册名与三种标定的判别式 ----

const TOL_CALIB: f64 = 1e-9;

testkit::test_cases! {
/// 已注册名解析成功，未注册名失败。
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
fn softmax_sums_to_one() {
    let d = Calib::Softmax.to_dist(&[1.0f64, 2.0, 3.0]);
    assert!((d.iter().sum::<f64>() - 1.0).abs() <= TOL_CALIB);
}

/// 球面校准输出 L2 范数为 1（非负分数）。
fn spherical_has_unit_l2_norm() {
    let d = Calib::Spherical.to_dist(&[3.0f64, 4.0]);
    let norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() <= TOL_CALIB);
}

/// 温度缩放：τ 越小分布越尖锐。
fn temperature_sharpens_distribution() {
    let cold = Calib::Temperature(0.5).to_dist(&[1.0f64, 2.0]);
    let hot = Calib::Temperature(2.0).to_dist(&[1.0f64, 2.0]);
    assert!(cold[1] / cold[0] > hot[1] / hot[0]);
}

/// Brier 评分：命中时为 0（one-hot），未命中为正，值域 `[0,1]`。
fn brier_scores_perfect_as_zero() {
    let p = vec![0.0f64, 1.0, 0.0];
    assert!((brier(&p, 1)).abs() <= TOL_CALIB);
    let wrong = brier(&p, 0);
    assert!(wrong > 0.0 && wrong <= 1.0);
}
}
