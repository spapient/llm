//! 原先内联在 `src/policy.rs` 的测试。
//!
//! 用 `#[path]` 挂回原模块，模块树位置不变，故 `super::`/`crate::` 路径照旧。

use super::*;

const TOL: f64 = 1e-12;

/// 门控始终落在 `[0,1]`。
#[test]
fn gates_are_clipped() {
    assert_eq!(pi(-1.0f64, 0.0), 0.0);
    assert_eq!(pi(2.0, 0.0), 1.0);
    assert_eq!(gamma(-0.5f64), 0.0);
    assert_eq!(gamma(1.5), 1.0);
}

/// 短上下文（ν 小）抬 π，长上下文（ν 大）压 π。
#[test]
fn short_context_raises_pi() {
    assert!(pi(0.0, 0.0) > pi(0.0, 4.0));
}

/// RMS 与 τ 的比例关系。
#[test]
fn tau_is_scaled_rms() {
    let x = vec![1.0f64, 2.0, 3.0, 4.0];
    let expected = rms(&x) * TAU_SCALE;
    assert!((tau(&x) - expected).abs() <= TOL);
}

/// 稀疏度统计 `|xᵢ| < τ` 的占比。
#[test]
fn sparsity_counts_below_tau() {
    let x = vec![0.1f64, 0.2, 5.0, 6.0];
    assert!((sparsity(&x, 1.0) - 0.5).abs() <= TOL);
}

/// 空输入退化为零。
#[test]
fn empty_input_is_zero() {
    assert_eq!(rms::<f64>(&[]), 0.0);
    assert_eq!(sparsity::<f64>(&[], 1.0), 0.0);
}

/// 系数是单位剖分：三项非负且和为 1。
#[test]
fn gating_coefficients_partition_unity() {
    for (pi_t, gamma_t) in [(0.0f64, 0.0), (0.25, 0.5), (1.0, 0.0), (0.7, 1.0)] {
        let c = Gating::new(pi_t, gamma_t).coefficients();
        assert!(c.iter().all(|&w| w >= 0.0), "{c:?}");
        assert!((c.iter().sum::<f64>() - 1.0).abs() <= TOL, "{c:?}");
    }
}

/// 主导通道 = 系数最大者，等值按 α、β、γ 的声明次序。
#[test]
fn the_largest_coefficient_picks_the_channel() {
    assert_eq!(Gating::new(1.0f64, 0.0).channel(), Channel::Alpha);
    assert_eq!(Gating::new(0.0f64, 0.0).channel(), Channel::Beta);
    assert_eq!(Gating::new(1.0f64, 0.9).channel(), Channel::Gamma);
    assert_eq!(Gating::new(0.5f64, 0.0).channel(), Channel::Alpha);
}

/// 在线窗口只保留最近 `W` 步，比率随之滑动。
#[test]
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
#[test]
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
#[test]
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
#[test]
fn channel_short_names_are_stable() {
    assert_eq!(Channel::Alpha.as_str(), "alpha");
    assert_eq!(Channel::Beta.as_str(), "beta");
    assert_eq!(Channel::Gamma.as_str(), "gamma");
}
