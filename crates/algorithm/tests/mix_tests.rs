//! `mix` 的用例：混合、门控与合成分解。
//!
//! 生产文件只留执行与判据，用例集中于此。

use super::*;

const TOL: f64 = 1e-12;

fn close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= TOL, "{x} vs {y}");
    }
}

/// 系数构成单位剖分：和为 1。
#[test]
fn coefficients_partition_unity() {
    for pi in [0.0, 0.25, 0.5, 1.0] {
        for gamma in [0.0, 0.25, 0.5, 1.0] {
            let c_alpha = (1.0 - gamma) * pi;
            let c_beta = (1.0 - gamma) * (1.0 - pi);
            let c_gamma = gamma;
            assert!((c_alpha + c_beta + c_gamma - 1.0).abs() <= TOL);
        }
    }
}

/// γ=0 时输出就是连续合成。
#[test]
fn gamma_zero_is_continuous() {
    let (ya, yb) = (vec![1.0, 2.0], vec![3.0, 4.0]);
    let yc = cont(&ya, &yb, 0.25);
    close(&synthesize(&ya, &yb, &ya, 0.25, 0.0), &yc);
}

/// γ=1 时输出就是事件通道。
#[test]
fn gamma_one_is_event() {
    let (ya, yb, yg) = (vec![1.0, 2.0], vec![3.0, 4.0], vec![0.0, 1.0]);
    close(&synthesize(&ya, &yb, &yg, 0.25, 1.0), &yg);
}

/// IEEE 恒等编码下（y_γ = y_cont），γ 混合与纯连续一致（F0-2）。
#[test]
fn ieee_gamma_mix_is_continuous() {
    let (ya, yb) = (vec![1.0, -2.0], vec![3.0, 4.0]);
    let yc = cont(&ya, &yb, 0.4);
    for gamma in [0.0, 0.3, 0.7, 1.0] {
        close(&synthesize(&ya, &yb, &yc, 0.4, gamma), &yc);
    }
}

/// 融合写法必须与两步合成逐位一致：省的是分配，不是精度。
#[test]
fn synthesized_matches_the_two_step_composition() {
    let ya: Vec<f64> = vec![0.5, -1.25, 3.0, 0.0, 1e-9];
    let yb: Vec<f64> = vec![2.0, 0.75, -0.5, 4.0, 1.0];
    let yg: Vec<f64> = vec![-1.0, 0.125, 2.5, 1.5, -3.0];
    for pi in [0.0, 0.125, 0.5, 0.875, 1.0] {
        for gamma in [0.0, 0.125, 0.5, 0.875, 1.0] {
            let fused = synthesize(&ya, &yb, &yg, pi, gamma);
            let two_step = gate(&cont(&ya, &yb, pi), &yg, gamma);
            assert_eq!(fused, two_step, "pi={pi} gamma={gamma}");
        }
    }
}

/// 输出落在各通道值的凸包（min/max）内。
#[test]
fn output_in_convex_hull() {
    let ya: Vec<f64> = vec![1.0, 5.0];
    let yb: Vec<f64> = vec![3.0, 2.0];
    let yg: Vec<f64> = vec![9.0, -1.0];
    for pi in [0.0, 0.5, 1.0] {
        for gamma in [0.0, 0.5, 1.0] {
            let y = synthesize(&ya, &yb, &yg, pi, gamma);
            for (i, &v) in y.iter().enumerate() {
                let lo = ya[i].min(yb[i]).min(yg[i]);
                let hi = ya[i].max(yb[i]).max(yg[i]);
                assert!(v >= lo && v <= hi, "v={v} 不在 [{lo},{hi}]");
            }
        }
    }
}
