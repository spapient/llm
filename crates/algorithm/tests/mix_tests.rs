//! `mix` 的用例：混合、门控与合成分解。
//!
//! 生产文件只留执行与判据，用例集中于此。

use super::*;

use testkit::assert_close;

const TOL: f64 = 1e-12;

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
    assert_close(&synthesize(&ya, &yb, &ya, 0.25, 0.0), &yc, TOL, 0.0);
}

/// γ=1 时输出就是事件通道。
#[test]
fn gamma_one_is_event() {
    let (ya, yb, yg) = (vec![1.0, 2.0], vec![3.0, 4.0], vec![0.0, 1.0]);
    assert_close(&synthesize(&ya, &yb, &yg, 0.25, 1.0), &yg, TOL, 0.0);
}

/// IEEE 恒等编码下（y_γ = y_cont），γ 混合与纯连续一致（F0-2）。
#[test]
fn ieee_gamma_mix_is_continuous() {
    let (ya, yb) = (vec![1.0, -2.0], vec![3.0, 4.0]);
    let yc = cont(&ya, &yb, 0.4);
    for gamma in [0.0, 0.3, 0.7, 1.0] {
        assert_close(&synthesize(&ya, &yb, &yc, 0.4, gamma), &yc, TOL, 0.0);
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

// ---- `mix::pulse`：事件编码与 INT8-Spiking 事件臂 ----

/// 阈值以下积分不发放，膜电位累加。
#[test]
fn integrates_without_spike_below_threshold() {
    let mut lif = Lif::new(1.0f64, 0.0, 0.0);
    assert_eq!(lif.step(0.3), 0.0);
    assert_eq!(lif.step(0.3), 0.0);
    assert!((lif.mem() - 0.6).abs() < 1e-12);
}

/// 达到阈值发放并重置。
#[test]
fn spikes_and_resets_at_threshold() {
    let mut lif = Lif::new(1.0f64, 0.0, 0.0);
    assert_eq!(lif.step(0.6), 0.0);
    assert_eq!(lif.step(0.6), 1.0, "超过阈值发放");
    assert!((lif.mem() - 0.0).abs() < 1e-12, "发放后重置");
}

/// 泄漏使膜电位随时间衰减。
#[test]
fn leak_decays_membrane() {
    let mut lif = Lif::new(1.0f64, 0.5, 0.0);
    lif.step(0.8);
    let before = lif.mem();
    lif.step(0.0);
    assert!(lif.mem() < before, "泄漏后膜电位下降");
}

/// IEEE 恒等编码逐元素不变。
#[test]
fn encode_id_is_identity() {
    let y = vec![1.0f64, -2.0, 3.0];
    assert_eq!(encode_id(&y), y);
}

/// 编码器：步长与门槛各自越界都被拒，边界值放行。
#[test]
fn encoder_params_outside_the_lattice_are_rejected() {
    assert!(Encoder::new(0.0f64, 1).is_err());
    assert!(Encoder::new(-0.5f64, 1).is_err());
    assert!(Encoder::new(f64::NAN, 1).is_err());
    assert!(Encoder::new(0.5f64, 0).is_err());
    assert!(Encoder::new(0.5f64, 128).is_err());
    assert!(Encoder::new(0.5f64, 127).is_ok());
}

/// 只有码位达到门槛的位置发放：门槛按码位而不是按浮点值。
#[test]
fn only_positions_at_or_above_the_threshold_fire() {
    let enc = Encoder::new(0.5f64, 2).expect("参数合法");
    let events = enc.encode(&[0.2, 0.6, -0.9, 3.0]).expect("合法输入");
    assert_eq!(events.width(), 4);
    assert_eq!(events.active(), 2);
    assert_eq!(events.idxs(), &[2, 3]);
    assert_eq!(events.codes(), &[-2i8, 6]);
    assert!((events.sparsity() - 0.5).abs() < 1e-12);
}

/// 超量程夹取，不回绕：回绕后的数仍像一个数。
#[test]
fn out_of_range_values_clip_into_the_code_lattice() {
    let enc = Encoder::new(1.0f64, 1).expect("参数合法");
    let events = enc.encode(&[1.0e6, -1.0e6, 300.0]).expect("合法输入");
    assert_eq!(events.codes(), &[127i8, -127, 127]);
    assert_eq!(events.decode(), vec![127.0, -127.0, 127.0]);
}

/// 还原只需要事件本身：同一组码配不同步长解出不同的数。
#[test]
fn decode_takes_its_scale_from_the_events() {
    let coarse = Encoder::new(1.0f64, 1)
        .expect("参数合法")
        .encode(&[1.0])
        .expect("合法输入");
    let fine = Encoder::new(0.25f64, 1)
        .expect("参数合法")
        .encode(&[0.25])
        .expect("合法输入");
    assert_eq!(coarse.codes(), fine.codes());
    assert_eq!(coarse.decode(), vec![1.0]);
    assert_eq!(fine.decode(), vec![0.25]);
}

/// 实测两项：稀疏度 `1 − N_active/N_total` 与相对重构损失。
#[test]
fn audit_reports_sparsity_and_relative_loss() {
    let enc = Encoder::new(0.5f64, 1).expect("参数合法");
    let audit = Audit::measure(&enc, &[0.6, 0.0, 0.0]).expect("合法输入");
    assert!((audit.sparsity() - 2.0 / 3.0).abs() < 1e-12);
    // 还原成 0.5：残差 0.1，相对损失 = 0.1 / 0.6 = 1/6。
    assert!((audit.loss() - 1.0 / 6.0).abs() < 1e-12);
}

/// 零输入没有相对误差可言：记 0 而不是 `0/0`。
#[test]
fn zero_input_has_no_relative_error() {
    let enc = Encoder::new(0.5f64, 1).expect("参数合法");
    let audit = Audit::measure(&enc, &[0.0f64; 3]).expect("合法输入");
    assert_eq!(audit.sparsity(), 1.0);
    assert_eq!(audit.loss(), 0.0);
}

/// 重合的一对没有距离，放大比无定义——记 `None` 而不是 0。
#[test]
fn coincident_pairs_give_no_gain_estimate() {
    let enc = Encoder::new(0.5f64, 1).expect("参数合法");
    let u = [0.7f64, 1.1];
    assert_eq!(steepest(&enc, &[(u.as_slice(), u.as_slice())]), Ok(None));
    assert_eq!(steepest(&enc, &[]), Ok(None));
}

/// 粗间距比值不超过 1；细间距跨码位台阶时超过 1——量化器不是真 Lipschitz。
#[test]
fn fine_pairs_exceed_the_lipschitz_constant() {
    let enc = Encoder::new(0.5f64, 1).expect("参数合法");
    let coarse = steepest(&enc, &[(&[0.0f64][..], &[1.0f64][..])]).expect("合法");
    assert_eq!(coarse, Some(1.0));
    let fine = steepest(&enc, &[(&[0.0f64][..], &[0.3f64][..])])
        .expect("合法")
        .expect("可比的一对");
    assert!(fine > 1.0, "细间距的放大比 {fine} 应大于 1");
    assert!((fine - 0.5 / 0.3).abs() < 1e-9);
}

/// 事件驱动累加与「补零后整行累加」逐位相同。
#[test]
fn accumulate_agrees_with_the_dense_path() {
    let enc = Encoder::new(0.5f64, 1).expect("参数合法");
    let events = enc.encode(&[0.2, 1.6, -0.1, 0.9]).expect("合法输入");
    let w = [3i8, -2, 5, 1, -4, 7, 0, 2];
    let got = accumulate(&w, 2, 4, &events).expect("几何合法");

    let mut table = [0i8; 4];
    for (&at, &code) in events.idxs().iter().zip(events.codes()) {
        table[at] = code;
    }
    let dense: Vec<i32> = (0..2)
        .map(|row| {
            (0..4)
                .map(|col| i32::from(w[row * 4 + col]) * i32::from(table[col]))
                .sum()
        })
        .collect();
    assert_eq!(got, dense);
    // 码表是 `[0, 3, 0, 2]`：逐行 -2·3 + 1·2 与 7·3 + 2·2。
    assert_eq!(got, vec![-4, 25]);
}

/// 权重长度或事件宽度不符都在入口被拒。
#[test]
fn accumulate_rejects_bad_geometry() {
    let enc = Encoder::new(0.5f64, 1).expect("参数合法");
    let two = enc.encode(&[1.0f64, 0.0]).expect("合法输入");
    assert!(accumulate(&[1i8; 3], 2, 2, &two).is_err());
    let three = enc.encode(&[1.0f64; 3]).expect("合法输入");
    assert!(accumulate(&[1i8; 4], 2, 2, &three).is_err());
}

/// 非有限值不被拾成码位 0：编码、对拍与实测三处都拒。
#[test]
fn non_finite_input_is_rejected_not_absorbed() {
    let enc = Encoder::new(0.5f64, 1).expect("参数合法");
    assert!(enc.encode(&[1.0, f64::NAN]).is_err());
    assert!(steepest(&enc, &[(&[1.0f64][..], &[f64::INFINITY][..])]).is_err());
    assert!(Audit::measure(&enc, &[f64::NAN]).is_err());
}

/// `γ = 0` 时事件臂原样放行输入，`γ = 1` 时恰是解码结果。
#[test]
fn blend_endpoints_are_the_input_and_the_decoded_events() {
    let enc = Encoder::new(0.5f64, 2).expect("参数合法");
    let y = vec![0.1f64, 2.5, -1.2];
    assert_eq!(enc.blend(&y, 0.0).expect("合法输入"), y);
    let events = enc.encode(&y).expect("合法输入");
    assert_eq!(enc.blend(&y, 1.0).expect("合法输入"), events.decode());
}

/// `f32` 与 `f64` 落在同一组码位上。
#[test]
fn f32_and_f64_quantize_to_the_same_codes() {
    let wide = [0.2f64, 1.6, -0.7, 2.4];
    let narrow = [0.2f32, 1.6, -0.7, 2.4];
    let a = Encoder::new(0.5f64, 1).expect("合法").encode(&wide);
    let b = Encoder::new(0.5f32, 1).expect("合法").encode(&narrow);
    let (a, b) = (a.expect("合法输入"), b.expect("合法输入"));
    assert_eq!(a.idxs(), b.idxs());
    assert_eq!(a.codes(), b.codes());
}
