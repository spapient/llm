//! `spike` 的用例：前向、代理导数反传与量化感知训练三段。
//!
//! 生产文件只保留执行与判据，"算得对不对"全部由本文件回答，于是增删用例不会把
//! `spike.rs` 推过行数门限。

use super::*;
use crate::mix::Encoder;
use diagnostic::Code;

/// 门槛调到不可能达到、无泄漏：此时膜电位就是这一步的整数累加。
fn quiet(weights: Vec<i8>, scale: Vec<f32>, inputs: usize, outputs: usize) -> Spiking {
    Spiking::new(weights, scale, inputs, outputs, 1.0e6, 0.0, 0.0).expect("几何合法")
}

#[test]
fn integer_accumulation_is_exact_against_the_f32_dot() {
    let layer = quiet(vec![1, -2, 3, 4], vec![1.0, 1.0], 2, 2);
    let mut state = layer.membrane();
    let mut out = [0i8; 2];
    let fired = layer
        .step(&[1, -1], &mut state, &mut out)
        .expect("合法脉冲");
    assert_eq!(fired, 0);
    // 第 0 行：1·1 + (−2)·(−1) = 3；第 1 行：3·1 + 4·(−1) = −1。
    assert_eq!(state.potential()[0].to_bits(), 3.0f32.to_bits());
    assert_eq!(state.potential()[1].to_bits(), (-1.0f32).to_bits());
    assert_eq!(out, [0, 0]);
}

#[test]
fn no_pulses_leave_every_membrane_untouched() {
    let layer = quiet(vec![5, 5], vec![2.0], 2, 1);
    let mut state = layer.membrane();
    let mut out = [9i8; 1];
    for _ in 0..3 {
        let fired = layer
            .step(&[0, 0], &mut state, &mut out)
            .expect("零脉冲合法");
        assert_eq!(fired, 0);
        assert_eq!(out, [0]);
    }
    assert_eq!(state.potential()[0].to_bits(), 0.0f32.to_bits());
}

#[test]
fn a_potential_over_the_threshold_fires_and_resets() {
    let layer = Spiking::new(vec![1], vec![1.0], 1, 1, 1.0, 0.0, -0.5).expect("几何合法");
    let mut state = layer.membrane();
    let mut out = [0i8; 1];
    assert_eq!(layer.step(&[1], &mut state, &mut out).expect("合法"), 1);
    assert_eq!(out, [1]);
    assert_eq!(state.potential()[0].to_bits(), (-0.5f32).to_bits());
}

#[test]
fn leak_shapes_the_membrane_between_steps() {
    let layer = Spiking::new(vec![1], vec![1.0], 1, 1, 4.0, 0.5, 0.0).expect("几何合法");
    let mut state = layer.membrane();
    let mut out = [0i8; 1];
    layer.step(&[1], &mut state, &mut out).expect("合法");
    // 第二步：1·0.5 + 1 = 1.5（无泄漏时会是 2）。
    layer.step(&[1], &mut state, &mut out).expect("合法");
    assert_eq!(state.potential()[0].to_bits(), 1.5f32.to_bits());
}

#[test]
fn the_event_path_and_the_dense_path_agree_bitwise() {
    let layer = quiet(vec![2, -1, 4, 3, -5, 1], vec![0.5, 2.0], 3, 2);
    let events = Encoder::new(0.5f32, 1)
        .expect("参数合法")
        .encode(&[0.0, 0.9, -1.4])
        .expect("合法输入");
    assert_eq!(events.idxs(), &[1, 2]);
    // 事件码的符号：0.9/0.5 → +1，−1.4/0.5 → −2 → 符号 −1。
    let mut driven = layer.membrane();
    let mut sparse = [0i8; 2];
    let sparse_fired = layer
        .drive(&events, &mut driven, &mut sparse)
        .expect("宽度一致");

    let mut dense_state = layer.membrane();
    let mut dense = [0i8; 2];
    let dense_fired = layer
        .step(&[0, 1, -1], &mut dense_state, &mut dense)
        .expect("合法脉冲");

    assert_eq!(sparse_fired, dense_fired);
    assert_eq!(sparse, dense);
    assert_eq!(driven.potential(), dense_state.potential());
}

#[test]
fn the_count_matches_the_emitted_spikes() {
    let layer =
        Spiking::new(vec![1, 1, -1], vec![1.0, 1.0, 1.0], 1, 3, 1.0, 0.0, 0.0).expect("几何合法");
    let mut state = layer.membrane();
    let mut out = [0i8; 3];
    let fired = layer.step(&[-1], &mut state, &mut out).expect("合法脉冲");
    assert_eq!(out, [0, 0, 1]);
    assert_eq!(fired, 1);
    assert_eq!(fired, out.iter().filter(|&&s| s == 1).count());
}

#[test]
fn events_of_another_width_are_rejected() {
    let layer = quiet(vec![1, 1], vec![1.0], 2, 1);
    let events = Encoder::new(1.0f32, 1)
        .expect("参数合法")
        .encode(&[1.0, 1.0, 1.0])
        .expect("合法输入");
    let mut state = layer.membrane();
    let mut out = [0i8; 1];
    assert!(layer.drive(&events, &mut state, &mut out).is_err());
    assert_eq!(state.potential()[0].to_bits(), 0.0f32.to_bits());
}

#[test]
fn geometry_failures_are_rejected_before_any_object_exists() {
    assert!(Spiking::new(vec![1], vec![1.0], 0, 1, 1.0, 0.0, 0.0).is_err());
    assert!(Spiking::new(vec![1], vec![1.0], 1, 0, 1.0, 0.0, 0.0).is_err());
    assert!(Spiking::new(vec![1], vec![1.0], 2, 1, 1.0, 0.0, 0.0).is_err());
    assert!(Spiking::new(vec![1, 1], vec![1.0, 2.0], 1, 1, 1.0, 0.0, 0.0).is_err());
}

#[test]
fn an_input_width_that_would_overflow_the_accumulator_is_rejected() {
    let wide = i32::MAX as usize / WEIGHT_MAGNITUDE + 1;
    assert!(Spiking::new(Vec::new(), vec![1.0], wide, 1, 1.0, 0.0, 0.0).is_err());
}

#[test]
fn dynamics_outside_the_lif_range_are_rejected() {
    let base = (vec![1i8], vec![1.0f32], 1usize, 1usize);
    let build = |threshold, leak, reset| {
        Spiking::new(base.0.clone(), base.1.clone(), 1, 1, threshold, leak, reset)
    };
    assert!(build(0.0, 0.0, 0.0).is_err());
    assert!(build(f32::NAN, 0.0, 0.0).is_err());
    assert!(build(1.0, 1.5, 0.0).is_err());
    assert!(build(1.0, 0.0, 1.0).is_err());
    assert!(build(1.0, 0.0, f32::INFINITY).is_err());
    assert!(Spiking::new(vec![1], vec![0.0], 1, 1, 1.0, 0.0, 0.0).is_err());
    assert!(Spiking::new(vec![1], vec![f32::NAN], 1, 1, 1.0, 0.0, 0.0).is_err());
    assert!(build(1.0, 0.0, -0.5).is_ok());
}

#[test]
fn dense_pulses_outside_the_ternary_set_are_rejected() {
    let layer = quiet(vec![1, 1], vec![1.0], 2, 1);
    let mut state = layer.membrane();
    let mut out = [0i8; 1];
    assert!(layer.step(&[2, 0], &mut state, &mut out).is_err());
    assert!(layer.step(&[0, -3], &mut state, &mut out).is_err());
    assert!(layer.step(&[1], &mut state, &mut out).is_err());
    assert!(layer.step(&[1, 0], &mut state, &mut [0i8; 2]).is_err());
    assert_eq!(state.potential()[0].to_bits(), 0.0f32.to_bits());
}

/// E14 稀疏度与「事件臂 vs 稠密臂」的等效乘加对照（`#[ignore]` 实测：
/// `cargo test -p llm-nn --lib -- spike_arm_costs --release --ignored --nocapture`）。
///
/// 几何冻结在 1024 → 1024，权重与激活都由固定种子的 xorshift 生成，`Encoder` 的码位门槛决定
/// 发放比例——于是「事件臂做了多少次整数累加」与「稠密臂做了多少次」可由 `active` 直接比出来，
/// 不必假设稀疏度。两条臂在同一份权重上用同一段脉冲，墙钟差才是「少做多少个乘加」的实测。
///
/// 注意两份耗时里都含相同的一份固定开销（`outputs` 次反量化 + LIF 推进），因此加速比**低于**
/// 乘加比；这条只报数，判决仍归 `tools::card` 的协议卡。
#[test]
#[ignore = "E14 事件臂实测：cargo test -p llm-nn --lib -- spike_arm_costs --release --ignored --nocapture"]
fn spike_arm_costs() {
    use std::hint::black_box;
    use std::time::Instant;

    let width = 1024usize;
    let mut seed = 0x5eed_f00d_1234_5678u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let weights: Vec<i8> = (0..width * width)
        .map(|_| ((next() % 255) as i16 - 128) as i8)
        .collect();
    let values: Vec<f32> = (0..width)
        .map(|_| ((next() % 2001) as f32 - 1000.0) / 1000.0)
        .collect();
    let events = Encoder::new(0.25f32, 3)
        .expect("参数合法")
        .encode(&values)
        .expect("合法输入");
    let dense: Vec<i8> = events
        .decode()
        .iter()
        .map(|value| value.signum() as i8)
        .collect();

    let layer = quiet(weights, vec![1.0; width], width, width);
    let mut state = layer.membrane();
    let mut out = vec![0i8; width];
    let reps = 100usize;

    let start = Instant::now();
    for _ in 0..reps {
        black_box(layer.step(&dense, &mut state, &mut out).expect("合法脉冲"));
    }
    let dense_ns = start.elapsed().as_secs_f64() / reps as f64;

    let start = Instant::now();
    for _ in 0..reps {
        black_box(
            layer
                .drive(&events, &mut state, &mut out)
                .expect("宽度一致"),
        );
    }
    let event_ns = start.elapsed().as_secs_f64() / reps as f64;

    // 乘加按「一个输出行要对多少个输入位置累加」计：密集实现是 `inputs`，事件实现是
    // `active`，两者都再乘 `outputs`——比因此等于 `1 − S`，这正是 E14 的定义。
    let active = events.active();
    let dense_macs = width * width;
    let event_macs = active * width;
    println!(
        "E14 稀疏度 S={:.4}（active={active}/{width}） 乘加 dense={dense_macs} event={event_macs} 比={:.4}",
        events.sparsity(),
        event_macs as f64 / dense_macs as f64
    );
    println!(
        "E13 索引开销对照（同一份整数算术）dense={:.1} ms  event={:.1} ms  墙钟比={:.2}×",
        dense_ns * 1e3,
        event_ns * 1e3,
        dense_ns / event_ns
    );
}

#[test]
fn a_membrane_from_another_layer_is_rejected() {
    let wide = quiet(vec![1, 1, 1, 1], vec![1.0, 1.0], 2, 2);
    let narrow = quiet(vec![1, 1], vec![1.0], 2, 1);
    let mut state = wide.membrane();
    let mut out = [0i8; 1];
    assert!(narrow.step(&[1, 1], &mut state, &mut out).is_err());
    assert_eq!(state.potential()[0].to_bits(), 0.0f32.to_bits());
}

/// 单通道单输入：门槛 1、无泄漏、重置 0，整数累加直接就是膜电位。
fn single() -> Spiking {
    Spiking::new(vec![1], vec![1.0], 1, 1, 1.0, 0.0, 0.0).expect("几何合法")
}

#[test]
fn the_step_leaves_the_pre_commit_potential_behind_for_the_backward_pass() {
    let layer = single();
    let mut state = layer.membrane();
    let mut out = [0i8; 1];
    // 累加 1 → 电位 1，达到门槛并发放，于是 `potential()` 当场变成重置值 0。
    layer.step(&[1], &mut state, &mut out).expect("合法脉冲");
    assert_eq!(out, [1]);
    assert_eq!(state.potential()[0].to_bits(), 0.0f32.to_bits());
    // 反传要的是发放前那个 1，不是重置后那个 0。
    assert_eq!(state.pre()[0].to_bits(), 1.0f32.to_bits());
}

#[test]
fn a_second_step_overwrites_the_potential_the_backward_pass_reads() {
    // 有泄漏：第二步的发放前电位是「第一步残留 × 0.5 + 电流」。
    let layer = Spiking::new(vec![1], vec![1.0], 1, 1, 4.0, 0.5, 0.0).expect("几何合法");
    let mut state = layer.membrane();
    let mut out = [0i8; 1];
    layer.step(&[1], &mut state, &mut out).expect("合法脉冲");
    assert_eq!(state.pre()[0].to_bits(), 1.0f32.to_bits());
    layer.step(&[1], &mut state, &mut out).expect("合法脉冲");
    assert_eq!(state.pre()[0].to_bits(), 1.5f32.to_bits());
}

#[test]
fn both_surrogate_shapes_have_unit_area_and_nothing_outside_the_window() {
    let rect = Surrogate::Rect { width: 0.5 };
    let triangle = Surrogate::Triangle { width: 0.5 };
    let steps = 1000;
    for shape in [rect, triangle] {
        let mut area = 0.0f32;
        for idx in 0..steps {
            let offset = -0.5 + (idx as f32 + 0.5) / steps as f32;
            area += shape.derivative(offset, 0.0) / steps as f32;
        }
        assert!((area - 1.0).abs() < 1e-2, "{shape:?} 的面积 {area}");
    }
    for shape in [rect, triangle] {
        // 恰好压在边上不算窗内；`NaN` 同样在窗外（否则它会拿到一个假导数）。
        assert_eq!(shape.derivative(0.5, 0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(shape.derivative(-0.5, 0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(shape.derivative(f32::NAN, 0.0).to_bits(), 0.0f32.to_bits());
    }
    // 峰在阈值上：矩形是 1/(2w)，三角形是 1/w——面积相同，摊法不同。
    assert_eq!(rect.derivative(0.0, 0.0).to_bits(), 1.0f32.to_bits());
    assert_eq!(triangle.derivative(0.0, 0.0).to_bits(), 2.0f32.to_bits());
    assert_eq!(rect.width().to_bits(), triangle.width().to_bits());
}

#[test]
fn the_gradients_follow_the_analytic_chain() {
    let layer = single();
    let mut state = layer.membrane();
    let mut out = [0i8; 1];
    layer.step(&[1], &mut state, &mut out).expect("合法脉冲");
    let mut grad = layer.gradients();
    // 峰上的三角形窗：∂s/∂v = 1/w = 1；d_out = 2。
    layer
        .backward(
            &[1],
            &state,
            Surrogate::Triangle { width: 1.0 },
            &[2.0],
            &mut grad,
        )
        .expect("入口合法");
    // 电流 2 × 1 = 2；尺度 2 × acc(1) = 2；权重 2 × scale(1) × spike(1) = 2。
    assert_eq!(grad.current()[0].to_bits(), 2.0f32.to_bits());
    assert_eq!(grad.scale()[0].to_bits(), 2.0f32.to_bits());
    assert_eq!(grad.weights()[0].to_bits(), 2.0f32.to_bits());
}

#[test]
fn a_silent_input_contributes_nothing_to_the_scale_or_the_weights() {
    let layer = single();
    let mut state = layer.membrane();
    let mut out = [0i8; 1];
    layer.step(&[0], &mut state, &mut out).expect("合法脉冲");
    let mut grad = layer.gradients();
    // 窗口放宽到覆盖电位，于是 d_out 确实进了电流；但 acc 与 spike 都是零。
    layer
        .backward(
            &[0],
            &state,
            Surrogate::Rect { width: 2.0 },
            &[3.0],
            &mut grad,
        )
        .expect("入口合法");
    assert_eq!(grad.current()[0].to_bits(), 0.75f32.to_bits());
    assert_eq!(grad.scale()[0].to_bits(), 0.0f32.to_bits());
    assert_eq!(grad.weights()[0].to_bits(), 0.0f32.to_bits());
}

#[test]
fn channels_outside_the_window_take_no_gradient_at_all() {
    // 门槛高到不可能达到：电位离阈值极远，整层都在窗外。
    let layer = quiet(vec![1], vec![1.0], 1, 1);
    let mut state = layer.membrane();
    let mut out = [0i8; 1];
    layer.step(&[1], &mut state, &mut out).expect("合法脉冲");
    let mut grad = layer.gradients();
    layer
        .backward(
            &[1],
            &state,
            Surrogate::Triangle { width: 1.0 },
            &[5.0],
            &mut grad,
        )
        .expect("入口合法");
    for values in [grad.weights(), grad.scale(), grad.current()] {
        assert!(values.iter().all(|value| value.to_bits() == 0));
    }
}

#[test]
fn the_backward_step_rejects_bad_inputs_before_touching_the_buf() {
    let layer = single();
    let mut state = layer.membrane();
    let mut out = [0i8; 1];
    layer.step(&[1], &mut state, &mut out).expect("合法脉冲");
    let mut grad = layer.gradients();
    let window = Surrogate::Rect { width: 1.0 };
    let rejects = [
        layer.backward(&[1, 1], &state, window, &[1.0], &mut grad),
        layer.backward(&[2], &state, window, &[1.0], &mut grad),
        layer.backward(&[1], &state, window, &[1.0, 1.0], &mut grad),
        layer.backward(&[1], &state, window, &[f32::NAN], &mut grad),
        layer.backward(
            &[1],
            &state,
            Surrogate::Rect { width: 0.0 },
            &[1.0],
            &mut grad,
        ),
    ];
    for result in rejects {
        assert_eq!(result.unwrap_err().code(), Code::Invalid);
    }
    assert_eq!(grad.weights()[0].to_bits(), 0.0f32.to_bits());
}

#[test]
fn the_master_quantizes_by_rounding_and_clamps_to_the_code_range() {
    // 1.6 → 2、−1.4 → −1；超出 `i8` 对称范围的两个都夹到 ±127，不回绕成别的码。
    let master = Master::new(
        vec![1.6, -1.4, 200.0, -300.0],
        vec![1.0],
        4,
        1,
        1.0,
        0.0,
        0.0,
    )
    .expect("几何合法");
    let view = master.view().expect("量化视图可构造");
    assert_eq!(view.weights, [2, -1, 127, -127]);
    // 视图带走同一份尺度，不在量化这一步改写它。
    assert_eq!(view.scale, master.scale);
}

#[test]
fn a_training_step_moves_the_float_weights_and_leaves_the_codes_to_follow() {
    let mut master =
        Master::new(vec![0.4, -0.4], vec![1.0], 2, 1, 1.0, 0.0, 0.0).expect("几何合法");
    // 视图把 0.4 / −0.4 都读成 0：这正是 QAT 存在的理由——梯度落在浮点侧，码位下一拍再取。
    assert_eq!(master.view().expect("可量化").weights, [0, 0]);
    let mut grad = master.view().expect("可量化").gradients();
    grad.weights.copy_from_slice(&[1.0, -2.0]);
    grad.scale[0] = 0.5;
    master.descend(&grad, 0.5).expect("步长合法");
    for (got, want) in master.weights().iter().zip([0.4f32 - 0.5, -0.4 + 1.0]) {
        assert_eq!(got.to_bits(), want.to_bits());
    }
    assert_eq!(master.scale()[0].to_bits(), 0.75f32.to_bits());
    // 权重变成 −0.1 / 0.6：只有跨过半个码位的那个换码，落在 0 附近的那个仍是 0。
    assert_eq!(master.view().expect("可量化").weights, [0, 1]);
}

#[test]
fn a_step_that_would_make_a_scale_non_positive_is_rejected_without_writing() {
    let mut master = Master::new(vec![0.4], vec![0.1], 1, 1, 1.0, 0.0, 0.0).expect("几何合法");
    let before = master.clone();
    let mut grad = master.view().expect("可量化").gradients();
    grad.weights[0] = 1.0;
    grad.scale[0] = 1.0;
    // 0.1 − 1.0 × 0.5 < 0：步长把量化区间走穿了。权重一格也不能动，否则半截状态要等到下一次
    // `view` 才被发现。
    assert_eq!(
        master.descend(&grad, 0.5).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(master, before);
}

#[test]
fn the_master_rejects_a_non_finite_weight_before_it_becomes_a_code() {
    // `NaN` 经 `round().clamp()` 会落成某个「看着正常」的 `i8`，从视图上再也认不出来。
    let bad_master = Master::new(vec![f32::NAN], vec![1.0], 1, 1, 1.0, 0.0, 0.0);
    assert_eq!(bad_master.unwrap_err().code(), Code::Invalid);
    let bad_scale = Master::new(vec![1.0], vec![0.0], 1, 1, 1.0, 0.0, 0.0);
    assert_eq!(bad_scale.unwrap_err().code(), Code::Invalid);
    let bad_geometry = Master::new(vec![1.0, 1.0], vec![1.0], 1, 1, 1.0, 0.0, 0.0);
    assert_eq!(bad_geometry.unwrap_err().code(), Code::Invalid);
}

#[test]
fn a_non_positive_learning_rate_is_rejected() {
    let mut master = Master::new(vec![1.0], vec![1.0], 1, 1, 1.0, 0.0, 0.0).expect("几何合法");
    let grad = master.view().expect("可量化").gradients();
    for lr in [0.0, -0.1, f32::NAN] {
        assert_eq!(
            master.descend(&grad, lr).unwrap_err().code(),
            Code::Invalid,
            "lr = {lr}"
        );
    }
    assert_eq!(master.weights()[0].to_bits(), 1.0f32.to_bits());
}

#[test]
fn a_gradient_buf_of_another_shape_is_rejected() {
    let wide = quiet(vec![1, 1, 1, 1], vec![1.0, 1.0], 2, 2);
    let layer = quiet(vec![1, 1], vec![1.0], 2, 1);
    let mut state = layer.membrane();
    let mut out = [0i8; 1];
    layer.step(&[1, 1], &mut state, &mut out).expect("合法脉冲");
    // 缓冲来自另一层：行长与 `outputs` 都对不上，静默写下去会把梯度落到别的行上。
    let mut foreign = wide.gradients();
    assert!(
        layer
            .backward(
                &[1, 1],
                &state,
                Surrogate::Rect { width: 1.0 },
                &[1.0],
                &mut foreign,
            )
            .is_err()
    );
}

/// QAT 一步不是摆设：梯度落在浮点侧，几十步之后量化重建误差真的落进量化底噪。
///
/// 目标是一条 8 维线性行，初值全零、尺度固定 1（对称 `i8` 一格就是 1，所以底噪是「每格 ±0.5」）。
/// 梯度取 `2·(w − target)`——这是这条残差自己的导数，不是从别处搬来的近似；`descend` 只动浮点
/// 权重，码位在下一拍 `view` 时才重新取。于是「训练在改变什么」在这条用例里是可直接读出来的：
/// 训练前的量化误差由初值占据，训练后的残差只剩 `round()` 那半格。
///
/// 跑：`cargo test -p llm-nn --lib --release -- qat_costs --ignored --nocapture`
#[test]
#[ignore = "训练闭环实测：cargo test -p llm-nn --lib -- qat_costs --ignored --nocapture"]
fn qat_costs() {
    const TARGET: [f32; 8] = [0.8, -1.2, 2.4, -0.6, 1.0, -2.0, 0.3, 1.7];
    let mut master = Master::new(vec![0.0; 8], vec![1.0], 8, 1, 1.0, 0.0, 0.0).expect("几何合法");
    let mse = |m: &Master| {
        let view = m.view().expect("可量化");
        let sum: f32 = view
            .weights
            .iter()
            .zip(TARGET)
            .map(|(code, want)| (f32::from(*code) - want).powi(2))
            .sum();
        sum / TARGET.len() as f32
    };

    println!("step  0: mse = {:.4}", mse(&master));
    for step in 1..=64 {
        let view = master.view().expect("可量化");
        let mut grad = view.gradients();
        for (slot, (weight, want)) in grad
            .weights
            .iter_mut()
            .zip(master.weights().iter().zip(TARGET))
        {
            *slot = 2.0 * (weight - want);
        }
        master.descend(&grad, 0.25).expect("步长合法");
        if step % 16 == 0 {
            println!("step {step:>2}: mse = {:.4}", mse(&master));
        }
    }

    // 64 步之后每条权重都应贴在目标的那一格上：剩下的误差只可能来自 `round()` 的半格。
    let settled = mse(&master);
    assert!(settled < 0.25, "训练后应进到量化底噪内，得到 {settled}");
}
