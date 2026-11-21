//! `mamba` 的用例。
//!
//! 生产文件只留执行与判据，用例集中于此；N3 行数门限不再受用例增删影响。

use super::*;

const WIDTH: usize = 3;

/// 一层：λ 实部 `−0.5`、`ω = 0.4`、直连 `1.0`，参数逐维都相同。
fn layer(pairs: usize, rank: usize) -> Mamba<f64> {
    let n = WIDTH * pairs;
    Mamba::new(
        WIDTH,
        pairs,
        rank,
        vec![-0.5; n],
        vec![0.4; n],
        vec![1.0; WIDTH],
    )
    .expect("参数合法")
}

/// 一层只跑一个通道（宽度 1），用来做闭式对照。
fn single(decay: f64, phase: f64, skip: f64) -> Mamba<f64> {
    Mamba::new(1, 1, 1, vec![decay], vec![phase], vec![skip]).expect("参数合法")
}

fn units(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

#[test]
fn new_rejects_bad_geometry() {
    let ok = (WIDTH, 2, 1, 6usize, 3usize);
    assert!(Mamba::new(0, 2, 1, vec![0.0; 2], vec![0.0; 2], vec![0.0; ok.0]).is_err());
    assert!(Mamba::new(WIDTH, 0, 1, vec![], vec![], vec![0.0; WIDTH]).is_err());
    assert!(
        Mamba::new(
            WIDTH,
            1,
            0,
            vec![0.0; WIDTH],
            vec![0.0; WIDTH],
            vec![0.0; WIDTH]
        )
        .is_err()
    );
    // 秩超过状态维数。
    assert!(
        Mamba::new(
            WIDTH,
            1,
            2,
            vec![0.0; WIDTH],
            vec![0.0; WIDTH],
            vec![0.0; WIDTH]
        )
        .is_err()
    );
    // 三组参数各自短一格。
    assert!(Mamba::new(WIDTH, 2, 1, vec![0.0; 5], vec![0.0; ok.3], vec![0.0; WIDTH]).is_err());
    assert!(Mamba::new(WIDTH, 2, 1, vec![0.0; ok.3], vec![0.0; 5], vec![0.0; WIDTH]).is_err());
    assert!(
        Mamba::new(
            WIDTH,
            2,
            1,
            vec![0.0; ok.3],
            vec![0.0; ok.3],
            vec![0.0; ok.4 - 1]
        )
        .is_err()
    );
}

#[test]
fn new_rejects_unstable_decay_and_non_finite() {
    assert!(Mamba::new(1, 1, 1, vec![0.1], vec![0.0], vec![0.0]).is_err());
    assert!(Mamba::new(1, 1, 1, vec![f64::NAN], vec![0.0], vec![0.0]).is_err());
    assert!(Mamba::new(1, 1, 1, vec![0.0], vec![f64::INFINITY], vec![0.0]).is_err());
    assert!(Mamba::new(1, 1, 1, vec![0.0], vec![0.0], vec![f64::NAN]).is_err());
    // λ 实部为 0 合法：纯旋转，不放大也不衰减（旋转用例要用它）。
    assert!(Mamba::new(1, 1, 1, vec![0.0], vec![0.7], vec![0.0]).is_ok());
}

#[test]
fn state_is_zero_and_its_size_ignores_the_sequence() {
    let m = layer(4, 2);
    let mut carry = m.state();
    assert_eq!(carry.planes().len(), WIDTH * 4 * 2);
    assert!(carry.planes().iter().all(|&v| v == 0.0));
    assert_eq!(carry.taps(), &[0.0; WIDTH]);

    let delta = units(WIDTH);
    let (b, c) = (units(2 * 4), units(2 * 4));
    for _ in 0..128 {
        m.step(&mut carry, &units(WIDTH), &delta, &b, &c)
            .expect("合法一步");
    }
    // 跑过 128 个 token，状态长度一格不变——这是「与序列长无关」的可测形态。
    assert_eq!(carry.planes().len(), WIDTH * 4 * 2);
    assert_eq!(carry.taps().len(), WIDTH);
}

#[test]
fn step_rejects_bad_widths_and_step_sign() {
    let m = layer(2, 1);
    let mut carry = m.state();
    let delta = units(WIDTH);
    let (b, c) = (units(2), units(2));
    assert!(
        m.step(&mut carry, &units(WIDTH - 1), &delta, &b, &c)
            .is_err()
    );
    assert!(
        m.step(&mut carry, &units(WIDTH), &units(WIDTH + 1), &b, &c)
            .is_err()
    );
    assert!(
        m.step(&mut carry, &units(WIDTH), &delta, &units(1), &c)
            .is_err()
    );
    assert!(
        m.step(&mut carry, &units(WIDTH), &delta, &b, &units(3))
            .is_err()
    );
    assert!(
        m.step(&mut carry, &units(WIDTH), &[0.5, 0.0, 0.5], &b, &c)
            .is_err()
    );
    assert!(
        m.step(&mut carry, &units(WIDTH), &[0.5, -0.5, 0.5], &b, &c)
            .is_err()
    );
    assert!(
        m.step(&mut carry, &[1.0, f64::NAN, 1.0], &delta, &b, &c)
            .is_err()
    );
}

#[test]
fn a_failed_step_leaves_the_state_untouched() {
    let m = layer(2, 1);
    let mut carry = m.state();
    let delta = units(WIDTH);
    let (b, c) = (units(2), units(2));
    m.step(&mut carry, &units(WIDTH), &delta, &b, &c)
        .expect("合法一步");
    let before = carry.clone();
    // 步长非正：在更新之前就被拒。
    assert!(
        m.step(&mut carry, &units(WIDTH), &[1.0, 0.0, 1.0], &b, &c)
            .is_err()
    );
    assert_eq!(carry, before);
    // 选择量长度不符同样在写之前。
    assert!(
        m.step(&mut carry, &units(WIDTH), &delta, &units(9), &c)
            .is_err()
    );
    assert_eq!(carry, before);
}

#[test]
fn state_from_another_layer_is_rejected() {
    let narrow = layer(1, 1);
    let wide = layer(3, 1);
    let mut carry = narrow.state();
    let delta = units(WIDTH);
    assert!(
        wide.step(&mut carry, &units(WIDTH), &delta, &units(3), &units(3))
            .is_err()
    );
}

/// 读数是**更新之后**的状态：从零状态出发，一步的闭式结果是 `C·(dt/2·x·B) + D·x`。
#[test]
fn output_reads_the_updated_state() {
    let m = single(-0.5, 0.4, 1.0);
    let mut carry = m.state();
    let (x, dt) = (2.0f64, 0.25f64);
    let y = m
        .step(&mut carry, &[x], &[dt], &[0.7], &[0.3])
        .expect("合法一步");
    // 左端为 0，故 b̄ 的实部是 (dt/2)·x，虚部为 0；直连项 D·x 另加。
    let expected = 0.3 * (dt / 2.0 * x * 0.7) + x;
    assert!((y[0] - expected).abs() < 1e-12, "{} vs {expected}", y[0]);
}

/// 未来不影响过去：改掉第 5 个 token 不影响第 2 个 token 的输出。
#[test]
fn later_tokens_do_not_change_earlier_outputs() {
    let m = layer(2, 1);
    let delta = units(WIDTH);
    let (b, c) = (units(2), units(2));
    let inputs = [0.5, -1.5, 2.0, 0.25, 3.0];
    let run = |tokens: &[f64]| {
        let mut carry = m.state();
        tokens
            .iter()
            .map(|&x| {
                m.step(&mut carry, &[x; WIDTH], &delta, &b, &c)
                    .expect("合法一步")
            })
            .collect::<Vec<_>>()
    };
    let first = run(&inputs);
    let mut changed = inputs;
    changed[4] = -7.0;
    let second = run(&changed);
    assert_eq!(first[..4], second[..4]);
    assert_ne!(first[4], second[4]);
}

/// 同一个输入写两次是**叠加**而不是覆盖：这与 `kda` 的 delta 规则正好相反。
#[test]
fn two_writes_accumulate() {
    let m = single(-0.5, 0.4, 0.0);
    let mut carry = m.state();
    m.step(&mut carry, &[1.0], &[0.25], &[1.0], &[0.0])
        .expect("合法一步");
    let once = carry.planes()[0];
    m.step(&mut carry, &[1.0], &[0.25], &[1.0], &[0.0])
        .expect("合法一步");
    let twice = carry.planes()[0];
    assert!(twice > once, "{twice} 应大于 {once}");
}

/// 旋转只改相位：`λ_re = 0` 时第二次（选择量为零的）一步保持模长，`ω = 0` 时状态原样不动。
#[test]
fn rotation_keeps_the_state_magnitude() {
    for (phase, moves) in [(0.4f64, true), (0.0, false)] {
        let m = single(0.0, phase, 0.0);
        let mut carry = m.state();
        m.step(&mut carry, &[1.0], &[0.25], &[1.0], &[0.0])
            .expect("合法一步");
        let before = [carry.planes()[0], carry.planes()[1]];
        // 选择量为零：状态只吃 `ā h`，这正是「纯旋转」的那一步。
        m.step(&mut carry, &[1.0], &[0.25], &[0.0], &[0.0])
            .expect("合法一步");
        let after = [carry.planes()[0], carry.planes()[1]];
        let norm = |s: [f64; 2]| s[0].hypot(s[1]);
        assert!((norm(after) - norm(before)).abs() < 1e-12, "ω={phase}");
        if moves {
            assert!(after != before);
        } else {
            assert_eq!(after, before);
        }
    }
}

/// 虚部不是装饰：同一组参数，`ω = 0` 与 `ω ≠ 0` 读出的结果不同。
///
/// 零状态上 `b̄` 是实数（左端为 0），旋转只在**已有状态**上才显出来，因此先写一步、
/// 再以零选择量读那一步——读到的正是 `Re(ā h)`，相位不同就不同。
#[test]
fn the_imaginary_part_changes_the_output() {
    let mut ys = Vec::new();
    for phase in [0.0f64, 0.9] {
        let m = single(-0.2, phase, 0.0);
        let mut carry = m.state();
        m.step(&mut carry, &[1.0], &[0.5], &[1.0], &[0.0])
            .expect("合法一步");
        let y = m
            .step(&mut carry, &[0.0], &[0.5], &[0.0], &[1.0])
            .expect("合法一步");
        ys.push(y[0]);
    }
    assert!(ys[0] != ys[1], "{ys:?}");
}

/// 低秩 MIMO：第二组选择量为零时，与单秩层逐步给出相同的读数与相同的状态。
#[test]
fn a_silent_rank_does_not_change_anything() {
    let shape = (vec![-0.5; 2], vec![0.4; 2], vec![0.5]);
    let one =
        Mamba::new(1, 2, 1, shape.0.clone(), shape.1.clone(), shape.2.clone()).expect("参数合法");
    let two = Mamba::new(1, 2, 2, shape.0, shape.1, shape.2).expect("参数合法");
    let (mut a, mut b) = (one.state(), two.state());
    let (b1, c1) = ([0.7f64, 0.5], [0.3f64, -0.2]);
    // 第二组选择量整行为零：秩的**行**多一组，但这一组什么都没写、也没读到。
    let (b2, c2) = ([0.7f64, 0.5, 0.0, 0.0], [0.3f64, -0.2, 0.0, 0.0]);
    for x in [0.5f64, -1.0, 2.0] {
        let ya = one.step(&mut a, &[x], &[0.25], &b1, &c1).expect("合法一步");
        let yb = two.step(&mut b, &[x], &[0.25], &b2, &c2).expect("合法一步");
        assert_eq!(ya, yb);
    }
    assert_eq!(a.planes(), b.planes());
}

/// `f32` 上同一套算术跑得动：泛型不是只为 `f64` 测试写的。
#[test]
fn f32_runs_the_same_loop() {
    let m = Mamba::new(2, 2, 2, vec![-1.0f32; 4], vec![0.3; 4], vec![0.5; 2]).expect("参数合法");
    let mut carry = m.state();
    let mut last = Vec::new();
    for _ in 0..8 {
        last = m
            .step(&mut carry, &[1.0, -1.0], &[0.5, 0.5], &[1.0; 4], &[1.0; 4])
            .expect("合法一步");
    }
    assert!(last.iter().all(|y| y.is_finite()));
}
