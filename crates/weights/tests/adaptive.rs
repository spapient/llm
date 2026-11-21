//! `adaptive` 模块的单元测试。

use super::*;



/// 档位表与位宽表必须同长：两处错位会让 `widths` 报出另一档的位宽。

#[test]

fn ladder_and_bits_stay_in_step() {

    assert_eq!(LADDER.len(), BITS.len());

    for (level, &scheme) in LADDER.iter().enumerate() {

        assert_eq!(scheme.bits, BITS[level]);

    }

}



/// 分组区间：末组按剩余元素收窄，不越界也不丢元素。

#[test]

fn spans_cover_the_input_exactly() {

    assert_eq!(span(4, 10, 0), 0..4);

    assert_eq!(span(4, 10, 2), 8..10);

    assert_eq!(span(4, 8, 1), 4..8);

}



/// 极紧预算下所有组停在最宽档：`I8` 也有均匀量化误差，零预算因此一律未达标。

///

/// 「停在 8bit」与「达标」是两件事：预估误差是 `n·scale²/12`，`scale` 只在 `max_abs = 0` 时

/// 才为零，所以零预算下没有一档能过——把它读成「8bit 无损」会掩盖这一步其实没达标。

#[test]

fn tight_budget_keeps_every_group_wide() {

    let x = [0.5f32, -0.25, 1.0, -2.0];

    let adaptive = adaptive_quantize(2, &x, 0.0).expect("合法输入");

    assert_eq!(adaptive.widths, vec![8, 8]);

    assert!(!adaptive.met);

    assert_eq!(adaptive.bpw, 8.0);

    // 预算为零 ⇒ 全部停在 `I8`，逐位等于父模块的固定 `I8` 往返（不是「近似无损」）。

    let (packed, scales) = quantize(I8, 2, &x).expect("合法输入");

    let mut expect = vec![0.0f32; x.len()];

    dequantize(I8, &packed, &scales, 2, &mut expect).expect("合法输入");

    assert_eq!(adaptive.recon, expect);

}



/// 极宽预算下所有组降到最窄档：贪心没有「只降一档就停」的口子。

#[test]

fn loose_budget_reaches_the_narrowest_rung() {

    let x = [0.5f32, -0.25, 1.0, -2.0];

    let adaptive = adaptive_quantize(2, &x, 1e9).expect("合法输入");

    assert_eq!(adaptive.widths, vec![2, 2]);

    assert!(adaptive.met);

    assert_eq!(adaptive.bpw, 2.0);

}



/// 预算不可能达标时 `met = false`，位宽图仍给出（调用方据此决定拒压，而不是拿到空结果）。

#[test]

fn unmet_budget_still_reports_the_map() {

    let x = [0.5f32, -0.25, 1.0, -2.0];

    let adaptive = adaptive_quantize(2, &x, 0.0).expect("合法输入");

    let tight = adaptive_quantize(2, &x, 1e-12).expect("合法输入");

    assert!(!tight.met);

    assert_eq!(tight.widths, adaptive.widths);

}



/// 预算越宽，选出的位宽只会更低：单调性是贪心可用的前提。

#[test]

fn wider_budgets_never_pick_wider_widths() {

    let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();

    let mut previous = u8::MAX;

    for budget in [0.0f64, 0.5, 2.0, 8.0, 32.0] {

        let adaptive = adaptive_quantize(8, &x, budget).expect("合法输入");

        let widest = adaptive.widths.iter().copied().max().expect("非空");

        assert!(widest <= previous, "预算 {budget} 反而选了更宽的档");

        previous = widest;

    }

}



/// 三项入口契约各自独立失败：分组为零、非有限输入、负预算。

#[test]

fn invalid_inputs_are_rejected_before_any_allocation() {

    assert!(adaptive_quantize(0, &[1.0f32], 1.0).is_err());

    assert!(adaptive_quantize(2, &[1.0f32, f32::NAN], 1.0).is_err());

    assert!(adaptive_quantize(2, &[1.0f32, 2.0], -0.5).is_err());

    assert!(LADDER.iter().all(|s| s.max_code > 0.0));

}



/// 补偿形态在无谱隙时拒压，而不是把满秩矩阵当成低秩。

#[test]

fn compensation_refuses_without_a_spectral_gap() {

    let w = [1.0f32, 0.0, 0.0, 1.0];

    assert!(

        compensate_adaptive(2, &w, 2, 2, 20.0)

            .expect("合法形状")

            .is_none()

    );

}



/// 目标 SNR 非有限在入口被拒：`+inf` 会静默退化成零误差预算。

#[test]

fn non_finite_target_snr_is_rejected() {

    let w = [1.0f32, 0.0, 0.0, 1.0];

    assert!(compensate_adaptive(2, &w, 2, 2, f64::INFINITY).is_err());

    assert!(compensate_adaptive(2, &w, 2, 2, f64::NAN).is_err());

}



/// 确定性合成矩阵：`r` 条 rank-1 项加正弦噪声（固定公式，无随机源）。

///

/// E5 明确「不得使用合成矩阵代替模型级结果」，所以这条只作**机制探针**：它量的是「位宽图与

/// 平均码位是否随预算单调、重构 SNR 是否达到同一预算」，不出 E5 的判决。

fn synthetic(m: usize, n: usize, rank: usize, noise: f32) -> Vec<f32> {

    let mut w = vec![0.0f32; m * n];

    for row in 0..m {

        for col in 0..n {

            let mut acc = 0.0f32;

            for k in 0..rank {

                let a = ((row * (k + 1)) as f32 * 0.017).sin();

                let b = ((col * (k + 1)) as f32 * 0.013).cos();

                acc += a * b;

            }

            let jitter = ((row * 31 + col * 17) as f32 * 0.001).sin() * noise;

            w[row * n + col] = acc + jitter;

        }

    }

    w

}



/// 实测探针：一层合成权重上的秩、位宽图、平均码位与**权重空间** SNR。

///

/// 命令：`cargo test -p llm-weights --release --lib -- --ignored e5_width_map --nocapture`。

///

/// 报的 SNR 是权重空间口径——E5 禁止拿它替代输出空间 SNR，这里只用来确认「预算越紧、位宽

/// 越宽、SNR 越高」这条单调关系在实现里真的成立。

#[test]

#[ignore = "实测探针：打印一层合成权重的位宽图与权重空间 SNR，不参与默认门禁"]

fn e5_width_map() {

    let (m, n, group) = (512usize, 512usize, 32usize);

    let w = synthetic(m, n, 8, 0.01);

    let energy: f64 = w.iter().map(|&v| f64::from(v) * f64::from(v)).sum();

    println!("E5 探针：{m}×{n}（rank-8 + 噪声），group={group}，‖W‖²_F={energy:.1}");

    for target in [30.0f64, 40.0, 50.0, 60.0] {

        let Some(card) = compensate_adaptive(group, &w, m, n, target).expect("合法输入")

        else {

            println!("  target {target} dB：无谱隙，拒压");

            continue;

        };

        let residual: f64 = card

            .recon

            .iter()

            .zip(&w)

            .map(|(&a, &b)| {

                let d = f64::from(a) - f64::from(b);

                d * d

            })

            .sum();

        let snr = 10.0 * (energy / residual.max(f64::MIN_POSITIVE)).log10();

        let mut histogram = [0usize; 4];

        for width in &card.widths {

            let slot = match width {

                2 => 0,

                3 => 1,

                4 => 2,

                _ => 3,

            };

            histogram[slot] += 1;

        }

        println!(

            "  target {target} dB：rank={} gap={:.3e} sigma1={:.1} bpw={:.3} met={} \

             位宽图 2/3/4/8 = {}/{}/{}/{}，权重空间 SNR={snr:.2} dB",

            card.rank,

            card.gap,

            card.sigma1,

            card.bpw,

            card.met,

            histogram[0],

            histogram[1],

            histogram[2],

            histogram[3]

        );

        assert!(

            card.met,

            "目标 {target} dB 未达标：位宽图仍给出，但预算没满足"

        );

    }

}
