//! `l3` 模块的单元测试。
use super::*;

/// 一段的简写：`(ρ, ρ̂, ε, L, b)`。
fn rung(rho: f64, rho_hat: f64, eps: f64, lipschitz: f64, offset: f64) -> Rung {
    Rung::new(rho, rho_hat, eps, lipschitz, offset).expect("段")
}

/// 恒等激活的那一段：`L = 1`、`b = 0`，`ρ = ρ̂ = 1`，只有 `ε` 在变。
fn lossy(eps: f64) -> Rung {
    rung(1.0, 1.0, eps, 1.0, 0.0)
}

fn bracket(input: f64, rungs: Vec<Rung>) -> Bracket {
    Chain::new(input, rungs).expect("链").bracket().expect("界")
}

#[test]
fn the_witness_from_the_spec_comes_out_as_two_e_minus_four() {
    // §4.5 给的数值见证：两段、L = 1、ρ = ρ̂ = 1、ε = 1e-4、R = 1 → 2e-4。

    let bound = bracket(1.0, vec![lossy(1e-4), lossy(1e-4)]);

    assert!((bound.error() - 2e-4).abs() < 1e-18);

    let closed = bound.closed_form().expect("闭式适用");

    assert!((closed - bound.error()).abs() < 1e-18);

    assert!((bound.activation() - 1.0).abs() < 1e-18);

    assert!(!bound.is_lossless());

    assert_eq!(bound.ruling(true), Ruling::Refuse);
}

#[test]
fn a_growing_lipschitz_amplifies_every_later_epsilon() {
    // L = 2：e₁ = 2e-4、e₂ = 8e-4、e₃ = 2.4e-3；闭式必须整条不给。

    let steps = (1..=3).map(|_| rung(1.0, 1.0, 1e-4, 2.0, 0.0)).collect();

    let bound = bracket(1.0, steps);

    assert!((bound.error() - 2.4e-3).abs() < 1e-15);

    assert!((bound.activation() - 8.0).abs() < 1e-15);

    assert_eq!(bound.closed_form(), None);
}

#[test]
fn a_bias_alone_withdraws_the_closed_form() {
    // b ≠ 0 时递推里 a_{i-1} 更高，闭式会低估；判定上就是"不给闭式"。

    let bound = bracket(1.0, vec![rung(1.0, 1.0, 1e-4, 1.0, 0.5), lossy(1e-4)]);

    assert_eq!(bound.closed_form(), None);

    // 第二段的 ε 乘的是抬高后的 a₁ = 1.5，因此严格大于无偏置时的 2e-4。

    assert!(bound.error() > 2e-4);

    assert!((bound.activation() - 1.5).abs() < 1e-15);
}

#[test]
fn rho_hat_is_not_substituted_for_rho() {
    // 压缩把范数放大了（ρ̂ = 3 > ρ = 2）：只有左侧乘 ρ̂ 才会反映到后面几段上。

    let steps = vec![lossy(1e-4), rung(2.0, 3.0, 1e-4, 1.0, 0.0), lossy(0.0)];

    let bound = bracket(1.0, steps);

    // e₁ = 1e-4；e₂ = 2·1e-4 + 1e-4·1 = 3e-4；e₃ = 1·3e-4 + 0 = 3e-4。

    assert!((bound.error() - 3e-4).abs() < 1e-15);

    // 闭式走的是 Πρ̂（= 3）而不是 Πρ（= 2），两条路径仍给同一个数。

    assert!((bound.activation() - 3.0).abs() < 1e-15);

    let closed = bound.closed_form().expect("闭式适用");

    assert!((closed - 3e-4).abs() < 1e-15);
}

#[test]
fn a_zero_epsilon_stack_brackets_zero_error() {
    let bound = bracket(7.0, vec![lossy(0.0), lossy(0.0)]);

    assert_eq!(bound.error(), 0.0);

    assert!(bound.is_lossless());

    assert_eq!(bound.closed_form(), Some(0.0));

    assert_eq!(bound.ruling(true), Ruling::Lossless);
}

#[test]
fn a_zero_bound_without_a_round_trip_is_unverified() {
    // §4.5：上界为零不能替代 artifact 往返。

    let bound = bracket(1.0, vec![lossy(0.0)]);

    assert_eq!(bound.ruling(false), Ruling::Unverified);
}

#[test]
fn a_segment_outside_the_theorem_keeps_the_certificate_partial() {
    let chain = Chain::new(1.0, vec![lossy(0.0)])
        .expect("链")
        .with_uncovered(2);

    let bound = chain.bracket().expect("界");

    assert_eq!(bound.uncovered(), 2);

    assert_eq!(bound.ruling(true), Ruling::Partial);

    // 缺往返证据先于覆盖缺口：没有"产物确实还原"就还谈不上覆盖面。

    assert_eq!(bound.ruling(false), Ruling::Unverified);
}

#[test]
fn a_lossy_rung_is_refused_before_the_round_trip_is_consulted() {
    let bound = bracket(1.0, vec![lossy(0.0), lossy(1e-9)]);

    assert_eq!(bound.ruling(true), Ruling::Refuse);

    assert_eq!(bound.ruling(false), Ruling::Refuse);
}

#[test]
fn an_empty_chain_is_the_identity() {
    let bound = bracket(2.5, Vec::new());

    assert_eq!(bound.error(), 0.0);

    assert_eq!(bound.activation(), 2.5);

    assert_eq!(bound.closed_form(), Some(0.0));

    assert_eq!(bound.ruling(true), Ruling::Lossless);
}

#[test]
fn the_closed_form_agrees_with_the_recursion_wherever_it_applies() {
    // 每条链都满足 L = 1、b = 0，于是两个算式必须给同一个数（浮点顺序不同，取 1e-12 相对差）。

    let stacks = [
        vec![rung(0.5, 0.5, 1e-3, 1.0, 0.0)],
        vec![
            rung(4.0, 4.0, 1e-3, 1.0, 0.0),
            rung(0.25, 0.25, 1e-2, 1.0, 0.0),
        ],
        vec![
            rung(1.5, 1.5, 1e-4, 1.0, 0.0),
            rung(1.5, 1.5, 1e-4, 1.0, 0.0),
            rung(1.5, 1.5, 1e-4, 1.0, 0.0),
        ],
    ];

    for steps in stacks {
        let bound = bracket(3.0, steps);

        let closed = bound.closed_form().expect("闭式适用");

        let scale = bound.error().abs().max(1.0);

        assert!((closed - bound.error()).abs() / scale < 1e-12);
    }
}

#[test]
fn every_negative_or_non_finite_fact_is_rejected() {
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(Rung::new(value, 1.0, 0.0, 1.0, 0.0).is_err(), "ρ = {value}");

        assert!(Rung::new(1.0, value, 0.0, 1.0, 0.0).is_err(), "ρ̂ = {value}");

        assert!(Rung::new(1.0, 1.0, value, 1.0, 0.0).is_err(), "ε = {value}");

        assert!(Rung::new(1.0, 1.0, 0.0, value, 0.0).is_err(), "L = {value}");

        assert!(Rung::new(1.0, 1.0, 0.0, 1.0, value).is_err(), "b = {value}");
    }

    assert!(Chain::new(f64::NAN, Vec::new()).is_err());

    assert!(Chain::new(-0.5, Vec::new()).is_err());
}

#[test]
fn an_overflowing_bound_is_not_a_certificate() {
    let huge = rung(f64::MAX, f64::MAX, f64::MAX, 2.0, 0.0);

    let chain = Chain::new(f64::MAX, vec![huge, huge]).expect("链");

    assert!(chain.bracket().is_err());
}

#[test]
fn a_certificate_supplies_the_epsilon_of_its_own_decomposition() {
    // 对角阵：σ = [4, 2, 0.5, 0.25]，最后一点谱隙在 r = 3 之后。

    let mut w = [0.0f32; 16];

    for (idx, value) in [4.0f32, 2.0, 0.5, 0.25].into_iter().enumerate() {
        w[idx * 5] = value;
    }

    let certified = crate::ledger::certify(&w, 4, 4)
        .expect("证书")
        .expect("有谱隙");

    assert_eq!(certified.note.rank, 3);

    let from_note = Rung::from_note(&certified.note, 4.0, 4.0, 1.0, 0.0).expect("段");

    assert_eq!(from_note.eps(), certified.note.eps);

    assert_eq!(from_note.rho(), 4.0);

    // 四处先决条件各自挡住手工构造的证书。

    let failed = Note {
        stable: Stable::Fail,

        ..certified.note.clone()
    };

    assert!(Rung::from_note(&failed, 4.0, 4.0, 1.0, 0.0).is_err());

    let unequal = Note {
        equal: false,

        ..certified.note.clone()
    };

    assert!(Rung::from_note(&unequal, 4.0, 4.0, 1.0, 0.0).is_err());

    // 候选口径：`certify` 从不签这条路，但读端可能拿到别人签的。

    let scratch = Note {
        method: EpsMethod::Candidate,

        ..certified.note.clone()
    };

    assert!(Rung::from_note(&scratch, 4.0, 4.0, 1.0, 0.0).is_err());

    let candidate_bound = Note {
        bound: Bound::Candidate,

        stable: Stable::Open,

        ..certified.note.clone()
    };

    assert!(Rung::from_note(&candidate_bound, 4.0, 4.0, 1.0, 0.0).is_err());
}

#[test]
fn the_ruling_names_are_stable() {
    let names: Vec<&str> = [
        Ruling::Lossless,
        Ruling::Partial,
        Ruling::Unverified,
        Ruling::Refuse,
    ]
    .into_iter()
    .map(Ruling::as_str)
    .collect();

    assert_eq!(names, ["lossless", "partial", "unverified", "refuse"]);
}
