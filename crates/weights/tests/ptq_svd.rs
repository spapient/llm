//! `ptq::svd` 的单元测试：Jacobi SVD 的分解与重建。
use super::*;

fn frob_resid(a: &[f64], back: &[f64]) -> f64 {
    a.iter()
        .zip(back)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn diagonal_singular_values_are_exact() {
    // diag(3,2,1,0.5) 行主序。
    let mut a = vec![0.0f64; 16];
    let diag = [3.0, 2.0, 1.0, 0.5];
    for (i, d) in diag.iter().enumerate() {
        a[i * 4 + i] = *d;
    }
    let svd = compute(&a, 4, 4).expect("svd");
    for (got, want) in svd.sigma.iter().zip(diag.iter()) {
        assert!((got - want).abs() < 1e-10, "got {got} want {want}");
    }
}

#[test]
fn full_reconstruction_residual_is_machine_order() {
    let mut rng = 0x1234_5678u64;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 40) as f64 / (1u64 << 24) as f64 - 0.5
    };
    for (m, n) in [(8, 8), (12, 6), (6, 12), (20, 10)] {
        let a: Vec<f64> = (0..m * n).map(|_| next()).collect();
        let svd = compute(&a, m, n).expect("svd");
        let k = m.min(n);
        assert_eq!(svd.sigma.len(), k);
        let back = svd.reconstruct(k);
        let scale = svd.sigma[0];
        assert!(
            frob_resid(&a, &back) < 1e-10 * scale * (m as f64).sqrt(),
            "残差应在机器精度量级 m={m} n={n}"
        );
    }
}

#[test]
fn truncated_reconstruction_is_monotonic() {
    let mut a = vec![0.0f64; 36];
    for i in 0..6 {
        a[i * 6 + i] = 6.0 - i as f64;
    }
    let svd = compute(&a, 6, 6).expect("svd");
    let mut prev = f64::INFINITY;
    for r in 1..=6 {
        let err = frob_resid(&a, &svd.reconstruct(r));
        assert!(err <= prev, "增加秩不应让误差变大");
        prev = err;
    }
}

#[test]
fn singular_values_are_descending_and_nonnegative() {
    let mut rng = 99u64;
    let a: Vec<f64> = (0..64)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 40) as f64 / (1u64 << 24) as f64
        })
        .collect();
    let svd = compute(&a, 8, 8).expect("svd");
    for pair in svd.sigma.windows(2) {
        assert!(pair[0] >= pair[1]);
        assert!(pair[0] >= 0.0);
    }
}

#[test]
fn bad_shape_is_an_error() {
    assert!(compute(&[1.0], 2, 2).is_err());
    assert!(compute(&[], 0, 4).is_err());
}

/// 信号秩选择：结构化的秩边界有相对谱隙 → 秩 10；满秩噪声平坦谱 → 拒压；单一奇异值无从比较 → 拒压。
#[test]
fn signal_rank_separates_structure_from_noise() {
    let mut structured = vec![10.0f64; 10];
    structured.extend(vec![0.1f64; 6]);
    assert_eq!(signal_rank(&structured), Some(10));

    let noise: Vec<f64> = (0..16).map(|k| 10.0 - k as f64 * 0.1).collect();
    assert_eq!(signal_rank(&noise), None);

    assert_eq!(signal_rank(&[1.0]), None);
}
