use super::*;

#[test]
fn xavier_uniform_within_bound() {
    let fan_in = 64;
    let fan_out = 128;
    let bound = (6.0 / 192.0f32).sqrt();
    let w = xavier_uniform(fan_in, fan_out, 5).unwrap();
    assert_eq!(w.len(), fan_in * fan_out);
    assert!(w.iter().all(|v| v.abs() <= bound + 1e-6), "越界");
}

#[test]
fn kaiming_uniform_within_bound() {
    let fan_in = 256;
    let w = kaiming_uniform(fan_in, 8, 0.0, 6).unwrap();
    let std = (2.0 / fan_in as f32).sqrt();
    let bound = 3.0f32.sqrt() * std;
    assert!(w.iter().all(|v| v.abs() <= bound + 1e-6));
}

#[test]
fn orthogonal_has_unit_columns() {
    let (rows, cols) = (32, 16);
    let w = orthogonal(rows, cols, 1.0, 7).unwrap();
    assert_eq!(w.len(), rows * cols);
    // 列正交：WᵀW ≈ I。
    for a in 0..cols {
        for b in 0..cols {
            let mut dot = 0.0f32;
            for r in 0..rows {
                dot += w[r * cols + a] * w[r * cols + b];
            }
            let want = if a == b { 1.0 } else { 0.0 };
            assert!((dot - want).abs() < 1e-3, "({a},{b}) = {dot}");
        }
    }
}

#[test]
fn orthogonal_has_unit_rows_when_fat() {
    let (rows, cols) = (8, 20);
    let w = orthogonal(rows, cols, 1.0, 8).unwrap();
    for a in 0..rows {
        for b in 0..rows {
            let mut dot = 0.0f32;
            for c in 0..cols {
                dot += w[a * cols + c] * w[b * cols + c];
            }
            let want = if a == b { 1.0 } else { 0.0 };
            assert!((dot - want).abs() < 1e-3, "({a},{b}) = {dot}");
        }
    }
}

#[test]
fn same_seed_reproduces_weights() {
    let a = kaiming_normal(512, 64, 0.01, 99).unwrap();
    let b = kaiming_normal(512, 64, 0.01, 99).unwrap();
    assert_eq!(a, b);
}

#[test]
fn zero_dims_are_rejected() {
    assert!(xavier_uniform(0, 8, 1).is_err());
    assert!(kaiming_uniform(0, 8, 0.0, 1).is_err());
    assert!(orthogonal(0, 4, 1.0, 1).is_err());
}
