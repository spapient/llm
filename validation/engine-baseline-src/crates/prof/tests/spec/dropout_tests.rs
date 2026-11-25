use super::*;

#[test]
fn eval_is_identity() {
    let mut x = vec![1.0f32, -2.0, 3.5];
    let want = x.clone();
    dropout(&mut x, 0.7, 9, false).unwrap();
    assert_eq!(x, want);
}

#[test]
fn zero_probability_is_identity() {
    let mut x = vec![1.0f32, -2.0, 3.5];
    dropout(&mut x, 0.0, 9, true).unwrap();
    assert_eq!(x, vec![1.0, -2.0, 3.5]);
}

#[test]
fn same_seed_gives_same_mask() {
    let base: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let mut a = base.clone();
    let mut b = base;
    dropout(&mut a, 0.3, 42, true).unwrap();
    dropout(&mut b, 0.3, 42, true).unwrap();
    assert_eq!(a, b);
}

#[test]
fn drop_rate_is_statistically_honest() {
    let mut x = vec![1.0f32; 10_000];
    dropout(&mut x, 0.25, 7, true).unwrap();
    let zeros = x.iter().filter(|v| **v == 0.0).count();
    let rate = zeros as f32 / x.len() as f32;
    assert!((rate - 0.25).abs() < 0.02, "实测丢弃率 {rate}");
    // 保留下来的元素必须缩放为 1/(1-p)。
    let kept = x.iter().find(|v| **v != 0.0).copied().unwrap();
    assert!((kept - 4.0 / 3.0).abs() < 1e-5, "保留值 {kept}");
}

#[test]
fn rejects_out_of_range_probability() {
    let mut x = vec![1.0f32];
    assert!(dropout(&mut x, 1.0, 1, true).is_err());
    assert!(dropout(&mut x, -0.1, 1, true).is_err());
}
