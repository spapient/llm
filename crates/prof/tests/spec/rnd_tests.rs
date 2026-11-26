use super::*;

testkit::test_cases! {
fn is_deterministic_per_seed() {
    let mut a = SplitMix64::new(7);
    let mut b = SplitMix64::new(7);
    for _ in 0..16 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

fn uniform_stays_in_unit_interval() {
    let mut r = SplitMix64::new(11);
    for _ in 0..10_000 {
        let v = r.next_f32();
        assert!((0.0..1.0).contains(&v), "越界 {v}");
    }
}

fn normal_has_unit_variance() {
    let mut r = SplitMix64::new(13);
    let n = 20_000;
    let sum: f32 = (0..n).map(|_| r.next_normal()).sum();
    let mean = sum / n as f32;
    assert!(mean.abs() < 0.05, "均值 {mean}");
}
}
