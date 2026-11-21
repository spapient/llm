use super::*;

#[test]
fn uniform_respects_endpoints_and_is_reproducible() {
    let distribution = Uniform::new(-3.0, 5.0).expect("区间合法");
    let mut left = Rng::new(2024);
    let mut right = Rng::new(2024);
    for _ in 0..16 {
        let value = distribution.sample(&mut left);
        assert!((-3.0..5.0).contains(&value), "落在 [lo,hi)");
        assert_eq!(distribution.sample(&mut right), value, "同种子可复现");
    }
}

#[test]
fn uniform_rejects_empty_or_non_finite_range() {
    assert!(Uniform::new(1.0, 1.0).is_err());
    assert!(Uniform::new(1.0, f32::NAN).is_err());
    assert!(Uniform::new(f32::INFINITY, 2.0).is_err());
    assert!(Uniform::new(2.0, 1.0).is_err());
}

#[test]
fn bernoulli_probability_is_bounded() {
    assert!(Bernoulli::new(0.0).is_ok());
    assert!(Bernoulli::new(1.0).is_ok());
    assert!(Bernoulli::new(-0.1).is_err());
    assert!(Bernoulli::new(1.2).is_err());
    assert!(Bernoulli::new(f32::NAN).is_err());

    let always = Bernoulli::new(1.0).expect("p=1 合法");
    let never = Bernoulli::new(0.0).expect("p=0 合法");
    let mut rng = Rng::new(1);
    assert!(always.sample(&mut rng));
    assert!(!never.sample(&mut rng));
}

#[test]
fn normal_samples_are_finite_and_reproducible() {
    let distribution = Normal::new(1.0, 0.25).expect("参数合法");
    let mut left = Rng::new(11);
    let mut right = Rng::new(11);
    for _ in 0..16 {
        let value = distribution.sample(&mut left);
        assert!(value.is_finite(), "正态样本必须有限");
        assert_eq!(distribution.sample(&mut right), value, "同种子可复现");
    }
}

#[test]
fn normal_rejects_non_positive_or_non_finite_std() {
    assert!(Normal::new(0.0, 0.0).is_err());
    assert!(Normal::new(0.0, -1.0).is_err());
    assert!(Normal::new(f32::NAN, 1.0).is_err());
    assert!(Normal::new(0.0, f32::INFINITY).is_err());
}
