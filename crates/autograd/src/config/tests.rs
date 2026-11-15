use super::*;

#[test]
fn default_config_uses_depth_and_auto() {
    let config = Config::default();
    assert_eq!(config.ordering(), Ordering::Depth);
    assert_eq!(config.policy(), MemoryPolicy::Auto);
}

#[test]
fn config_carries_both_strategies() {
    let config = Config::new(Ordering::Insertion, MemoryPolicy::Recompute);
    assert_eq!(config.ordering(), Ordering::Insertion);
    assert_eq!(config.policy(), MemoryPolicy::Recompute);
}
