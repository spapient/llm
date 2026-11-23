use crate::Topology;
use crate::policy::{ParallelConfig, Priority};
use diagnostic::Code;

/// 优先级短名、序数与判别式一一对应；序数越小越先调度。
#[test]
fn priority_names_and_ordinals_are_consistent() {
    let expected = [("high", 0u8), ("normal", 1u8), ("low", 2u8)];
    for (priority, (name, ordinal)) in Priority::ALL.iter().zip(expected) {
        assert_eq!(priority.as_str(), name, "短名不符");
        assert_eq!(priority.ordinal(), ordinal, "调度序数不符");
        assert_eq!(Priority::from_name(name), Some(*priority), "短名反解不符");
    }
    assert_eq!(
        Priority::from_name("nope"),
        None,
        "未知优先级不应当成合法值"
    );
    assert!(Priority::High < Priority::Normal && Priority::Normal < Priority::Low);
}

/// 单进程策略：1×1×1、求和、平均、普通优先级，恒合法。
#[test]
fn single_process_config_is_the_identity_policy() {
    let config = ParallelConfig::single();
    assert_eq!((config.tp(), config.pp(), config.dp()), (1, 1, 1));
    assert!(config.average());
    assert_eq!(config.priority(), Priority::Normal);
    let topology = config.topology().expect("单进程拓扑合法");
    assert_eq!(topology.world(), 1);
}

/// 三维度数进拓扑；派生世界数为三者乘积。
#[test]
fn config_derives_the_matching_topology() {
    let config = ParallelConfig::new(2, 2, 2).expect("2·2·2 合法");
    let topology = config.topology().expect("派生拓扑");
    assert_eq!(topology.world(), 8);
    assert_eq!((config.tp(), config.pp(), config.dp()), (2, 2, 2));
}

/// 零维度数在构造期拒绝，与 [`Topology::new`] 同一判据。
#[test]
fn zero_degree_is_rejected_at_construction() {
    assert!(ParallelConfig::new(0, 2, 2).is_err());
    assert!(ParallelConfig::new(2, 0, 2).is_err());
    assert!(ParallelConfig::new(2, 2, 0).is_err());
}

/// 旋钮链式覆盖：归约、平均、优先级各自独立可改。
#[test]
fn knobs_are_overridden_independently() {
    let config = ParallelConfig::new(1, 1, 2)
        .expect("dp 2 合法")
        .with_reduce(crate::Reduce::Max)
        .with_average(false)
        .with_priority(Priority::High);
    assert_eq!(config.reduce(), crate::Reduce::Max);
    assert!(!config.average());
    assert_eq!(config.priority(), Priority::High);
}

/// 非法度数上抛 [`Code::Invalid`]，不把拓扑错误留到装配期。
#[test]
fn rejected_config_carries_invalid_code() {
    let error = ParallelConfig::new(0, 1, 1).expect_err("零维应拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

/// 策略度数与直接构造的拓扑一致：配置与拓扑不是两套可漂移的事实。
#[test]
fn config_topology_matches_direct_construction() {
    let config = ParallelConfig::new(2, 1, 3).expect("tp2 dp3 合法");
    let direct = Topology::new(2, 1, 3).expect("直接构造合法");
    assert_eq!(config.topology().expect("派生"), direct);
}
