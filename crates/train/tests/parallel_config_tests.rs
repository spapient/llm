use crate::session::{Distributed, Remote};
use annotation::parallel;
use diagnostic::Result;
use shard::{Priority, Reduce};

/// 带完整策略标注的装配描述：TP=2、DP=2，最大值归约、不取平均、高优先级。
#[parallel(
    tp = 2,
    pp = 1,
    dp = 2,
    reduce = "max",
    average = false,
    priority = "high"
)]
struct Sharded;

/// 裸注解：回落单进程默认口径。
#[parallel]
struct Single;

#[test]
fn annotation_policy_flows_into_distributed_from_config() -> Result<()> {
    // 注解产物只生成策略访问器；这里从访问器取出配置，整条链不复制键表。
    let policy = Sharded::__parallel_config();
    assert_eq!(policy.tp(), 2);
    assert_eq!(policy.pp(), 1);
    assert_eq!(policy.dp(), 2);
    assert_eq!(policy.reduce(), Reduce::Max);
    assert!(!policy.average());
    assert_eq!(policy.priority(), Priority::High);

    let distributed = Distributed::from_config(0, Remote, policy)?;
    assert_eq!(distributed.topology().world(), 4);
    assert_eq!(distributed.policy().priority(), Priority::High);
    Ok(())
}

#[test]
fn bare_annotation_falls_back_to_single_process_policy() {
    let policy = Single::__parallel_config();
    assert_eq!((policy.tp(), policy.pp(), policy.dp()), (1, 1, 1));
    assert_eq!(policy.reduce(), Reduce::Sum);
    assert!(policy.average());
    assert_eq!(policy.priority(), Priority::Normal);
}
