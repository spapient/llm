use crate::session::{Distributed, Remote};
use annotation::parallel;
use diagnostic::Result;
use nn::{Linear, Stack};
use shard::{Assembler, Priority, Reduce};

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

#[test]
fn annotation_policy_drives_layer_assembly_placement() -> Result<()> {
    // 策略从注解取出后，直接驱动模型层的并行归属：PP 段保留、TP 分片几何落到本 rank。
    let policy = Sharded::__parallel_config();
    let stack = Stack::new(vec![
        Linear::new("blk.0.q", 12, 24)?,
        Linear::new("blk.0.o", 24, 12)?,
    ])?;

    let assembler = Assembler::from_config(0, &policy)?;
    let placed = assembler.place_stack(&stack, 1)?;
    assert!(placed.iter().all(|p| p.held()));

    let q = placed
        .iter()
        .find(|p| p.name() == "blk.0.q")
        .expect("Q 列切");
    assert_eq!(q.shard_shape(), Some([12, 12]));
    let o = placed
        .iter()
        .find(|p| p.name() == "blk.0.o")
        .expect("O 行切");
    assert_eq!(o.shard_shape(), Some([12, 12]));
    Ok(())
}

#[test]
fn annotation_policy_drives_the_local_weight_slice() -> Result<()> {
    // 策略 → 归属几何 → 完整权重取本 rank 本地子块；只切片，不装载进图、不发起通信。
    let policy = Sharded::__parallel_config();
    let stack = Stack::new(vec![Linear::new("blk.0.q", 12, 24)?])?;

    let assembler = Assembler::from_config(0, &policy)?;
    let placed = assembler.place_stack(&stack, 1)?;
    let q = placed.iter().find(|p| p.name() == "blk.0.q").expect("Q");

    let full: Vec<f32> = (0..24 * 12).map(|i| i as f32).collect();
    let local = q.split(&full).expect("Q 有分片")?;
    assert_eq!(local.len(), 12 * 12, "TP=2 列切取前 12 行");
    assert_eq!(local[..12], full[..12], "rank 0 子块首行与完整权重首行一致");
    Ok(())
}
