use super::*;
use tensor::Shape;

#[test]
fn the_schedule_warms_up_then_decays_to_zero() {
    let schedule = Schedule::new(10, 100).expect("合法");
    assert_eq!(schedule.lr(1.0, 0), 0.1);
    assert_eq!(schedule.lr(1.0, 9), 1.0);
    assert!(
        (schedule.lr(1.0, 55) - 0.5).abs() < 1e-6,
        "warmup 后线性衰减"
    );
    assert_eq!(schedule.lr(1.0, 100), 0.0);
    assert_eq!(schedule.lr(1.0, 1_000), 0.0);
}

#[test]
fn a_schedule_without_warmup_starts_at_the_peak() {
    let schedule = Schedule::new(0, 10).expect("合法");
    assert_eq!(schedule.lr(0.5, 0), 0.5);
    assert!(
        (schedule.lr(0.5, 9) - 0.05).abs() < 1e-6,
        "无 warmup 时直接线性衰减"
    );
}

#[test]
fn an_impossible_schedule_is_rejected() {
    assert_eq!(Schedule::new(0, 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Schedule::new(11, 10).unwrap_err().code(), Code::Invalid);
    assert_eq!(
        Schedule::with(0, 10, Anneal::Cosine, 1.5)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
}

#[test]
fn cosine_decays_from_the_peak_to_the_floor_ratio() {
    let schedule = Schedule::with(0, 100, Anneal::Cosine, 0.1).expect("合法");
    assert_eq!(schedule.lr(1.0, 0), 1.0, "进度起点在峰值");
    let floor = schedule.lr(1.0, 99);
    assert!((floor - 0.1).abs() < 1e-3, "末段落到下限比例：{floor}");
    assert_eq!(schedule.lr(1.0, 100), 0.0, "超出总步数为 0");
}

#[test]
fn a_constant_schedule_holds_the_peak_after_warmup() {
    let schedule = Schedule::with(10, 100, Anneal::Constant, 0.1).expect("合法");
    assert_eq!(schedule.lr(1.0, 9), 1.0, "warmup 结束到峰值");
    assert_eq!(schedule.lr(1.0, 50), 1.0, "恒定曲线不向下限压");
}

#[test]
fn linear_decay_respects_the_floor_ratio() {
    let schedule = Schedule::with(0, 100, Anneal::Linear, 0.2).expect("合法");
    let floor = schedule.lr(1.0, 99);
    assert!(floor >= 0.2 - 1e-6, "线性谷底不低于下限：{floor}");
    assert!(floor <= 1.0, "学习率不超过峰值：{floor}");
}

#[test]
fn a_checkpoint_records_the_minimum_state_needed_to_resume() {
    let mut tape = autograd::Tape::new();
    let x = tape.leaf();
    tape.record("gemv_i4", &[x]).expect("记录");
    let tokens = [1, 2, 3, 4];
    let batch = data::Batch::new(&tokens, 2, 2).expect("批");
    let header = weights::Header::new(
        vec![weights::Entry::new("w", tensor::DType::F32, Shape::new([2])).at(64)],
        64,
    )
    .expect("头部");
    let topology = shard::Topology::new(1, 1, 2).expect("拓扑");
    let checkpoint = Checkpoint::new(7, &tape, &batch, 2, &header, topology).expect("检查点");
    assert_eq!(checkpoint.step(), 7);
    assert_eq!(checkpoint.tape_nodes(), 1);
    assert_eq!(checkpoint.batch_rows(), 2);
    assert_eq!(checkpoint.weight_entries(), 1);
    assert_eq!(checkpoint.world(), 2);
    resume(&checkpoint, topology).expect("同拓扑可恢复");
    let other = shard::Topology::new(1, 1, 4).expect("拓扑");
    assert_eq!(
        resume(&checkpoint, other).unwrap_err().code(),
        Code::Invalid
    );
}

#[test]
fn a_checkpoint_becomes_a_snapshot_that_round_trips() {
    let mut tape = autograd::Tape::new();
    let x = tape.leaf();
    tape.record("gemv_i4", &[x]).expect("记录");
    let tokens = [1, 2, 3, 4];
    let batch = data::Batch::new(&tokens, 2, 2).expect("批");
    let header = weights::Header::new(
        vec![weights::Entry::new("w", tensor::DType::F32, Shape::new([2])).at(64)],
        64,
    )
    .expect("头部");
    let topology = shard::Topology::new(1, 1, 2).expect("拓扑");
    let checkpoint = Checkpoint::new(7, &tape, &batch, 2, &header, topology).expect("检查点");

    let snapshot = checkpoint.snapshot(&header).expect("快照");
    let bytes = snapshot.encode().expect("编码");
    let restored = snapshot::Snapshot::decode(&bytes).expect("解码");
    let facts = restored.facts();
    assert_eq!(facts.step(), checkpoint.step());
    assert_eq!(facts.world(), checkpoint.world());
    assert_eq!(facts.layers(), checkpoint.layers());
    assert_eq!(facts.batch_rows(), checkpoint.batch_rows());
    assert_eq!(
        restored.weights().entries().len(),
        checkpoint.weight_entries()
    );
}

#[test]
fn a_checkpoint_without_layers_is_rejected() {
    let tape = autograd::Tape::new();
    let tokens = [1];
    let batch = data::Batch::new(&tokens, 1, 1).expect("批");
    let header = weights::Header::new(Vec::new(), 0).expect("空头部");
    let topology = shard::Topology::new(1, 1, 1).expect("拓扑");
    assert_eq!(
        Checkpoint::new(0, &tape, &batch, 0, &header, topology)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
}

#[test]
fn a_layer_exposes_the_operators_it_was_validated_against() {
    let layer = nn::Linear::new("blk.0.attn_q", 4, 4).expect("层");
    assert_eq!(operators(&layer), ["gemv_i4"]);
}
