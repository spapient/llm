//! 可恢复检查点：只记"要复现这次训练必须知道的东西"。
//!
//! [`Checkpoint`] 是内存里的最小事实集；字节格式归 [`crate::snapshot`]，编码、版本与偏移校验
//! 不在训练循环里再做一遍。[`resume`] 在恢复前拦住并行度不一致这类最贵的静默错位。

use diagnostic::{Code, Error, Result};

/// 一条训练步记录。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step {
    index: u64,
    loss: f32,
}

impl Step {
    /// 由步号与损失构造。
    pub fn new(index: u64, loss: f32) -> Self {
        Self { index, loss }
    }

    /// 步号（从 0 开始）。
    pub fn index(&self) -> u64 {
        self.index
    }

    /// 该步的平均损失。
    pub fn loss(&self) -> f32 {
        self.loss
    }
}

/// 可恢复检查点：只记"要复现这次训练必须知道的东西"。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Checkpoint {
    step: u64,
    tape_nodes: usize,
    batch_rows: usize,
    batch_cols: usize,
    layers: usize,
    weight_entries: usize,
    world: usize,
}

impl Checkpoint {
    /// 由当前状态构造；`layers` 为 0 说明没有可训练的层，直接报错。
    pub fn new(
        step: u64,
        tape: &autograd::Tape,
        batch: &data::Batch<'_>,
        layers: usize,
        weights: &weights::Header,
        topology: shard::Topology,
    ) -> Result<Self> {
        if layers == 0 {
            return Err(Error::new(Code::Invalid, "没有可训练的层").with("train::Checkpoint::new"));
        }
        Ok(Self {
            step,
            tape_nodes: tape.len(),
            batch_rows: batch.rows(),
            batch_cols: batch.cols(),
            layers,
            weight_entries: weights.entries().len(),
            world: topology.world(),
        })
    }

    /// 步号。
    pub fn step(&self) -> u64 {
        self.step
    }

    /// 反向图节点数的内存诊断量；不进快照、恢复时不据此对账。图结构一致性由权重条目表
    /// 与层算子契约守卫，而不是靠节点数。
    pub fn tape_nodes(&self) -> usize {
        self.tape_nodes
    }

    /// 批的行数。
    pub fn batch_rows(&self) -> usize {
        self.batch_rows
    }

    /// 批的列宽。
    pub fn batch_cols(&self) -> usize {
        self.batch_cols
    }

    /// 训练层数。
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// 权重条目数。
    pub fn weight_entries(&self) -> usize {
        self.weight_entries
    }

    /// 恢复时要求的进程数。
    pub fn world(&self) -> usize {
        self.world
    }

    /// 把这些事实与当前权重表组合成可落盘的快照。
    ///
    /// 文件格式由 [`crate::snapshot`] 拥有：本层只回答"恢复需要知道什么"，编码、版本与偏移校验
    /// 不在训练循环里再做一遍。
    pub fn snapshot(&self, weights: &weights::Header) -> Result<crate::snapshot::Snapshot> {
        let facts = crate::snapshot::Facts::new(
            self.step,
            self.world,
            self.layers,
            self.batch_rows,
            self.batch_cols,
        )?;
        Ok(crate::snapshot::Snapshot::new(facts, weights.clone()))
    }
}

/// 一个检查点能否在同一结构上恢复。
///
/// 并行度、层数与批行数不一致，强行恢复只会读出形状正确、内容错位的权重——这是最贵的一类
/// 静默错误，所以在这里逐项拦住。`layers` / `batch_rows` 是当前运行的对应事实，用于和快照
/// 里记录的事实对账。
pub fn resume(
    checkpoint: &Checkpoint,
    topology: &shard::Topology,
    layers: usize,
    batch_rows: usize,
    batch_cols: usize,
) -> Result<()> {
    topology.check(checkpoint.world()).map_err(|error| {
        error.with(format!(
            "检查点在第 {} 步、{} 进程下保存",
            checkpoint.step(),
            checkpoint.world()
        ))
    })?;
    let context = format!("检查点在第 {} 步保存", checkpoint.step());
    reconcile(&context, "训练层数", "层", checkpoint.layers(), layers)?;
    reconcile(
        &context,
        "批行数",
        "行",
        checkpoint.batch_rows(),
        batch_rows,
    )?;
    reconcile(
        &context,
        "批列宽",
        "列",
        checkpoint.batch_cols(),
        batch_cols,
    )?;
    Ok(())
}

/// 一项恢复事实必须与当前运行逐项相同；口径与 [`resume`] 一致，三条对账共用同一份判据。
fn reconcile(context: &str, label: &str, unit: &str, saved: usize, current: usize) -> Result<()> {
    if saved == current {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!("{label}不一致：检查点记录 {saved} {unit}，当前是 {current} {unit}"),
    )
    .with(context))
}

/// 训练层的算子引用：`nn` 的层定义必须先通过 `op` 的闭集校验。
///
/// 这里只提供"取全部算子名"的只读便利，避免训练循环自己拼字符串。
pub fn operators(layer: &dyn nn::Contract) -> &[String] {
    layer.spca().ops()
}

#[cfg(test)]
mod tests {
    use super::*;
    use diagnostic::Code;
    use tensor::Shape;

    /// 造一个可恢复的最小检查点：`gemv_i4` 单节点图 + 2×2 批 + 2 副本拓扑，返回检查点、头部与拓扑。
    fn fixture() -> (Checkpoint, weights::Header, shard::Topology) {
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
        (checkpoint, header, topology)
    }

    #[test]
    fn a_checkpoint_records_the_minimum_state_needed_to_resume() {
        let (checkpoint, _header, topology) = fixture();
        assert_eq!(checkpoint.step(), 7);
        assert_eq!(checkpoint.tape_nodes(), 1);
        assert_eq!(checkpoint.batch_rows(), 2);
        assert_eq!(checkpoint.batch_cols(), 2);
        assert_eq!(checkpoint.weight_entries(), 1);
        assert_eq!(checkpoint.world(), 2);
        resume(&checkpoint, &topology, 2, 2, 2).expect("同结构可恢复");
        let other = shard::Topology::new(1, 1, 4).expect("拓扑");
        assert_eq!(
            resume(&checkpoint, &other, 2, 2, 2).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(
            resume(&checkpoint, &topology, 3, 2, 2).unwrap_err().code(),
            Code::Invalid,
            "层数不一致必须拒绝"
        );
        assert_eq!(
            resume(&checkpoint, &topology, 2, 4, 2).unwrap_err().code(),
            Code::Invalid,
            "批行数不一致必须拒绝"
        );
        assert_eq!(
            resume(&checkpoint, &topology, 2, 2, 4).unwrap_err().code(),
            Code::Invalid,
            "批列宽不一致必须拒绝"
        );
    }

    #[test]
    fn a_checkpoint_becomes_a_snapshot_that_round_trips() {
        let (checkpoint, header, _topology) = fixture();
        let snapshot = checkpoint.snapshot(&header).expect("快照");
        let bytes = snapshot.encode().expect("编码");
        let restored = crate::snapshot::Snapshot::decode(&bytes).expect("解码");
        let facts = restored.facts();
        assert_eq!(facts.step(), checkpoint.step());
        assert_eq!(facts.world(), checkpoint.world());
        assert_eq!(facts.layers(), checkpoint.layers());
        assert_eq!(facts.batch_rows(), checkpoint.batch_rows());
        assert_eq!(facts.batch_cols(), checkpoint.batch_cols());
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
}
