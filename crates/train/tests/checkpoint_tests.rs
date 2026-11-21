    use super::*;
    use diagnostic::Code;
    use tensor::Shape;

    /// 造一个可恢复的最小检查点：`gemv_i4` 单节点图 + 2×2 批 + 2 副本拓扑，返回检查点、头部与拓扑。
    fn fixture() -> (Checkpoint, weights::Header, shard::Topology) {
        let mut tape = autograd::Tape::new();
        let x = tape.leaf();
        tape.record("gemv_i4", &[x]).expect("记录");
        let tokens = [1, 2, 3, 4];
        let batch = crate::Batch::new(&tokens, 2, 2).expect("批");
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
        let batch = crate::Batch::new(&tokens, 1, 1).expect("批");
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
