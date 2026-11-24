use super::*;
use nn::{Embedding, Linear, Stack};

use crate::{Plan, Sharding, Topology};

/// 两层模型：块 0 的 Q/O、块 1 的 Q，外加一块无块号的嵌入参数。
fn sample() -> Stack {
    Stack::of(vec![
        Box::new(Linear::new("blk.0.q", 12, 24).expect("in=12, out=24")),
        Box::new(Linear::new("blk.0.o", 24, 12).expect("in=24, out=12")),
        Box::new(Linear::new("blk.1.q", 12, 24).expect("in=12, out=24")),
        Box::new(Embedding::new("tok_embeddings", 48, 12, 1).expect("嵌入表")),
    ])
    .expect("层名唯一")
}

#[test]
fn single_process_holds_every_layer_with_full_shape() {
    let assembler = Assembler::new(Plan::single());
    let placed = assembler.place_stack(&sample(), 2).expect("单进程合法");

    assert!(placed.iter().all(super::Placement::held));
    let q = placed
        .iter()
        .find(|p| p.name() == "blk.0.q")
        .expect("Q 投影");
    assert_eq!(q.shard_shape(), Some([24, 12]));
    assert_eq!(q.shard_range(), Some((0, 24)));
}

#[test]
fn pp_splits_blocks_across_pipeline_ranks() {
    let topology = Topology::new(1, 2, 1).expect("拓扑合法");
    let rank0 = Assembler::new(Plan::new(topology, 0).expect("rank 0"));
    let rank1 = Assembler::new(Plan::new(topology, 1).expect("rank 1"));

    let a = rank0.place_stack(&sample(), 2).expect("rank 0 归属");
    let b = rank1.place_stack(&sample(), 2).expect("rank 1 归属");

    let held = |placed: &[Placement<()>], name: &str| {
        placed
            .iter()
            .find(|p| p.name() == name)
            .expect("层在清单里")
            .held()
    };

    assert!(held(&a, "blk.0.q") && !held(&a, "blk.1.q"));
    assert!(!held(&b, "blk.0.q") && held(&b, "blk.1.q"));
    // 非分层参数两个副本都持有。
    assert!(held(&a, "tok_embeddings") && held(&b, "tok_embeddings"));
}

#[test]
fn tp_column_splits_output_and_row_splits_input() {
    let topology = Topology::new(2, 1, 1).expect("拓扑合法");
    let assembler = Assembler::new(Plan::new(topology, 0).expect("rank 0"));
    let placed = assembler.place_stack(&sample(), 1).expect("归属");

    let q = placed
        .iter()
        .find(|p| p.name() == "blk.0.q")
        .expect("Q 列切");
    assert_eq!(q.shard_shape(), Some([12, 12]), "输出维 24 列切成两半");
    assert_eq!(q.shard_range(), Some((0, 12)));

    let o = placed
        .iter()
        .find(|p| p.name() == "blk.0.o")
        .expect("O 行切");
    assert_eq!(o.shard_shape(), Some([12, 12]), "输入维 24 行切成两半");
    assert_eq!(o.shard_range(), Some((0, 12)));
}

#[test]
fn block_outside_the_pipeline_has_no_tiling() {
    let topology = Topology::new(2, 2, 1).expect("拓扑合法");
    let assembler = Assembler::new(Plan::new(topology, 0).expect("rank 0"));
    let placed = assembler.place_stack(&sample(), 2).expect("归属");

    let outside = placed
        .iter()
        .find(|p| p.name() == "blk.1.q")
        .expect("段外层");
    assert!(!outside.held());
    assert!(outside.tiling().is_none());
    assert!(outside.shard_shape().is_none());
    assert!(outside.shard_range().is_none());
}

#[test]
fn every_block_is_held_by_exactly_one_pipeline_rank() {
    let topology = Topology::new(1, 2, 1).expect("拓扑合法");
    let blocks = 4;
    let stacks: Vec<_> = (0..topology.world())
        .map(|rank| {
            let plan = Plan::new(topology, rank as u32).expect("rank 在世界内");
            Assembler::new(plan)
                .place_stack(&sample(), blocks)
                .expect("归属")
        })
        .collect();

    for name in ["blk.0.q", "blk.1.q"] {
        let holders = stacks
            .iter()
            .flatten()
            .filter(|p| p.name() == name && p.held())
            .count();
        assert_eq!(holders, 1, "{name} 应恰好由一个 PP rank 持有");
    }
    // 非分层参数每个副本都持有。
    for stack in &stacks {
        assert!(
            stack
                .iter()
                .any(|p| p.name() == "tok_embeddings" && p.held())
        );
    }
}

#[test]
fn unsplittable_output_width_is_rejected() {
    let stack =
        Stack::new(vec![Linear::new("blk.0.q", 12, 23).expect("奇数输出宽")]).expect("层名唯一");
    let topology = Topology::new(2, 1, 1).expect("拓扑合法");
    let assembler = Assembler::new(Plan::new(topology, 0).expect("rank 0"));
    assert!(
        assembler.place_stack(&stack, 1).is_err(),
        "23 不能被 TP=2 整除"
    );
}

#[test]
fn projection_suffix_selects_the_sharding_side() {
    assert_eq!(sharding_of("blk.0.q"), Sharding::Column);
    assert_eq!(sharding_of("blk.0.k"), Sharding::Column);
    assert_eq!(sharding_of("blk.0.v"), Sharding::Column);
    assert_eq!(sharding_of("blk.0.o"), Sharding::Row);
    assert_eq!(
        sharding_of("blk.0.mlp"),
        Sharding::Column,
        "未登记角色回退列切"
    );
}

#[test]
fn held_projection_yields_local_slice_and_non_projection_has_none() {
    let topology = Topology::new(2, 1, 1).expect("拓扑合法");
    let assembler = Assembler::new(Plan::new(topology, 0).expect("rank 0"));
    let placed = assembler.place_stack(&sample(), 1).expect("归属");

    let q = placed
        .iter()
        .find(|p| p.name() == "blk.0.q")
        .expect("Q 列切");
    let full = vec![0.0f32; 24 * 12];
    let local = q.split(&full).expect("Q 有分片").expect("切片成功");
    assert_eq!(local.len(), 12 * 12, "列切子块形状 [12, 12]");

    let embed = placed
        .iter()
        .find(|p| p.name() == "tok_embeddings")
        .expect("嵌入表");
    assert!(embed.split(&full).is_none(), "非投影层不切片");
}

#[test]
fn pipeline_outside_layer_has_no_slice() {
    let topology = Topology::new(1, 2, 1).expect("拓扑合法");
    let assembler = Assembler::new(Plan::new(topology, 0).expect("rank 0"));
    let placed = assembler.place_stack(&sample(), 2).expect("归属");

    let outside = placed
        .iter()
        .find(|p| p.name() == "blk.1.q")
        .expect("段外层");
    assert!(outside.split(&[0.0f32; 24 * 12]).is_none());
}
