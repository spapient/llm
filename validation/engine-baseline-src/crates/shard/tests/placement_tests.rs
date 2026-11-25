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

/// 装好 rank 0 的装配器并放置两层样例栈（`dp=1`）；单 rank 用例共用这一段装配样板。
fn placed(tp: usize, pp: usize, blocks: usize) -> Vec<Placement<()>> {
    let topology = Topology::new(tp, pp, 1).expect("拓扑合法");
    let assembler = Assembler::new(Plan::new(topology, 0).expect("rank 0"));
    assembler.place_stack(&sample(), blocks).expect("归属")
}

/// 按层名取放置结果；用例只关心命中的那一层。
fn by_name<'a>(placed: &'a [Placement<()>], name: &str) -> &'a Placement<()> {
    placed
        .iter()
        .find(|p| p.name() == name)
        .expect("层在清单里")
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
    let placed = placed(2, 1, 1);

    let q = by_name(&placed, "blk.0.q");
    assert_eq!(q.shard_shape(), Some([12, 12]), "输出维 24 列切成两半");
    assert_eq!(q.shard_range(), Some((0, 12)));

    let o = by_name(&placed, "blk.0.o");
    assert_eq!(o.shard_shape(), Some([12, 12]), "输入维 24 行切成两半");
    assert_eq!(o.shard_range(), Some((0, 12)));
}

#[test]
fn block_outside_the_pipeline_has_no_tiling() {
    let placed = placed(2, 2, 2);

    let outside = by_name(&placed, "blk.1.q");
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
    let placed = placed(2, 1, 1);

    let q = by_name(&placed, "blk.0.q");
    let full = vec![0.0f32; 24 * 12];
    let local = q.split(&full).expect("Q 有分片").expect("切片成功");
    assert_eq!(local.len(), 12 * 12, "列切子块形状 [12, 12]");

    let embed = by_name(&placed, "tok_embeddings");
    assert!(embed.split(&full).is_none(), "非投影层不切片");
}

#[test]
fn pipeline_outside_layer_has_no_slice() {
    let placed = placed(1, 2, 2);

    let outside = by_name(&placed, "blk.1.q");
    assert!(outside.split(&[0.0f32; 24 * 12]).is_none());
}

// ---- tiling_tests.rs ----

use diagnostic::Code;

#[test]
fn column_sharding_splits_the_output_rows() {
    let tiling = Tiling::new(Sharding::Column, 2, 8, 6).expect("整除");
    assert_eq!(tiling.sharding(), Sharding::Column);
    assert_eq!(tiling.parts(), 2);
    assert_eq!(tiling.range(0), Ok((0, 4)));
    assert_eq!(tiling.range(1), Ok((4, 4)));
    assert_eq!(tiling.weight_shape(0), Ok([4, 6]));
    assert_eq!(tiling.weight_shape(1), Ok([4, 6]));
    assert_eq!(
        tiling.sharding().merge(),
        None,
        "按列切的分片直接进下一层，不需要通信"
    );
}

#[test]
fn row_sharding_splits_the_input_columns_and_needs_a_sum() {
    let tiling = Tiling::new(Sharding::Row, 4, 3, 8).expect("整除");
    assert_eq!(tiling.sharding().axis(), 1);
    assert_eq!(tiling.range(3), Ok((6, 2)));
    assert_eq!(tiling.weight_shape(3), Ok([3, 2]));
    assert_eq!(tiling.sharding().merge(), Some(Reduce::Sum));
}

/// 各 rank 的分片必须拼回原始那一维：漏一块是"某些列没人算"，重一块是"某些列算两遍"。
///
/// 整除形态下这件事只剩两条要核对——每份宽 `split / parts` 且起点相接、各份的权重形状沿被切
/// 维加起来等于整条轴。逐元素拼回原轴的端到端判据在 `properties` 里由性质测试验证。
#[test]
fn every_rank_gets_a_slice_and_together_they_rebuild_the_axis() {
    for sharding in Sharding::ALL {
        for (split, parts) in [(24, 4), (12, 3), (7, 1), (10, 5)] {
            let (out, inp) = if sharding.axis() == 0 {
                (split, 12)
            } else {
                (12, split)
            };
            let tiling = Tiling::new(sharding, parts, out, inp).expect("整除");
            let width = split / parts;
            let mut cursor = 0;
            for rank in 0..parts {
                assert_eq!(
                    tiling.range(rank),
                    Ok((cursor, width)),
                    "{sharding:?} 把 {split} 切 {parts} 份：第 {rank} 份的起点没接上前一份"
                );
                assert_eq!(
                    tiling.weight_shape(rank).expect("形状")[sharding.axis()],
                    width,
                    "{sharding:?} 第 {rank} 份的宽度不一致"
                );
                cursor += width;
            }
            assert_eq!(cursor, split, "{sharding:?} 切 {parts} 份没有铺满 {split}");
        }
    }
}

/// TP 的硬要求：不整除就没有合法的分片形状，必须在构造期失败而不是摊余数。
///
/// 10 切 3 就是那条"均匀摊余数"的输入（旧 `Partition` 会给出 `[(0,4),(4,3),(7,3)]`）：
/// 各 rank 的权重形状不一致，这里必须拒绝。
/// TP 的硬要求：不整除就没有合法的分片形状，必须在构造期失败而不是摊余数。
///
/// 10 切 3 就是那条"均匀摊余数"的输入（旧 `Partition` 会给出 `[(0,4),(4,3),(7,3)]`）：
/// 各 rank 的权重形状不一致，这里必须拒绝。
#[test]
fn a_split_that_does_not_divide_is_refused() {
    let error = Tiling::new(Sharding::Column, 3, 8, 6).unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("整除"), "{error}");
    for (sharding, parts, out, inp) in [
        (Sharding::Column, 3, 10, 6),
        (Sharding::Row, 5, 4, 6),
        (Sharding::Column, 0, 4, 6),
        (Sharding::Column, 2, 0, 6),
    ] {
        assert_eq!(
            Tiling::new(sharding, parts, out, inp).unwrap_err().code(),
            Code::Invalid,
            "{sharding:?} 切 {parts} 份、形状 [{out}, {inp}] 必须被拒"
        );
    }
    // 份下标越界：两个派生量共用同一条判据。
    let tiling = Tiling::new(Sharding::Column, 2, 4, 6).expect("整除");
    assert_eq!(tiling.range(2).unwrap_err().code(), Code::Invalid);
    assert_eq!(tiling.weight_shape(2).unwrap_err().code(), Code::Invalid);
}

#[test]
fn sharding_names_and_axes_are_stable() {
    let names: Vec<&str> = Sharding::ALL.iter().map(|kind| kind.as_str()).collect();
    assert_eq!(names, ["column", "row"]);
    assert_eq!(Sharding::Column.axis(), 0);
    assert_eq!(Sharding::Row.axis(), 1);
}

/// 从定义推出、不重述实现的性质：切分的代数定律与"拼得回原样"的端到端定律。
///
/// 与上面的例子测试分工：例子给出可读的具体形状，这里给出对**所有** `parts`/`blocks` 成立的
/// 定律。旧形态是四份定律在 `tests/prop_tiling.rs` 里另挂一个模块，改动 `Tiling` 要在两个文件
/// 之间对照；同一被测对象的定律就放在这个文件里。
///
/// * 定义是"把权重某一维恰好分给每个 rank"——所以三段定律：首份起点为 0、相邻份首尾相接、
///   末份终点是整条轴。缺任何一条都会放过一类缺陷（漏算末份、重叠、错位）。
/// * 端到端定律是本模块的核心：把每一份真的取出来按份序拼回去，必须逐元素等于原轴。
///   只测 `(start, take)` 二元组时，"起点对了但取宽度算错"仍能通过；拼回去不能。
/// * 二维定律：各 rank 的权重形状元素数之和必须等于完整权重元素数——这是"分片不重叠且不
///   丢失"在 `[out, in]` 层面的同一句话，也是 TP 里唯一会被真正读到的量。
///
/// 分片是**整除**的：`Tiling::new` 拒绝任何摊余数的形状，所以定律两端都由 `parts` 派生。
/// "何时接受、何时拒绝"另立一条可接受性定律，不靠取样去碰余数。
mod properties {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(testkit::prop::config())]

        /// 可接受性：被切维能被份数整除时构造成功，否则失败——不多拒也不少拒。
        #[test]
        fn a_split_is_accepted_exactly_when_it_divides(parts in 1usize..=64, split in 1usize..4096) {
            for sharding in Sharding::ALL {
                let shape = if sharding.axis() == 0 {
                    (split, 3)
                } else {
                    (3, split)
                };
                let accepted = Tiling::new(sharding, parts, shape.0, shape.1).is_ok();
                prop_assert_eq!(
                    accepted,
                    split.is_multiple_of(parts),
                    "{} 对 {} 切 {} 份的判据与整除性不一致",
                    sharding.as_str(),
                    split,
                    parts
                );
            }
        }

        /// 切分恰好覆盖被切维：不重叠、不留缝、不多算。
        #[test]
        fn the_ranges_tile_the_split_axis_exactly(parts in 1usize..=64, blocks in 1usize..=64) {
            let split = parts * blocks;
            let tiling = Tiling::new(Sharding::Column, parts, split, 3).expect("可整除");
            let mut cursor = 0usize;
            for rank in 0..parts {
                let (start, take) = tiling.range(rank).expect("份下标在区间内");
                prop_assert_eq!(start, cursor, "第 {} 份的起点没接上前一份的终点", rank);
                cursor += take;
            }
            prop_assert_eq!(cursor, split, "末份终点不是轴长：漏算或多算");
        }

        /// 端到端：逐份取出再按份序拼回，必须逐元素等于原轴。
        #[test]
        fn slicing_then_concatenating_restores_the_axis(parts in 1usize..=64, blocks in 1usize..=64) {
            let split = parts * blocks;
            let tiling = Tiling::new(Sharding::Row, parts, 3, split).expect("可整除");
            let source: Vec<usize> = (0..split).collect();
            let mut rebuilt = Vec::with_capacity(split);
            for rank in 0..parts {
                let (start, take) = tiling.range(rank).expect("份下标在区间内");
                rebuilt.extend_from_slice(&source[start..start + take]);
            }
            prop_assert_eq!(rebuilt, source, "拼回来的轴与原轴不同：某份取宽或起点错了");
        }

        /// 二维：各 rank 的权重形状元素数之和 = 完整权重元素数（不重叠且不丢失）。
        #[test]
        fn the_weight_shapes_partition_the_full_shape(
            parts in 1usize..=64,
            out_blocks in 1usize..=64,
            inp_blocks in 1usize..=64,
        ) {
            let out = parts * out_blocks;
            let inp = parts * inp_blocks;
            for sharding in Sharding::ALL {
                let tiling = Tiling::new(sharding, parts, out, inp).expect("可整除");
                let mut total = 0usize;
                for rank in 0..parts {
                    let shape = tiling.weight_shape(rank).expect("份下标在区间内");
                    total += shape[0] * shape[1];
                }
                prop_assert_eq!(
                    total,
                    out * inp,
                    "{} 切 {} 份后各 rank 元素数之和与完整形状不符",
                    sharding.as_str(),
                    parts
                );
            }
        }
    }
}

/// 行序 = 判别式：三个访问器都走 `TABLE[self as usize]`，行序错了只会静默切错轴或漏掉归约。
#[test]
fn every_sharding_row_matches_its_discriminant() {
    let table = [
        (Sharding::Column, "column", 0usize, None),
        (Sharding::Row, "row", 1, Some(Reduce::Sum)),
    ];
    for (row, (sharding, name, axis, merge)) in table.into_iter().enumerate() {
        assert_eq!(sharding as usize, row, "{name} 的行号与判别式不一致");
        assert_eq!(sharding.as_str(), name);
        assert_eq!(sharding.axis(), axis);
        assert_eq!(sharding.merge(), merge);
    }
}

/// 列切取连续行段：两份按行序拼回必须逐元素等于原权重。
#[test]
fn column_split_keeps_the_contiguous_row_block() {
    let tiling = Tiling::new(Sharding::Column, 2, 4, 2).expect("整除");
    let full: Vec<i32> = (0..8).collect();

    let local = |rank| tiling.split(rank, &full).expect("份下标在区间内");
    let (a, b) = (local(0), local(1));

    assert_eq!(a, [0, 1, 2, 3]);
    assert_eq!(b, [4, 5, 6, 7]);
    assert_eq!([a, b].concat(), full, "列切两份按行序拼回原样");
}

/// 行切按行取列段：每行左右两段接回原行；跨行不连续，子块行序与原权重一致。
#[test]
fn row_split_cuts_columns_within_each_row() {
    let tiling = Tiling::new(Sharding::Row, 2, 2, 4).expect("整除");
    let full: Vec<i32> = (0..8).collect();

    let a = tiling.split(0, &full).expect("rank 0");
    let b = tiling.split(1, &full).expect("rank 1");
    assert_eq!(a, [0, 1, 4, 5]);
    assert_eq!(b, [2, 3, 6, 7]);

    for row in 0..2 {
        let lo = row * 4;
        let merged = [&a[row * 2..row * 2 + 2], &b[row * 2..row * 2 + 2]].concat();
        assert_eq!(merged, full[lo..lo + 4], "第 {row} 行左右两段接回原行");
    }
}

/// 切片泛型到任意 `Copy` 元素档：u32 与 i32 各自独立于归约标量档。
#[test]
fn split_is_generic_over_a_copy_element() {
    let tiling = Tiling::new(Sharding::Row, 2, 2, 2).expect("整除");
    let full = vec![1u32, 2, 3, 4];
    let local = tiling.split(1, &full).expect("u32 档切片");
    assert_eq!(local, [2, 4]);
}

/// 缓冲长度不符形状积、份下标越界都必须拒绝；切片不做长度猜测补齐。
#[test]
fn split_rejects_length_mismatch_and_rank_out_of_range() {
    let tiling = Tiling::new(Sharding::Column, 2, 4, 2).expect("整除");
    let wrong = vec![0i32; 7];
    assert_eq!(
        tiling.split(0, &wrong).unwrap_err().code(),
        Code::Invalid,
        "长度 7 不等于形状积 8"
    );
    assert_eq!(
        tiling.split(2, &[0i32; 8]).unwrap_err().code(),
        Code::Invalid,
        "份下标 2 越出 0..2"
    );
}

#[test]
fn split_into_reuses_capacity_and_never_exposes_a_partial_shard() {
    let tiling = Tiling::new(Sharding::Row, 2, 2, 4).expect("整除");
    let full: Vec<i32> = (0..8).collect();
    let mut local = Vec::with_capacity(4);
    let capacity = local.capacity();
    tiling.split_into(0, &full, &mut local).expect("首份");
    assert_eq!(local, [0, 1, 4, 5]);
    tiling.split_into(1, &full, &mut local).expect("次份");
    assert_eq!(local, [2, 3, 6, 7]);
    assert_eq!(local.capacity(), capacity, "复用时不应重新分配");

    assert!(tiling.split_into(0, &full[..7], &mut local).is_err());
    assert!(local.is_empty(), "失败不能留下旧值或半份新值");
}
