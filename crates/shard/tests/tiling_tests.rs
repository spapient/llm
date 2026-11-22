use super::*;
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
