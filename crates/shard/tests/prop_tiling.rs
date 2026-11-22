//! `llm-shard` 的性质测试：切分的代数定律与"拼得回原样"的端到端定律。
//!
//! 这里只写**从定义推出**的性质，不重述实现：
//!
//! * 切分的定义是"把权重某一维恰好分给每个 rank"——所以三段定律：首份起点为 0、相邻份首尾
//!   相接、末份终点是整条轴。三者合起来等价于"并集等于全区且互不重叠"，缺任何一条都会放过
//!   一类缺陷（漏算末份、重叠、错位）。
//! * 端到端定律是本文件的核心：把每一份真的取出来按份序拼回去，必须逐元素等于原轴。
//!   只测 `(start, take)` 二元组时，"起点对了但取宽度算错"仍能通过；拼回去不能。
//! * 二维定律：各 rank 的权重形状元素数之和必须等于完整权重元素数——这是"分片不重叠且不
//!   丢失"在 `[out, in]` 层面的同一句话，也是 TP 里唯一会被真正读到的量。
//!
//! 分片是**整除**的：`Tiling::new` 拒绝任何摊余数的形状，所以定律两端都由 `parts` 派生。
//! "何时接受、何时拒绝"另立一条可接受性定律，不靠取样去碰余数。

use crate::{Sharding, Tiling};
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
