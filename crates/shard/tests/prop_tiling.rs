//! `llm-shard` 的性质测试：切分的代数定律与"拼得回原样"的端到端定律。
//!
//! 这里只写**从定义推出**的性质，不重述实现：
//!
//! * 切分的定义是"把 `0..dimension` 恰好分成 `parts` 段"——所以三段定律：
//!   首段起点为 0、相邻段首尾相接、末段终点是 `dimension`。三者合起来等价于
//!   "并集等于全区且互不重叠"，缺任何一条都会放过一类缺陷（漏算末段、重叠、错位）。
//! * 端到端定律是本文件的核心：把每一份真的取出来按份序拼回去，必须逐元素等于原轴。
//!   只测 `(start, take)` 三元组时，"起点对了但取宽度算错"仍能通过；拼回去不能。
//! * 二维定律：各 rank 的权重形状元素数之和必须等于完整权重元素数——这是"分片不重叠
//!   且不丢失"在 `[out, in]` 层面的同一句话，也是 TP 里唯一会被真正读到的量。
//!
//! 空分片是**合法**的（`dimension < parts` 时后几份宽度为 0），所以定律只禁止负数与重叠，
//! 不对宽度设下界——把宽度钉成正数会在合法输入上假红。

use crate::Partition;
use proptest::prelude::*;

proptest! {
    #![proptest_config(testkit::prop::config())]

    /// 切分恰好覆盖 `0..dimension`：不重叠、不留缝、不多算。
    #[test]
    fn the_cover_tiles_the_axis_exactly(dimension in 0usize..4096, parts in 1usize..=64) {
        let partition = Partition::new(0, parts).expect("份数非零");
        let cover = partition.cover(dimension).expect("各份");
        prop_assert_eq!(cover.len(), parts, "份数必须与请求一致");
        let mut cursor = 0usize;
        for (part, (start, take)) in cover.iter().enumerate() {
            prop_assert_eq!(*start, cursor, "第 {} 份的起点没接上前一份的终点", part);
            cursor += take;
        }
        prop_assert_eq!(cursor, dimension, "末份终点不是轴长：漏算或多算");
    }

    /// 端到端：逐份取出再按份序拼回，必须逐元素等于原轴。
    #[test]
    fn slicing_then_concatenating_restores_the_axis(
        dimension in 0usize..4096,
        parts in 1usize..=64,
    ) {
        let partition = Partition::new(0, parts).expect("份数非零");
        let source: Vec<usize> = (0..dimension).collect();
        let mut rebuilt = Vec::with_capacity(dimension);
        for part in 0..parts {
            let (start, take) = partition.slice(dimension, part).expect("份下标在区间内");
            rebuilt.extend_from_slice(&source[start..start + take]);
        }
        prop_assert_eq!(rebuilt, source, "拼回来的轴与原轴不同：某份取宽或起点错了");
    }

    /// 二维：各 rank 的权重形状元素数之和 = 完整权重元素数（不重叠且不丢失）。
    ///
    /// `Tiling::new` 要求被切维能被份数整除，故两端都按 `parts` 派生，不在这里宽恕余数。
    #[test]
    fn the_weight_shapes_partition_the_full_shape(
        parts in 1usize..=64,
        out_blocks in 1usize..=64,
        inp_blocks in 1usize..=64,
    ) {
        let out = parts * out_blocks;
        let inp = parts * inp_blocks;
        for sharding in crate::Sharding::ALL {
            let tiling = crate::Tiling::new(sharding, parts, out, inp).expect("可整除");
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
