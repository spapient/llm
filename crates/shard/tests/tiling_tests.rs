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
#[test]
fn every_rank_together_rebuilds_the_split_dimension() {
    for sharding in Sharding::ALL {
        for parts in 1..=4 {
            let tiling = Tiling::new(sharding, parts, 24, 12).expect("整除");
            let covered: usize = (0..parts)
                .map(|rank| tiling.weight_shape(rank).expect("形状")[sharding.axis()])
                .sum();
            let whole = [24, 12][sharding.axis()];
            assert_eq!(covered, whole, "{sharding:?} 切 {parts} 份没有拼回 {whole}");
        }
    }
}

/// 分片是**整除**的等分：被切维恰好铺满一遍，起点相接、宽度全等。
///
/// 这是从 `Partition` 搬过来的"恰好覆盖"判据：`Tiling::new` 已经拒掉余数形状，所以全覆盖
/// 只剩"每份宽 `split / parts`、第 `rank` 份起点 `rank · 宽`"这一种形态。
#[test]
fn the_ranges_tile_the_split_axis_exactly_once() {
    for (split, parts) in [(12, 3), (16, 4), (7, 1), (1, 1), (10, 5)] {
        let width = split / parts;
        let tiling = Tiling::new(Sharding::Column, parts, split, 3).expect("整除");
        let mut cursor = 0;
        for rank in 0..parts {
            assert_eq!(
                tiling.range(rank),
                Ok((cursor, width)),
                "{split} 切 {parts} 份"
            );
            cursor += width;
        }
        assert_eq!(cursor, split, "{split} 切 {parts} 份没有铺满");
    }
}

/// TP 的硬要求：不整除就没有合法的分片形状，必须在构造期失败而不是摊余数。
///
/// 10 切 3 就是那条"均匀摊余数"的输入（旧 `Partition` 会给出 `[(0,4),(4,3),(7,3)]`）：
/// 各 rank 的权重形状不一致，这里必须拒绝。
#[test]
fn a_split_that_does_not_divide_is_refused() {
    let error = Tiling::new(Sharding::Column, 3, 8, 6).unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("整除"), "{error}");
    assert_eq!(
        Tiling::new(Sharding::Column, 3, 10, 6).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        Tiling::new(Sharding::Row, 5, 4, 6).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        Tiling::new(Sharding::Column, 0, 4, 6).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        Tiling::new(Sharding::Column, 2, 0, 6).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        Tiling::new(Sharding::Column, 2, 4, 6)
            .expect("整除")
            .range(2)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
    assert_eq!(
        Tiling::new(Sharding::Column, 2, 4, 6)
            .expect("整除")
            .weight_shape(2)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
}

#[test]
fn sharding_names_and_axes_are_stable() {
    let names: Vec<&str> = Sharding::ALL.iter().map(|kind| kind.as_str()).collect();
    assert_eq!(names, ["column", "row"]);
    assert_eq!(Sharding::Column.axis(), 0);
    assert_eq!(Sharding::Row.axis(), 1);
}
