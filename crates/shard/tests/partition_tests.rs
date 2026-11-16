use super::*;

/// 切分的判据是**恰好覆盖**：首尾相接、互不重叠、并集是 `0..dimension`。
///
/// 这条用例遍历余数不为零的形状：`0..parts` 任一份漏算，并集就会短一截或重叠。
#[test]
fn every_partition_covers_the_dimension_exactly_once() {
    for dimension in [0, 1, 2, 3, 10, 17] {
        for parts in 1..=5 {
            let partition = Partition::new(0, parts).expect("切分");
            let cover = partition.cover(dimension).expect("各份");
            let mut expected = 0;
            for (part, (start, take)) in cover.iter().enumerate() {
                assert_eq!(*start, expected, "第 {part} 份没接上前一份");
                assert_eq!(partition.end(dimension, part).expect("终点"), start + take);
                expected += take;
            }
            assert_eq!(expected, dimension, "{dimension} 切 {parts} 份没有覆盖完");
        }
    }
}

#[test]
fn the_remainder_goes_to_the_earlier_parts() {
    let partition = Partition::new(1, 3).expect("切分");
    assert_eq!(partition.axis(), 1);
    assert_eq!(partition.cover(10).expect("各份"), [(0, 4), (4, 3), (7, 3)]);
    assert_eq!(partition.cover(0).expect("空维"), [(0, 0), (0, 0), (0, 0)]);
}

#[test]
fn an_index_outside_the_split_is_rejected() {
    let partition = Partition::new(0, 4).expect("切分");
    assert_eq!(partition.slice(16, 3), Ok((12, 4)));
    assert_eq!(partition.slice(16, 4).unwrap_err().code(), Code::Invalid);
    assert_eq!(Partition::new(0, 0).unwrap_err().code(), Code::Invalid);
}
