use super::*;

#[test]
fn a_split_always_leaves_a_training_row() {
    let split = Split::new(10, 30).expect("切分");
    assert_eq!((split.train(), split.valid()), (7, 3));
    assert_eq!(split.ranges(), (0..7, 7..10));
    let almost_all_valid = Split::new(100, 99).expect("切分");
    assert_eq!(
        (almost_all_valid.train(), almost_all_valid.valid()),
        (1, 99)
    );
    let half = Split::new(2, 50).expect("切分");
    assert_eq!((half.train(), half.valid()), (1, 1));
    // 2 行的 99% 是 1.98 行：向下取整得 1，于是训练集恰好留下 1 行（夹值在这里恒等）。
    let almost_half = Split::new(2, 99).expect("切分");
    assert_eq!((almost_half.train(), almost_half.valid()), (1, 1));
    let all_training = Split::new(100, 0).expect("切分");
    assert_eq!((all_training.train(), all_training.valid()), (100, 0));
    // 总行数只有一行时，任何非零比例都只能得到 0 验证行。
    for percent in [0, 1, 99] {
        let split = Split::new(1, percent).expect("切分");
        assert_eq!((split.train(), split.valid()), (1, 0), "百分比 {percent}");
    }
    assert_eq!(Split::new(0, 10).unwrap_err().code(), Code::Invalid);
    assert_eq!(Split::new(10, 101).unwrap_err().code(), Code::Invalid);
}

#[test]
fn a_split_may_not_hand_over_every_row() {
    for total in [1, 100] {
        assert_eq!(
            Split::new(total, 100).unwrap_err().code(),
            Code::Invalid,
            "total = {total} 时不该接受 100% 验证集"
        );
    }
}

#[test]
fn a_split_rounds_the_same_way_as_an_exact_proportion() {
    // 字面表，重点是余数：`100q + r` 的拆分只要漏掉 `r * p / 100` 这一项，
    // 下面的 `(99, 99)`、`(199, 50)`、`(12345, 37)` 就会各差一行或更多。
    for (total, percent, valid) in [
        (99, 99, 98),
        (101, 99, 99),
        (199, 50, 99),
        (999, 99, 989),
        (1000, 1, 10),
        (12345, 37, 4567),
    ] {
        assert_eq!(
            Split::new(total, percent).expect("切分").valid(),
            valid,
            "{total} 行的 {percent}% 向下取整应为 {valid}"
        );
    }
}

#[test]
fn a_split_at_usize_max_keeps_the_proportion() {
    let total = usize::MAX;
    let split = Split::new(total, 50).expect("一半");
    // 只断言性质，不重算实现里的同一公式。
    assert_eq!(
        split.train() + split.valid(),
        total,
        "两半必须恰好占满全部行"
    );
    assert!(split.train() >= 1, "训练集至少要留一行");
    assert!(
        split.valid() * 2 <= total,
        "验证集不能超过一半，饱和乘法会让它接近两倍"
    );
}
