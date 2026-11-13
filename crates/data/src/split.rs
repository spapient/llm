//! 训练集 / 验证集的行数划分。

use diagnostic::{Code, Error, Result};

/// 训练集 / 验证集的行数划分。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Split {
    train: usize,
    valid: usize,
}

impl Split {
    /// 按验证集百分比切分：`0..=99`，且总行数非零。
    ///
    /// 100% 验证集直接报错而不是夹成 0%：训练集至少要留一行，否则"训练"会在第一次取批时
    /// 就失败，而调用方看到的却是一个合法的 `(total, 0)`。
    pub fn new(total: usize, valid_percent: u8) -> Result<Self> {
        if total == 0 {
            return Err(Error::new(Code::Invalid, "总行数不能为 0").with("data::Split::new"));
        }
        if valid_percent >= 100 {
            return Err(Error::new(
                Code::Invalid,
                format!("验证集百分比 {valid_percent} 必须在 0..=99：训练集至少要留一行"),
            )
            .with("data::Split::new"));
        }
        let valid = valid_rows(total, valid_percent)?;
        // 兜底：`valid_percent <= 99` 时它恒等；留着是为了将来放宽百分比时不再产生 0 训练行。
        let valid = valid.min(total - 1);
        Ok(Self {
            train: total - valid,
            valid,
        })
    }

    /// 训练集行数。
    pub fn train(&self) -> usize {
        self.train
    }

    /// 验证集行数。
    pub fn valid(&self) -> usize {
        self.valid
    }

    /// 行区间：`(0..train, train..train+valid)`。
    pub fn ranges(&self) -> (core::ops::Range<usize>, core::ops::Range<usize>) {
        (0..self.train, self.train..self.train + self.valid)
    }
}

/// `total` 行的 `percent`%：中间宽度取 `u128`，因为 `total * percent` 在 `usize` 里会溢出，
/// 而 `saturating_mul` 在 `total` 接近 `usize::MAX` 时先饱和再除以 100，把比例放大近一倍。
fn valid_rows(total: usize, percent: u8) -> Result<usize> {
    let valid = u128::try_from(total)
        .ok()
        .and_then(|total| total.checked_mul(u128::from(percent)))
        .ok_or_else(|| Error::new(Code::Internal, "百分比算术溢出").with("data::Split::new"))?
        / 100;
    // `percent <= 99` 保证 `valid <= total`，所以这里必然装得回 `usize`。
    usize::try_from(valid)
        .map_err(|_| Error::new(Code::Internal, "验证集行数超出 usize").with("data::Split::new"))
}

#[cfg(test)]
mod tests {
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
}
