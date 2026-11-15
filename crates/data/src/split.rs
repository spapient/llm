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
        let valid = valid_rows(total, valid_percent);
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

/// `total` 行的 `percent`%：拆成「整百部分」与「余数部分」两次乘法，两项都装得进 `usize`。
///
/// 记 `total = 100q + r`（`r < 100`），则 `total * p / 100 == q * p + r * p / 100`：前者因
/// `p <= 99` 而 `q * p <= total * p / 100 <= total`，后者 `r * p / 100 < 100`，所以既不会溢出，
/// 也不需要「中间宽度」或饱和。饱和正是上一轮要修掉的那件事——`saturating_mul` 在 `total` 接近
/// `usize::MAX` 时先把乘积钉在 `usize::MAX` 再除以 100，会把比例放大近一倍。
fn valid_rows(total: usize, percent: u8) -> usize {
    let percent = usize::from(percent);
    let (hundreds, rest) = (total / 100, total % 100);
    hundreds * percent + rest * percent / 100
}

#[cfg(test)]
mod tests;
