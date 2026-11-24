//! 张量并行的权重切分：[`Sharding`] 与 [`Tiling`]。

use crate::collective::Reduce;
use diagnostic::Result;

/// 张量并行里线性层的两种切法。
///
/// 判别式就是它在 `TABLE` 里的行号（`#[repr(usize)]`），于是短名、被切轴、所需归约三件事
/// 写在同一行里：加一种切法只加一行。旧形态是三个各自两臂的 `match`——"`Row` 忘了要
/// `Sum`"这类错只会在反向传播时表现为静默的梯度翻倍。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Sharding {
    /// 按列切：切输出维，分片直接进下一层，不需要通信。
    Column = 0,
    /// 按行切：切输入维，每个 rank 产出一份部分和，必须求和归约。
    Row = 1,
}

/// 一种切法的全部事实：短名、被切的是权重 `[out, in]` 的哪一维、合并所需的归约。
type Row = (&'static str, usize, Option<Reduce>);

impl Sharding {
    /// 全部切法，供日志与测试遍历。
    pub const ALL: [Self; 2] = [Self::Column, Self::Row];

    /// 每种切法一行。行序必须与判别式一致（三个访问器都走 `TABLE[self as usize]`），
    /// 这条由 `tests` 里的 `every_sharding_row_matches_its_discriminant` 钉住。
    const TABLE: [Row; 2] = [("column", 0, None), ("row", 1, Some(Reduce::Sum))];

    fn row(self) -> Row {
        Self::TABLE[self as usize]
    }

    /// 稳定的短名。
    pub fn as_str(self) -> &'static str {
        self.row().0
    }

    /// 被切的是权重 `[out, in]` 的哪一维。
    ///
    /// 全 crate 只有这一处把切法翻译成轴：分片形状、随份数变化的长度都从它派生。
    pub fn axis(self) -> usize {
        self.row().1
    }

    /// 合并各 rank 的输出需要哪种归约；`Column` 不需要（`None`）。
    pub fn merge(self) -> Option<Reduce> {
        self.row().2
    }
}

/// 一层线性权重的 TP 切分计划：切法、份数与完整的 `[out, in]` 形状。
///
/// 分片是**整除**的等分：`Tiling::new` 拒绝任何摊余数的形状，因此第 `rank` 份永远是
/// `(rank · 被切维长度 / 份数, 被切维长度 / 份数)`。TP 要求各 rank 的权重形状一致，
/// 有余数就没有合法的分片，故这里不保留"前几份多拿一个"的均匀切分——那条分支在任何
/// 可达路径上都取不到。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tiling {
    sharding: Sharding,
    parts: usize,
    shape: [usize; 2],
}

impl Tiling {
    /// 由切法、份数与完整权重形状构造；被切的那一维必须能被份数整除。
    ///
    /// # Errors
    ///
    /// 份数为零、形状里出现零宽，或被切维不能被份数整除。
    pub fn new(sharding: Sharding, parts: usize, out: usize, inp: usize) -> Result<Self> {
        if parts == 0 {
            fail!("shard::Tiling::new", "切分份数不能为 0");
        }
        if out == 0 || inp == 0 {
            fail!(
                "shard::Tiling::new",
                "权重形状 [{out}, {inp}] 含零宽：切出来是空分片"
            );
        }
        let shape = [out, inp];
        let split = shape[sharding.axis()];
        if !split.is_multiple_of(parts) {
            fail!(
                "shard::Tiling::new",
                "{} 的第 {} 维长度 {split} 不能被 {parts} 整除：各 rank 的权重形状会不一致",
                sharding.as_str(),
                sharding.axis()
            );
        }
        Ok(Self {
            sharding,
            parts,
            shape,
        })
    }

    /// 切法。
    pub fn sharding(&self) -> Sharding {
        self.sharding
    }

    /// 份数（通常等于 `Topology::tp()`）。
    pub fn parts(&self) -> usize {
        self.parts
    }

    /// 第 `rank` 份的权重形状 `[out, in]`：完整形状把被切维换成该份的长度。
    pub fn weight_shape(&self, rank: usize) -> Result<[usize; 2]> {
        let (_, width) = self.range(rank)?;
        let mut shape = self.shape;
        shape[self.sharding.axis()] = width;
        Ok(shape)
    }

    /// 第 `rank` 份在完整权重里的 `(起点, 长度)`，沿 [`Sharding::axis`] 那一维。
    pub fn range(&self, rank: usize) -> Result<(usize, usize)> {
        if rank >= self.parts {
            fail!(
                "shard::Tiling::range",
                "份下标 {rank} 超出 0..{}",
                self.parts
            );
        }
        let width = self.width();
        Ok((rank * width, width))
    }

    /// 取第 `rank` 份的本地权重子块（行主序 `[out, in]`）。
    ///
    /// 列切的行段在缓冲里连续，一次切出整段；行切的列段跨行分散，只能按行拷列段——行内仍是
    /// 整块拷贝，外层按行循环是数据布局决定的分散拷贝，不是逐元素遍历。切片只产出本 rank 本地
    /// 子块，不装载进图、不发起通信。
    ///
    /// # Errors
    ///
    /// `rank` 越出份数（复用 [`Self::range`]）；`full` 长度不等于完整形状积，或形状积溢出。
    pub fn split<T: Copy>(&self, rank: usize, full: &[T]) -> Result<Vec<T>> {
        let (start, width) = self.range(rank)?;
        let [out, inp] = self.shape;
        self.check_len(full.len(), out, inp)?;
        Ok(match self.sharding {
            // 输出维行段连续：整段一次切片，不逐元素拷。
            Sharding::Column => self.column_slice(start, width, inp, full),
            // 输入维列段：行与行之间不连续，外层按行取列段；每行内部整块复制。
            Sharding::Row => self.row_slice(start, width, inp, out, full),
        })
    }

    /// 校验完整权重长度等于形状积；形状积溢出或不符即拒绝。
    fn check_len(&self, full_len: usize, out: usize, inp: usize) -> Result<()> {
        let expected = out
            .checked_mul(inp)
            .ok_or_else(|| invalid!("shard::Tiling::split", "权重形状积溢出"))?;
        if full_len != expected {
            fail!(
                "shard::Tiling::split",
                "权重长度 {full_len} 与完整形状 [{out}, {inp}] 的积 {expected} 不符",
            );
        }
        Ok(())
    }

    /// 列切：行段在缓冲里连续，一次切出整段。
    fn column_slice<T: Copy>(&self, start: usize, width: usize, inp: usize, full: &[T]) -> Vec<T> {
        let lo = start * inp;
        full[lo..lo + width * inp].to_vec()
    }

    /// 行切：列段跨行分散，外层按行取列段（布局决定的分散拷贝，非逐元素遍历），行内整块复制。
    fn row_slice<T: Copy>(&self, start: usize, width: usize, inp: usize, out: usize, full: &[T]) -> Vec<T> {
        let mut local = Vec::with_capacity(out * width);
        for row in 0..out {
            let begin = row * inp;
            local.extend_from_slice(&full[begin + start..begin + start + width]);
        }
        local
    }

    /// 被切维在每份里的长度：构造期已保证整除。
    fn width(&self) -> usize {
        self.shape[self.sharding.axis()] / self.parts
    }
}

#[cfg(test)]
#[path = "../tests/tiling_tests.rs"]
mod tests;
