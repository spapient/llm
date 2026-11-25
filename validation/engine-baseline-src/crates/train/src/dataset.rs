//! 训练数据集管线：源条目流 → 采样序 → 批装配。
//!
//! [`Dataset`] 是按批取样的对象安全契约；[`Loader`]/[`Chunks`] 是唯一装配实现。
//! 批借用调用方缓冲，不拷贝、不拥有；采样策略及 epoch 生命周期由 `sampler` 持有。
//! 分词、打包与训练/验证集切分不属于本模块。

use diagnostic::{Code, Error, Result};

pub use crate::rng::{LoaderConfig, Rng, Sampling};
#[derive(Debug, Clone)]
enum Policy {
    Sequential,
    Shuffled {
        seed: u64,
    },
    Weighted {
        weights: Vec<f32>,
        total: f32,
        seed: u64,
    },
}

/// 一条采样序：策略只保存差异，索引容量、游标和 epoch 在三种策略间共享。
#[derive(Debug, Clone)]
pub(crate) struct Sampler {
    policy: Policy,
    order: Vec<usize>,
    cursor: usize,
    epoch: u64,
}

fn configured_policy(n: usize, sampling: Sampling) -> Result<Policy> {
    Ok(match sampling {
        Sampling::Sequential => Policy::Sequential,
        Sampling::Shuffle { seed } => Policy::Shuffled { seed },
        Sampling::Weighted { weights, seed } => {
            let total = validate_weights(&weights, n)?;
            Policy::Weighted {
                weights,
                total,
                seed,
            }
        }
    })
}

impl Sampler {
    /// 按互斥配置装配采样序；首轮与后续 epoch 共用同一个索引重建入口。
    pub(crate) fn from_config(n: usize, sampling: Sampling) -> Result<Self> {
        let mut sampler = Self {
            policy: configured_policy(n, sampling)?,
            order: (0..n).collect(),
            cursor: 0,
            epoch: 0,
        };
        sampler.rebuild(0);
        Ok(sampler)
    }

    /// 还没取走的批数。
    pub(crate) fn remaining(&self) -> usize {
        self.order.len() - self.cursor
    }

    /// 取下一个批索引；`None` 当且仅当本轮取空。
    pub(crate) fn next(&mut self) -> Option<usize> {
        let idx = *self.order.get(self.cursor)?;
        self.cursor += 1;
        Some(idx)
    }

    /// 进入下一轮：游标归位，策略只通过 epoch 改变派生随机流。
    pub(crate) fn reset(&mut self) {
        self.epoch += 1;
        self.rebuild(self.epoch);
        self.cursor = 0;
    }

    /// 按派生 epoch 原位重建索引，保留索引表容量。
    fn rebuild(&mut self, epoch: u64) {
        match &mut self.policy {
            Policy::Sequential => {}
            Policy::Shuffled { seed } => {
                shuffle(&mut Rng::with_stream(*seed, epoch as u32), &mut self.order);
            }
            Policy::Weighted {
                weights,
                total,
                seed,
            } => draw_round(weights, *total, *seed, epoch, &mut self.order),
        }
    }
}

/// Fisher–Yates 洗牌：从末位向前，每位与 `[0, i]` 内随机位交换。
fn shuffle(rng: &mut Rng, order: &mut [usize]) {
    for i in (1..order.len()).rev() {
        let j = (rng.next_u32() as usize) % (i + 1);
        order.swap(i, j);
    }
}

/// 校验权重：长度对齐、有限、非负、总质量为正；返回总权重。
fn validate_weights(weights: &[f32], n: usize) -> Result<f32> {
    validation_rules!(SITE;
        weights.len() == n => format!("权重条数 {} 与批数 {n} 不符", weights.len()),
        weights.iter().all(|&w| w.is_finite() && w >= 0.0) => "权重必须是有限非负数",
    );
    let total: f32 = weights.iter().sum();
    validation_rules!(SITE;
        total.is_finite() && total > 0.0 => "权重总质量必须是有限正数",
    );
    Ok(total)
}

/// 轮盘有放回原位重抽；保留调用方索引缓冲的容量。
pub(crate) fn draw_round(weights: &[f32], total: f32, seed: u64, epoch: u64, out: &mut Vec<usize>) {
    let mut rng = Rng::with_stream(seed, epoch as u32);
    let n = out.len();
    out.clear();
    out.extend((0..n).map(|_| draw(&mut rng, weights, total)));
}

/// 抽一个批索引：命中首个累计权重超过随机靶值的位置。
fn draw(rng: &mut Rng, weights: &[f32], total: f32) -> usize {
    let target = rng.next_f32() * total;
    let mut acc = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if target < acc {
            return i;
        }
    }
    weights.len() - 1
}

const SITE: &str = "train::dataset";

/// 批几何：`rows`、`cols` 非零且乘积不溢出，返回每批元素数（`rows * cols`）。
pub(crate) fn stride(rows: usize, cols: usize, site: &str) -> Result<usize> {
    validation_rules!(site;
        rows != 0 => "批行数不能为 0",
        cols != 0 => "列宽不能为 0",
    );
    rows.checked_mul(cols).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("rows * cols 溢出（rows = {rows}，cols = {cols}）"),
        )
        .with(site)
    })
}

/// 一个批：`rows x cols` 个条目，行主序；借用调用方的缓冲，不拥有。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch<'a, Item> {
    items: &'a [Item],
    rows: usize,
    cols: usize,
}

impl<'a, Item> Batch<'a, Item> {
    /// 构造：`rows` 与 `cols` 都非零，且 `items.len() == rows * cols`。
    pub fn new(items: &'a [Item], rows: usize, cols: usize) -> Result<Self> {
        let expected = stride(rows, cols, SITE)?;
        validation_rules!(SITE;
            items.len() == expected => format!("条目数 {} 与 {rows} x {cols} 不符", items.len()),
        );
        Ok(Self::exact(items, rows, cols))
    }

    /// 不变量已由调用方在构造期保证（长度等于 `rows * cols` 且两者非零）时使用。
    fn exact(items: &'a [Item], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(items.len(), rows * cols, "内部批次长度必须等于 rows * cols");
        Self { items, rows, cols }
    }

    getters! {
        /// 条目序列（行主序）。
        items => items: &'a [Item],
        /// 行数。
        rows => rows: usize,
        /// 列数。
        cols => cols: usize,
    }
}

impl<'a> Batch<'a, u32> {
    getters! {
        /// token 序列（`Item = u32` 的兼容访问器）。
        tokens => items: &'a [u32],
    }
}

/// 数据集：按批取样本。三个计数语义互不重叠：列宽、完整批总数、还没取走的完整批数。
pub trait Dataset<'a> {
    /// 条目形态；当前唯一实现为 `u32`。
    type Item;

    /// 列宽（每行多少条目）。
    fn cols(&self) -> usize;

    /// 完整批的**总数**——不随消费推进。
    fn batches(&self) -> usize;

    /// **还没取走**的完整批数。
    fn remaining_batches(&self) -> usize;

    /// 取下一批；批次借用源缓冲，`None` 当且仅当本轮已取完。
    fn next_batch(&mut self) -> Option<Batch<'a, Self::Item>>;

    /// 回到本轮开头；多 epoch 复用时每轮开始前调用。
    fn reset(&mut self);
}

/// 批装配器：持有条目流、批几何与采样序，实现 [`Dataset`]；条目类型 `T` 由源缓冲决定。
#[derive(Debug, Clone)]
pub struct Loader<'a, T> {
    items: &'a [T],
    rows: usize,
    cols: usize,
    stride: usize,
    batches: usize,
    sampler: Sampler,
}

impl<'a, T> Loader<'a, T> {
    /// 旧顺序构造：连续条目流按固定几何切批（等价于顺序 [`LoaderConfig`]）。
    pub fn new(items: &'a [T], rows: usize, cols: usize) -> Result<Self> {
        let config = LoaderConfig::new(rows, cols)?;
        Self::with_config(items, config)
    }

    /// 按配置装配：几何与权重校验收口在构造期。
    pub fn with_config(items: &'a [T], config: LoaderConfig) -> Result<Self> {
        let stride = stride(config.rows, config.cols, SITE)?;
        let batches = items.len() / stride;
        let sampler = Sampler::from_config(batches, config.sampling)?;
        Ok(Self {
            items,
            rows: config.rows,
            cols: config.cols,
            stride,
            batches,
            sampler,
        })
    }
}

/// 顺序连续流的旧类型名，兼容既有导出面。
pub type Chunks<'a, T> = Loader<'a, T>;

/// 迭代器：`next` 就是 [`Dataset::next_batch`]，「还剩几条」只有一个来源。
impl<'a, T> Iterator for Loader<'a, T> {
    type Item = Batch<'a, T>;

    fn next(&mut self) -> Option<Batch<'a, T>> {
        self.next_batch()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining_batches();
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for Loader<'_, T> {}

impl<T> core::iter::FusedIterator for Loader<'_, T> {}

impl<'a, T> Dataset<'a> for Loader<'a, T> {
    type Item = T;

    fn cols(&self) -> usize {
        self.cols
    }

    fn batches(&self) -> usize {
        self.batches
    }

    fn remaining_batches(&self) -> usize {
        self.sampler.remaining()
    }

    fn next_batch(&mut self) -> Option<Batch<'a, T>> {
        let idx = self.sampler.next()?;
        let start = idx * self.stride;
        let end = start + self.stride;
        Some(Batch::exact(&self.items[start..end], self.rows, self.cols))
    }

    fn reset(&mut self) {
        self.sampler.reset();
    }
}

#[cfg(test)]
#[path = "../tests/samples_tests.rs"]
mod tests;
