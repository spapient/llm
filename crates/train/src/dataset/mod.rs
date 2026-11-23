//! 训练数据集管线：源（条目流）→ 采样序（[`sampler`]）→ 批装配（[`Loader`]）三层分离。
//!
//! [`Dataset`] 是按批取样本的对象安全契约，关联类型 [`Dataset::Item`] 表达条目形态；
//! [`Loader`]/[`Chunks`] 是带泛型参数 `T` 的唯一装配实现，当前消费者只实例化 `T = u32`
//! （行主序 token）。批借用调用方缓冲，不拷贝、不拥有，批次的生命周期停在调用方那份缓冲上。
//!
//! **不拥有**：分词、打包与训练/验证集切分（打乱只改变采样序的游标推进方式）；条目型样本
//! （输入/目标对）、逐条目变换与多源拼接待真实消费者出现后再挂到源层。
//!
//! **预留插点**（证据不足，未实现）：后台预取队列、加权抽样的别名表 O(1) 索引、批次到张量
//! 视图的装配；`Loader<T>` 的泛型层级已为这些落点留好条目类型缝。

mod sampler;

use diagnostic::{Code, Error, Result};

use self::sampler::Sampler;

/// 本模块诊断站点。
const SITE: &str = "train::dataset";

/// 批几何：`rows`、`cols` 非零且乘积不溢出，返回每批元素数（`rows * cols`）。
fn stride(rows: usize, cols: usize, site: &str) -> Result<usize> {
    if rows == 0 {
        return Err(Error::new(Code::Invalid, "批行数不能为 0").with(site));
    }
    if cols == 0 {
        return Err(Error::new(Code::Invalid, "列宽不能为 0").with(site));
    }
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
        if items.len() != expected {
            return Err(Error::new(
                Code::Invalid,
                format!("条目数 {} 与 {rows} x {cols} 不符", items.len()),
            )
            .with(SITE));
        }
        Ok(Self { items, rows, cols })
    }

    /// 不变量已由调用方在构造期保证（长度等于 `rows * cols` 且两者非零）时使用。
    fn exact(items: &'a [Item], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(items.len(), rows * cols, "内部批次长度必须等于 rows * cols");
        Self { items, rows, cols }
    }

    /// 条目序列（行主序）。
    pub fn items(&self) -> &'a [Item] {
        self.items
    }

    /// 行数。
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// 列数。
    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl<'a> Batch<'a, u32> {
    /// token 序列（`Item = u32` 的兼容访问器）。
    pub fn tokens(&self) -> &'a [u32] {
        self.items
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

/// 批装配配置：几何与采样方式一次性收口到构造期，热路径不再做合法性分支检查。
///
/// 采样三态互斥，按 [`Loader::with_config`] 的装配次序裁决：给了 [`weights`](Self::weights)
/// 就走加权有放回抽样，否则给了 [`shuffle`](Self::shuffle) 就按种子洗牌，两者都缺走顺序。
/// 洗牌与加权的随机流只由种子派生，同一种子重放同一条批序，不依赖线程或全局随机源。
#[derive(Debug, Clone, PartialEq)]
pub struct LoaderConfig {
    /// 每批行数；构造期校验非零。
    pub rows: usize,
    /// 列宽（每行条目数）；构造期校验非零且 `rows * cols` 不溢出。
    pub cols: usize,
    /// `Some(seed)` 每轮按种子派生独立流重洗批序；`None` 顺序推进。
    pub shuffle: Option<u64>,
    /// 每批非负权重；`Some` 时按权重有放回抽批；与 `shuffle` 互斥，权重优先。
    pub weights: Option<Vec<f32>>,
}

impl LoaderConfig {
    /// 校验行列非零且乘积不溢出；采样方式默认顺序。
    pub fn new(rows: usize, cols: usize) -> Result<Self> {
        stride(rows, cols, SITE)?;
        Ok(Self {
            rows,
            cols,
            shuffle: None,
            weights: None,
        })
    }
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
        let sampler = Sampler::from_config(batches, config.shuffle, config.weights)?;
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
#[path = "../../tests/samples_tests.rs"]
mod samples_tests;

#[cfg(test)]
#[path = "../../tests/dataset_tests.rs"]
mod dataset_tests;
