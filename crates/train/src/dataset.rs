//! 训练数据集管线：源（条目流）→ 采样序（采样序）→ 批装配（[`Loader`]）三层分离。
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

use diagnostic::{Code, Error, Result};

use crate::rng::Rng;

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

/// 批采样方式：三种策略由类型保证互斥，随机流只由显式种子派生。
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Sampling {
    /// 按条目原顺序取完整批。
    #[default]
    Sequential,
    /// 每轮按种子派生独立流重洗批序。
    Shuffle {
        /// 重洗随机流种子。
        seed: u64,
    },
    /// 按每批的非负权重有放回抽样。
    Weighted {
        /// 每批抽样权重（条数等于批数）。
        weights: Vec<f32>,
        /// 抽样随机流种子。
        seed: u64,
    },
}

/// 批装配配置：几何与采样方式一次性收口到构造期，热路径不再做合法性分支检查。
#[derive(Debug, Clone, PartialEq)]
pub struct LoaderConfig {
    /// 每批行数；构造期校验非零。
    pub rows: usize,
    /// 列宽（每行条目数）；构造期校验非零且 `rows * cols` 不溢出。
    pub cols: usize,
    /// 批采样方式。
    pub sampling: Sampling,
}

impl LoaderConfig {
    /// 校验行列非零且乘积不溢出；采样方式默认顺序。
    pub fn new(rows: usize, cols: usize) -> Result<Self> {
        let config = Self {
            rows,
            cols,
            sampling: Sampling::Sequential,
        };
        config.validate()?;
        Ok(config)
    }

    /// 校验批几何；公开字段被直接修改后也可在装配边界重新检查。
    pub fn validate(&self) -> Result<()> {
        stride(self.rows, self.cols, SITE).map(|_| ())
    }

    /// 设置互斥的采样策略。
    #[must_use]
    pub fn with_sampling(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            rows: 1,
            cols: 1,
            sampling: Sampling::Sequential,
        }
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

/// 策略状态只保存差异；索引表与游标在三种采样间共享一份实现。
#[derive(Debug, Clone)]
enum Policy {
    Sequential,
    Shuffled {
        seed: u64,
        epoch: u64,
    },
    Weighted {
        weights: Vec<f32>,
        total: f32,
        seed: u64,
        epoch: u64,
    },
}

/// 一条采样序：统一的索引表和游标，加上决定下轮如何重建索引表的策略。
#[derive(Debug, Clone)]
pub(crate) struct Sampler {
    policy: Policy,
    order: Vec<usize>,
    cursor: usize,
}

impl Sampler {
    /// 按互斥配置装配采样序。
    pub(crate) fn from_config(n: usize, sampling: Sampling) -> Result<Self> {
        let order: Vec<_> = (0..n).collect();
        match sampling {
            Sampling::Sequential => Ok(Self {
                policy: Policy::Sequential,
                order,
                cursor: 0,
            }),
            Sampling::Shuffle { seed } => {
                let mut sampler = Self {
                    policy: Policy::Shuffled { seed, epoch: 0 },
                    order,
                    cursor: 0,
                };
                let mut rng = Rng::with_stream(seed, 0);
                shuffle(&mut rng, &mut sampler.order);
                Ok(sampler)
            }
            Sampling::Weighted { weights, seed } => {
                let total = validate_weights(&weights, n)?;
                let mut sampler = Self {
                    policy: Policy::Weighted {
                        weights,
                        total,
                        seed,
                        epoch: 0,
                    },
                    order,
                    cursor: 0,
                };
                sampler.redraw(0);
                Ok(sampler)
            }
        }
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

    /// 进入新一轮：只有策略状态不同，游标归位与索引缓冲复用完全共用。
    pub(crate) fn reset(&mut self) {
        let redraw = match &mut self.policy {
            Policy::Sequential => None,
            Policy::Shuffled { seed, epoch } => {
                *epoch += 1;
                shuffle(&mut Rng::with_stream(*seed, *epoch as u32), &mut self.order);
                None
            }
            Policy::Weighted { epoch, .. } => Some(*epoch + 1),
        };
        if let Some(epoch) = redraw {
            self.redraw(epoch);
        }
        self.cursor = 0;
    }

    /// 按派生流原位重抽，保留索引表容量。
    fn redraw(&mut self, epoch: u64) {
        let Policy::Weighted {
            weights,
            total,
            seed,
            epoch: current,
        } = &mut self.policy
        else {
            return;
        };
        *current = epoch;
        draw_round(weights, *total, *seed, epoch, &mut self.order);
    }
}

/// Fisher–Yates 洗牌：从末位向前，每位与 `[0, i]` 内随机位交换。
fn shuffle(rng: &mut Rng, order: &mut [usize]) {
    for i in (1..order.len()).rev() {
        let j = (rng.next_u32() as usize) % (i + 1);
        order.swap(i, j);
    }
}

/// 单条权重合法：有限且非负。
fn weight_is_valid(w: f32) -> bool {
    w.is_finite() && w >= 0.0
}

/// 校验权重：长度对齐、有限、非负、总质量为正；返回总权重。
fn validate_weights(weights: &[f32], n: usize) -> Result<f32> {
    if weights.len() != n {
        return Err(Error::new(
            Code::Invalid,
            format!("权重条数 {} 与批数 {n} 不符", weights.len()),
        )
        .with(SITE));
    }
    if !weights.iter().all(|&w| weight_is_valid(w)) {
        return Err(Error::new(Code::Invalid, "权重必须是有限非负数").with(SITE));
    }
    let total: f32 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::new(Code::Invalid, "权重总质量必须是有限正数").with(SITE));
    }
    Ok(total)
}

/// 轮盘有放回原位重抽；保留调用方索引缓冲的容量。
fn draw_round(weights: &[f32], total: f32, seed: u64, epoch: u64, out: &mut Vec<usize>) {
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

#[cfg(test)]
#[path = "../tests/samples_tests.rs"]
mod tests;
