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
mod tests {
    use super::*;

    testkit::test_cases! {
    fn a_batch_must_match_its_shape() {
        let tokens = [1, 2, 3, 4];
        let batch = Batch::new(&tokens, 2, 2).expect("2 x 2");
        assert_eq!(batch.tokens(), tokens);
        assert_eq!((batch.rows(), batch.cols()), (2, 2));
        assert_eq!(
            Batch::new(&[1, 2, 3], 2, 2).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(
            Batch::<u32>::new(&[], 0, 2).unwrap_err().code(),
            Code::Invalid
        );
    }

    fn a_batch_must_reject_an_overflowing_shape() {
        let tokens = [1, 2];
        // 溢出必须报错，而不是回绕成一个恰好等于 `tokens.len()` 的形状：后者会让一个非法的
        // 2 x usize::MAX 被当成合法批，形状从此不再可信。
        assert_eq!(
            Batch::new(&tokens, usize::MAX, 2).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(
            Batch::new(&tokens, 2, usize::MAX).unwrap_err().code(),
            Code::Invalid
        );
    }

    fn a_batch_borrows_the_callers_buf() {
        let tokens = [7, 8, 9, 10, 11, 12];
        for (rows, cols) in [(1, 6), (2, 3), (3, 2), (6, 1)] {
            let batch = Batch::new(&tokens, rows, cols).expect("形状合法");
            assert_eq!(
                batch.tokens().as_ptr(),
                tokens.as_ptr(),
                "批必须借用调用方的缓冲而不是拷贝"
            );
            assert_eq!(rows * cols, batch.tokens().len());
        }
    }

    fn a_chunks_stream_rejects_bad_geometry() {
        let tokens = [1, 2, 3, 4];
        for (rows, cols) in [(1, 0), (0, 1), (usize::MAX, 2)] {
            assert_eq!(
                Chunks::new(&tokens, rows, cols).unwrap_err().code(),
                Code::Invalid,
                "rows = {rows}、cols = {cols} 必须在构造期被拒"
            );
        }
    }

    fn both_constructors_read_geometry_in_the_same_order() {
        let tokens: [u32; 6] = [0, 1, 2, 3, 4, 5];
        // `(2, 3)` 在两个构造函数里都必须表示「2 行 3 列」。此前 `Chunks::new` 收的是 `(cols, rows)`，
        // 于是同一对数字换个构造函数就是一次静默转置——而两种顺序都能通过校验，没有一条用例会红。
        let batch = Batch::new(&tokens, 2, 3).expect("2 x 3");
        let mut dataset = Chunks::new(&tokens, 2, 3).expect("2 x 3");
        let first = dataset.next_batch().expect("6 个 token 恰好是一批");
        assert_eq!(
            (first.rows(), first.cols()),
            (batch.rows(), batch.cols()),
            "两个构造函数必须读出同一形状"
        );
        assert_eq!(first.tokens(), batch.tokens());
        assert_eq!(
            (first.rows(), first.cols()),
            (2, 3),
            "顺序反了会得到 (3, 2)"
        );
        assert_eq!(dataset.remaining_batches(), 0);
    }

    fn an_empty_stream_yields_no_batch() {
        let mut dataset = Chunks::<u32>::new(&[], 1, 2).expect("空流也是合法输入");
        assert_eq!(dataset.cols(), 2);
        assert_eq!(dataset.batches(), 0);
        assert_eq!(dataset.remaining_batches(), 0);
        assert!(dataset.next_batch().is_none());
    }

    fn chunks_drop_the_partial_tail() {
        let tokens = [0, 1, 2, 3, 4, 5, 6];
        let mut dataset = Chunks::new(&tokens, 1, 2).expect("数据");
        assert_eq!(dataset.batches(), 3, "7 个 token 按 2 个一行只能凑 3 批");
        assert_eq!(dataset.remaining_batches(), 3);
        for expected in [&tokens[0..2], &tokens[2..4], &tokens[4..6]] {
            assert_eq!(dataset.next_batch().expect("整批").tokens(), expected);
        }
        assert!(
            dataset.next_batch().is_none(),
            "只剩 1 个 token，凑不满一行"
        );
        assert_eq!(dataset.remaining_batches(), 0);
    }

    fn consuming_advances_both_counters() {
        let tokens: [u32; 6] = [0, 1, 2, 3, 4, 5];
        let mut dataset = Chunks::new(&tokens, 1, 2).expect("数据");
        assert_eq!(
            (
                dataset.cols(),
                dataset.batches(),
                dataset.remaining_batches()
            ),
            (2, 3, 3)
        );
        let batch = dataset.next_batch().expect("第一批");
        assert_eq!(batch.tokens(), [0, 1]);
        assert_eq!(dataset.remaining_batches(), 2);
        assert_eq!(
            (
                batch.rows(),
                batch.cols(),
                dataset.batches(),
                dataset.cols()
            ),
            (1, 2, 3, 2),
            "每批几何、批总数、列宽都不随消费推进"
        );
    }

    fn two_batches_can_be_held_at_once() {
        let tokens: [u32; 6] = [0, 1, 2, 3, 4, 5];
        let mut dataset = Chunks::new(&tokens, 1, 2).expect("数据");
        let first = dataset.next_batch().expect("第一批");
        let second = dataset.next_batch().expect("第二批");
        // 两批的寿命来自 token 缓冲而不是数据集，所以可以同时存活且不重叠。
        assert_eq!(first.tokens(), [0, 1]);
        assert_eq!(second.tokens(), [2, 3]);
        assert_eq!(
            first.tokens().as_ptr().wrapping_add(first.tokens().len()),
            second.tokens().as_ptr(),
            "两批必须落在缓冲里相邻但不重叠的位置"
        );
    }

    fn iterating_yields_exactly_the_declared_batches() {
        let tokens: [u32; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut dataset = Chunks::new(&tokens, 1, 3).expect("数据");
        assert_eq!(dataset.len(), 3, "`len()` 就是还没取走的批数");
        assert_eq!(dataset.size_hint(), (3, Some(3)));
        let collected: Vec<_> = dataset.by_ref().collect();
        assert_eq!(collected.len(), 3, "迭代条数必须等于 `batches()`");
        assert_eq!(collected[2].tokens(), [6, 7, 8]);
        assert_eq!(dataset.len(), 0);
        // 取空之后 `len()` 与两种取法都不再变化（`FusedIterator`）。
        assert!(dataset.next().is_none());
        assert!(dataset.next_batch().is_none());
        assert_eq!(dataset.size_hint(), (0, Some(0)));
    }

    fn next_and_next_batch_advance_the_same_cursor() {
        let tokens: [u32; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut dataset = Chunks::new(&tokens, 1, 3).expect("数据");
        assert_eq!(dataset.next().expect("第一条").tokens(), [0, 1, 2]);
        assert_eq!(dataset.next_batch().expect("第二条").tokens(), [3, 4, 5]);
        assert_eq!(
            dataset.by_ref().count(),
            1,
            "两种取法共用同一个游标，不会跳批"
        );
        assert_eq!(dataset.remaining_batches(), 0);
    }

    fn the_trait_object_reports_and_yields_the_same_stream() {
        let tokens: [u32; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut chunks = Chunks::new(&tokens, 1, 3).expect("数据");
        // 走 trait 对象路径：三个计数与取批都由 trait 分发，验证对象安全契约。
        let ds: &mut dyn Dataset<'_, Item = u32> = &mut chunks;
        assert_eq!(ds.cols(), 3);
        assert_eq!(ds.batches(), 3);
        assert_eq!(ds.remaining_batches(), 3);
        for expected in [&tokens[0..3], &tokens[3..6], &tokens[6..9]] {
            assert_eq!(ds.next_batch().expect("整批").tokens(), expected);
        }
        assert_eq!(ds.remaining_batches(), 0);
        assert!(ds.next_batch().is_none(), "取空后 trait 路径同样终止");
    }
    }

    /// 收集一个数据集全部批的 token 序列（每批压成一段）。
    fn drain_tokens<'a>(dataset: &mut dyn Dataset<'a, Item = u32>) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        while let Some(batch) = dataset.next_batch() {
            out.push(batch.tokens().to_vec());
        }
        out
    }

    testkit::test_cases! {
    fn same_shuffle_seed_replays_the_same_order() {
        let tokens: Vec<u32> = (0..8).collect();
        let config = LoaderConfig::new(1, 1).expect("几何合法");
        let seeded = LoaderConfig {
            sampling: Sampling::Shuffle { seed: 7 },
            ..config
        };

        let first = drain_tokens(&mut Loader::with_config(&tokens, seeded.clone()).expect("装配"));
        let second = drain_tokens(&mut Loader::with_config(&tokens, seeded).expect("装配"));

        assert_eq!(first, second, "同种子必须重放出同一条批序");
    }

    fn different_shuffle_seeds_yield_different_orders() {
        let tokens: Vec<u32> = (0..8).collect();
        let make = |seed| {
            let config = LoaderConfig::new(1, 1).expect("几何合法");
            let seeded = LoaderConfig {
                sampling: Sampling::Shuffle { seed },
                ..config
            };
            drain_tokens(&mut Loader::with_config(&tokens, seeded).expect("装配"))
        };

        let a = make(0);
        let b = make(1);
        assert_ne!(a, b, "不同种子的批序应当不同");
    }

    fn shuffle_keeps_every_batch_exactly_once() {
        let tokens: Vec<u32> = (0..8).collect();
        let config = LoaderConfig::new(1, 1).expect("几何合法");
        let seeded = LoaderConfig {
            sampling: Sampling::Shuffle { seed: 3 },
            ..config
        };

        let collected = drain_tokens(&mut Loader::with_config(&tokens, seeded).expect("装配"));
        let mut flat: Vec<u32> = collected.iter().flatten().copied().collect();
        flat.sort_unstable();

        assert_eq!(flat, tokens, "洗牌不得丢批也不得重复批");
        assert_eq!(collected.len(), 8, "批数与顺序流一致");
    }

    fn reset_returns_to_the_first_batch_of_the_round() {
        let tokens: Vec<u32> = (0..6).collect();
        let mut loader = Loader::new(&tokens, 1, 1).expect("顺序流");

        let first_round = drain_tokens(&mut loader);
        loader.reset();
        let again = drain_tokens(&mut loader);

        assert_eq!(first_round, again, "顺序流重置后必须重放同一条序");
    }

    fn two_rounds_yield_twice_the_declared_batches() {
        let tokens: Vec<u32> = (0..6).collect();
        let mut loader = Loader::new(&tokens, 1, 1).expect("顺序流");
        let batches = loader.batches();

        let mut count = 0;
        for _round in 0..2 {
            loader.reset();
            while loader.next_batch().is_some() {
                count += 1;
            }
        }

        assert_eq!(count, 2 * batches, "每轮重置后批数回到声明的总数");
    }

    fn weighted_rejects_illegal_weights_at_construction() {
        let tokens: Vec<u32> = (0..4).collect();
        let cases = [
            (vec![1.0, 2.0], "权重条数与批数不符必须在构造期拒绝"),
            (vec![1.0, -1.0, 1.0, 1.0], "负权重必须拒绝"),
            (vec![f32::NAN; 4], "NaN 权重必须拒绝"),
            (vec![0.0; 4], "权重总质量为正才允许抽样"),
        ];
        for (weights, message) in cases {
            let config = LoaderConfig {
                sampling: Sampling::Weighted { weights, seed: 0 },
                ..LoaderConfig::new(1, 1).expect("几何合法")
            };
            assert_eq!(
                Loader::with_config(&tokens, config).unwrap_err().code(),
                Code::Invalid,
                "{message}"
            );
        }
    }

    fn weighted_draws_one_batch_per_declared_batch() {
        let tokens: Vec<u32> = (0..4).collect();
        let config = LoaderConfig::new(1, 1).expect("几何合法");
        let weighted = LoaderConfig {
            sampling: Sampling::Weighted {
                weights: vec![1.0, 1.0, 1.0, 1.0],
                seed: 0,
            },
            ..config
        };
        let mut loader = Loader::with_config(&tokens, weighted).expect("装配");

        let drawn: Vec<_> = (0..loader.batches())
            .filter_map(|_| loader.next_batch())
            .collect();
        assert_eq!(drawn.len(), 4, "每轮抽出的批数等于声明批数");

        let mut order: Vec<usize> = (0..4).collect();
        let (ptr, capacity) = (order.as_ptr(), order.capacity());
        draw_round(&[1.0; 4], 4.0, 7, 0, &mut order);
        draw_round(&[1.0; 4], 4.0, 7, 1, &mut order);
        assert_eq!(order.len(), 4);
        assert_eq!(
            (order.as_ptr(), order.capacity()),
            (ptr, capacity),
            "跨轮重抽必须复用索引缓冲"
        );
    }

    fn high_weight_batch_is_sampled_more_often() {
        let tokens: Vec<u32> = (0..2).collect();
        let config = LoaderConfig::new(1, 1).expect("几何合法");
        let weighted = LoaderConfig {
            sampling: Sampling::Weighted {
                weights: vec![1.0, 100.0],
                seed: 0,
            },
            ..config
        };
        let mut loader = Loader::with_config(&tokens, weighted).expect("装配");

        let mut counts = [0usize, 0];
        for _round in 0..20 {
            loader.reset();
            while let Some(batch) = loader.next_batch() {
                counts[batch.tokens()[0] as usize] += 1;
            }
        }

        assert!(
            counts[1] > counts[0],
            "高权重批（1）应比低权重批（0）更常被抽到：{counts:?}"
        );
    }

    fn config_rejects_bad_geometry_before_assembly() {
        assert_eq!(LoaderConfig::new(0, 2).unwrap_err().code(), Code::Invalid);
        assert_eq!(LoaderConfig::new(2, 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(
            LoaderConfig::new(usize::MAX, 2).unwrap_err().code(),
            Code::Invalid
        );
    }
    }
}
