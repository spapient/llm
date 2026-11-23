//! 采样序：把「批计数」映射成一条批索引迭代。顺序 / 洗牌 / 加权三族共用一张索引表与一条游标，
//! 不为每种采样各写一套计数与迭代器。
//!
//! 随机只走 [`crate::rng::Rng`]：种子进配置，序列可重建；无线程、无全局随机源。

use diagnostic::{Code, Error, Result};

use crate::rng::Rng;

/// 本模块诊断站点。
const SITE: &str = "train::dataset::sampler";

/// 一条采样序的闭集实现。
#[derive(Debug, Clone)]
pub(crate) enum Sampler {
    /// `0, 1, …, n−1`，不洗牌。
    Sequential { order: Vec<usize>, cursor: usize },
    /// Fisher–Yates 洗牌；构造时按种子洗定一次，序列可由种子重建。
    Shuffled {
        order: Vec<usize>,
        cursor: usize,
    },
    /// 轮盘有放回：构造时按权重抽 `n` 个批索引。
    Weighted {
        weights: Vec<f32>,
        total: f32,
        order: Vec<usize>,
        cursor: usize,
    },
}

impl Sampler {
    /// 按配置装配采样序：权重优先于洗牌；两者都缺则顺序。
    pub(crate) fn from_config(
        batches: usize,
        shuffle: Option<u64>,
        weights: Option<Vec<f32>>,
    ) -> Result<Self> {
        match (weights, shuffle) {
            (Some(weights), _) => Self::weighted(batches, weights, shuffle.unwrap_or(0)),
            (None, Some(seed)) => Ok(Self::shuffled(batches, seed)),
            (None, None) => Ok(Self::sequential(batches)),
        }
    }

    fn sequential(n: usize) -> Self {
        Self::Sequential {
            order: (0..n).collect(),
            cursor: 0,
        }
    }

    fn shuffled(n: usize, seed: u64) -> Self {
        let mut order: Vec<usize> = (0..n).collect();
        let mut rng = Rng::with_stream(seed, 0);
        shuffle(&mut rng, &mut order);
        Self::Shuffled { order, cursor: 0 }
    }

    fn weighted(n: usize, weights: Vec<f32>, seed: u64) -> Result<Self> {
        let total = validate_weights(&weights, n)?;
        let order = draw_round(&weights, total, seed, n);
        Ok(Self::Weighted {
            weights,
            total,
            order,
            cursor: 0,
        })
    }

    /// 还没取走的批数。
    pub(crate) fn remaining(&self) -> usize {
        let cursor = match self {
            Self::Sequential { cursor, .. } => cursor,
            Self::Shuffled { cursor, .. } => cursor,
            Self::Weighted { cursor, .. } => cursor,
        };
        self.order().len() - cursor
    }

    /// 取下一个批索引；`None` 当且仅当本轮取空。
    pub(crate) fn next(&mut self) -> Option<usize> {
        let (order, cursor) = match self {
            Self::Sequential { order, cursor } => (order, cursor),
            Self::Shuffled { order, cursor, .. } => (order, cursor),
            Self::Weighted { order, cursor, .. } => (order, cursor),
        };
        if *cursor == order.len() {
            return None;
        }
        let index = order[*cursor];
        *cursor += 1;
        Some(index)
    }

    /// 回到本轮开头：游标归位，重放构造时排好的同一条序；不在此重洗或重抽。
    pub(crate) fn reset(&mut self) {
        let cursor = match self {
            Self::Sequential { cursor, .. } => cursor,
            Self::Shuffled { cursor, .. } => cursor,
            Self::Weighted { cursor, .. } => cursor,
        };
        *cursor = 0;
    }

    fn order(&self) -> &[usize] {
        match self {
            Self::Sequential { order, .. } => order,
            Self::Shuffled { order, .. } => order,
            Self::Weighted { order, .. } => order,
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
    if weights.len() != n {
        return Err(Error::new(
            Code::Invalid,
            format!("权重条数 {} 与批数 {n} 不符", weights.len()),
        )
        .with(SITE));
    }
    let mut total = 0.0;
    for &w in weights {
        if !w.is_finite() || w < 0.0 {
            return Err(Error::new(Code::Invalid, "权重必须是有限非负数").with(SITE));
        }
        total += w;
    }
    if total <= 0.0 {
        return Err(Error::new(Code::Invalid, "权重总质量必须为正").with(SITE));
    }
    Ok(total)
}

/// 轮盘有放回抽 `n` 个批索引；规模未标定前用 O(n) 轮盘，别名表按基准再上。
fn draw_round(weights: &[f32], total: f32, seed: u64, n: usize) -> Vec<usize> {
    let mut rng = Rng::with_stream(seed, 0);
    (0..n).map(|_| draw(&mut rng, weights, total)).collect()
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
