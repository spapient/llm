//! 采样序：把「批计数」映射成一条批索引迭代。顺序 / 洗牌 / 加权三族共用一张索引表与一条游标，
//! 不为每种采样各写一套计数与迭代器。
//!
//! 随机只走 [`crate::rng::Rng`]：种子进配置，序列可重建；无线程、无全局随机源。

use diagnostic::{Code, Error, Result};

use crate::rng::Rng;

use super::Sampling;

/// 本模块诊断站点。
const SITE: &str = "train::dataset::sampler";

/// 一条采样序的闭集实现。
#[derive(Debug, Clone)]
pub(crate) enum Sampler {
    /// `0, 1, …, n−1`，不洗牌。
    Sequential { order: Vec<usize>, cursor: usize },
    /// Fisher–Yates 洗牌；每轮按种子派生独立流重洗，序列可由种子重建。
    Shuffled {
        seed: u64,
        epoch: u64,
        order: Vec<usize>,
        cursor: usize,
    },
    /// 轮盘有放回：每轮按权重重抽 `n` 个批索引。
    Weighted {
        weights: Vec<f32>,
        total: f32,
        seed: u64,
        epoch: u64,
        order: Vec<usize>,
        cursor: usize,
    },
}

impl Sampler {
    /// 按互斥配置装配采样序。
    pub(crate) fn from_config(batches: usize, sampling: Sampling) -> Result<Self> {
        match sampling {
            Sampling::Sequential => Ok(Self::sequential(batches)),
            Sampling::Shuffle { seed } => Ok(Self::shuffled(batches, seed)),
            Sampling::Weighted { weights, seed } => Self::weighted(batches, weights, seed),
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
        Self::Shuffled {
            seed,
            epoch: 0,
            order,
            cursor: 0,
        }
    }

    fn weighted(n: usize, weights: Vec<f32>, seed: u64) -> Result<Self> {
        let total = validate_weights(&weights, n)?;
        let order = draw_round(&weights, total, seed, 0, n);
        Ok(Self::Weighted {
            weights,
            total,
            seed,
            epoch: 0,
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
        let idx = order[*cursor];
        *cursor += 1;
        Some(idx)
    }

    /// 进入新一轮：顺序只归游标；洗牌 / 加权按派生流重洗或重抽，游标归位。
    pub(crate) fn reset(&mut self) {
        match self {
            Self::Sequential { cursor, .. } => *cursor = 0,
            Self::Shuffled {
                seed,
                epoch,
                order,
                cursor,
            } => {
                *epoch += 1;
                let mut rng = Rng::with_stream(*seed, *epoch as u32);
                shuffle(&mut rng, order);
                *cursor = 0;
            }
            Self::Weighted {
                weights,
                total,
                seed,
                epoch,
                order,
                cursor,
            } => {
                *epoch += 1;
                *order = draw_round(weights, *total, *seed, *epoch, order.len());
                *cursor = 0;
            }
        }
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
    if total <= 0.0 {
        return Err(Error::new(Code::Invalid, "权重总质量必须为正").with(SITE));
    }
    Ok(total)
}

/// 轮盘有放回抽 `n` 个批索引；规模未标定前用 O(n) 轮盘，别名表按基准再上。
fn draw_round(weights: &[f32], total: f32, seed: u64, epoch: u64, n: usize) -> Vec<usize> {
    let mut rng = Rng::with_stream(seed, epoch as u32);
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
