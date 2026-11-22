//! 双随机残差混合（mHC，§5.8.2）。

use crate::invalid;
use diagnostic::Result;
use op::scan::Scalar;

use crate::csa::{accumulate, finite, sinkhorn, spectral_norm};

/// 报错出处：几何与取值都在构造期失败，不留到混合的中途。
const SITE: &str = "algorithm::mhc";

/// Engram-27B / V4 的残差流数 `M`。
pub const MHC_STREAMS: usize = 4;

/// Sinkhorn 的迭代上限（与 CSA 的块权重同档）。
const ITERS: usize = 32;

/// 双随机判定容差。
const TOL: f64 = 1e-9;

/// 多流残差混合：一张已归一化的 `[M, M]` 矩阵 + 它的实测谱范数。
#[derive(Debug, Clone, PartialEq)]
pub struct Mhc<S: Scalar = f32> {
    streams: usize,
    mix: Vec<S>,
    sigma: S,
}

impl<S: Scalar> Mhc<S> {
    /// 由一张待归一的 `[streams, streams]` 矩阵构造：Sinkhorn 归一后校验双随机，并实测谱范数。
    ///
    /// # Errors
    ///
    /// `streams` 为零或其平方溢出、矩阵长度不是 `streams²`、任一元素非有限或为负、行或列全零
    /// （行/列和为零时归一化原地不动，永远到不了双随机）、归一后仍非双随机——都返回
    /// `Code::Invalid`。
    pub fn new(mix: &[S], streams: usize) -> Result<Self> {
        let size = size(streams)?;
        if mix.len() != size {
            return Err(invalid(SITE, format!(
                    "混合矩阵必须是 [{streams}, {streams}] = {size} 个元素，得到 {}",
                    mix.len()
                )));
        }
        check_entries(mix)?;
        let mut normalized = mix.to_vec();
        if !sinkhorn(&mut normalized, streams, ITERS, S::from_f64(TOL)) {
            return Err(invalid(
                SITE,
                "Sinkhorn 归一后仍不是双随机：矩阵里有行和或列和为零（归一时原地不动），这张矩阵没有分布可归一",
            ));
        }
        let sigma = spectral_norm(&normalized, streams, ITERS);
        Ok(Self {
            streams,
            mix: normalized,
            sigma,
        })
    }

    /// 单位混合：`M` 条流各自直通，是"mHC 未启用"的等价物，也是装配侧的默认起点。
    ///
    /// # Errors
    ///
    /// `streams` 为零或其平方溢出时返回 `Code::Invalid`。
    pub fn identity(streams: usize) -> Result<Self> {
        let mut mix = vec![S::ZERO; size(streams)?];
        for i in 0..streams {
            mix[i * streams + i] = S::ONE;
        }
        Self::new(&mix, streams)
    }

    /// 流数 `M`。
    pub fn streams(&self) -> usize {
        self.streams
    }

    /// 归一后的混合矩阵，行主序 `[M, M]`（行和、列和均 ≈ 1）。
    pub fn mix(&self) -> &[S] {
        &self.mix
    }

    /// 实测谱范数 `‖B‖₂`：对非负双随机阵，理论值 `≤ 1`。
    pub fn sigma(&self) -> S {
        self.sigma
    }

    /// 一次层间混合：`M` 条流进、`M` 条流出，第 `r` 条输出是输入各流的凸组合。
    ///
    /// # Errors
    ///
    /// 流的条数不是 `M`、任一条宽度为零、或各条宽度不一致——都返回 `Code::Invalid`。宽度不一致时
    /// 逐元素相加会把不同位置的数拼在一起，通道错位却没有一处报错。
    pub fn mix_streams(&self, streams: &[Vec<S>]) -> Result<Vec<Vec<S>>> {
        let width = self.check_streams(streams)?;
        Ok((0..self.streams)
            .map(|r| self.blend(r, streams, width))
            .collect())
    }

    /// 入口校验：流条数与各条宽度一致，返回公共宽度。
    fn check_streams(&self, streams: &[Vec<S>]) -> Result<usize> {
        if streams.len() != self.streams {
            return Err(invalid(SITE, format!("必须是 {} 条流，得到 {} 条", self.streams, streams.len())));
        }
        let width = streams.first().map_or(0, Vec::len);
        if width == 0 {
            return Err(invalid(SITE, "流宽度不能为零"));
        }
        if let Some(bad) = streams.iter().position(|row| row.len() != width) {
            return Err(invalid(SITE, format!(
                    "第 {bad} 条流宽度是 {}，与第 0 条的 {width} 不一致",
                    streams[bad].len()
                )));
        }
        Ok(width)
    }

    /// 第 `r` 条输出流：`Σ_c B[r][c] · x_c`。
    fn blend(&self, r: usize, streams: &[Vec<S>], width: usize) -> Vec<S> {
        let mut out = vec![S::ZERO; width];
        for (c, row) in streams.iter().enumerate() {
            accumulate(&mut out, row, self.mix[r * self.streams + c]);
        }
        out
    }
}

/// `M × M` 的元素数，零与溢出都拒绝。
fn size(streams: usize) -> Result<usize> {
    match streams.checked_mul(streams) {
        Some(0) | None => Err(invalid(SITE, format!("流数必须非零且平方不溢出，得到 {streams}"))),
        Some(size) => Ok(size),
    }
}

/// 每个元素都必须有限且非负。
fn check_entries<S: Scalar>(mix: &[S]) -> Result<()> {
    let Some(bad) = mix.iter().position(|&v| !finite(v) || v < S::ZERO) else {
        return Ok(());
    };
    Err(invalid(SITE, format!(
            "混合矩阵第 {bad} 项是 {:?}：必须是有限非负数——负号不会被 Sinkhorn 消掉，而带负号的矩阵即使双随机也可能把信号放大",
            mix[bad]
        )))
}

#[cfg(test)]
#[path = "../tests/mhc_tests.rs"]
mod mhc_tests;
