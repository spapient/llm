//! byte-latent 的表示侧与解码侧（§5.17，第 11 款标杆的另外两层）。

use std::ops::Range;

use diagnostic::{Code, Error, Result};
use op::scan::Scalar;

use crate::csa::{accumulate, dot, softmax_row};
use crate::patch::Patches;
use nn::Table;

/// 报错出处：几何与宽度的问题都在入口失败，不留到池化或投影的中途。
const SITE: &str = "algorithm::codec";

/// 哈希 N-gram 的阶闭集：BLT 取 `n = 3..=8`（§5.17）。
pub const ORDERS: [usize; 6] = [3, 4, 5, 6, 7, 8];

/// 最宽窗口的字节数：阶的上界，从 [`ORDERS`] 派生而不是另写一个字面量。
const WIDEST: usize = ORDERS[ORDERS.len() - 1];

/// 逐字节解码的通道数：一个字节的 256 种取值，通道下标就是字节值。
pub const BYTES: usize = 256;

/// 第 `i` 个字节处的六个窗口，从窄到宽：`bytes[..=i]` 的后缀，长度 `min(n, i + 1)`。
fn windows(bytes: &[u8], i: usize) -> [&[u8]; ORDERS.len()] {
    ORDERS.map(|order| &bytes[(i + 1).saturating_sub(order)..=i])
}

/// local encoder 的表示侧：N-gram 桶嵌入 + patch 内注意力池化。
#[derive(Debug, Clone)]
pub struct ByteEncoder<S: Scalar = f32> {
    table: Table<S>,
    seed: u64,
}

impl<S: Scalar> ByteEncoder<S> {
    /// 由桶表与哈希种子构造。表长已由 [`Table`] 提升到素数，嵌入维也由它在构造期挡过零。
    pub fn new(table: Table<S>, seed: u64) -> Self {
        Self { table, seed }
    }

    /// 桶数（`Table` 的素数表长）。
    pub fn buckets(&self) -> usize {
        self.table.len()
    }

    /// 表示宽度：每个 patch 一行。
    pub fn dim(&self) -> usize {
        self.table.dim()
    }

    /// 哈希种子。
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// 桶表。
    pub fn table(&self) -> &Table<S> {
        &self.table
    }

    /// 每个 patch 一行表示：`patches.len() × dim` 个元素，按 patch 顺序拼接。
    ///
    /// 行数就是全局 transformer 的遍数，这是 byte-latent 的免费缩放轴：patch 平均越长，全局模型在前向
    /// 里跑得越少（[`Patches::mean_len`] 与 [`Patches::len`] 一起给出这个比值）。
    ///
    /// # Errors
    ///
    /// 切分覆盖的字节数不等于 `bytes.len()` 时返回 `Code::Invalid`：切分来自另一次输入，拿它给这一次的
    /// 字节分行只会把两个不相干的内容拼在一起（越界的区间还会在切片处炸，报错离原因很远）。
    pub fn encode(&self, bytes: &[u8], patches: &Patches) -> Result<Vec<S>> {
        if patches.bytes() != bytes.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "切分覆盖 {} 字节，输入是 {}：切分不是这次输入的",
                    patches.bytes(),
                    bytes.len()
                ),
            )
            .with(SITE));
        }
        let mut out = Vec::with_capacity(patches.len() * self.dim());
        for range in patches.ranges() {
            out.extend_from_slice(&self.patch(bytes, range)?);
        }
        Ok(out)
    }

    /// 单个 patch 的表示：`dim` 个元素，`range` 是它的字节区间（半开）。
    ///
    /// [`ByteEncoder::encode`] 对整条切分就是逐 patch 调这个函数再接起来，于是"一个 patch 一行"只有
    /// 一处实现；逐字节生成时也只算**刚收口的那一个**（[`crate::Emitter`]），收口过的不重算。
    ///
    /// # Errors
    ///
    /// 区间为空或越出 `bytes` 时返回 `Code::Invalid`：空 patch 没有边界字节，寻址无从取；越界的区间
    /// 会在切片处炸，报错离原因很远。
    pub fn patch(&self, bytes: &[u8], range: Range<usize>) -> Result<Vec<S>> {
        if range.start >= range.end || range.end > bytes.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "patch 区间 {range:?} 不在 {} 个字节内：patch 是非空字节区间",
                    bytes.len()
                ),
            )
            .with(SITE));
        }
        let dim = self.dim();
        let mut rows = vec![S::ZERO; range.len() * dim];
        for (offset, row) in rows.chunks_exact_mut(dim).enumerate() {
            self.vector(bytes, range.start + offset, row);
        }
        let mut out = vec![S::ZERO; dim];
        self.pool(&rows, &mut out);
        Ok(out)
    }

    /// 第 `index` 个字节的 N-gram 向量：逐字节生成时的**本地上下文**。
    ///
    /// 取的是池化用的同一份逐字节向量（同一套哈希、同一组阶、同一张表），不因为"要走的是残差"
    /// 再算一套；宽度与 patch 表示相同，于是两者可以直接相加。
    ///
    /// # Errors
    ///
    /// `index` 越出 `bytes` 时返回 `Code::Invalid`：越界下标取窗口会在切片处炸。
    pub fn glyph(&self, bytes: &[u8], index: usize) -> Result<Vec<S>> {
        if index >= bytes.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "第 {idx} 个字节不在 {} 个字节内：本地上下文只取已有字节",
                    bytes.len()
                ),
            )
            .with(SITE));
        }
        let mut out = vec![S::ZERO; self.dim()];
        self.vector(bytes, index, &mut out);
        Ok(out)
    }

    /// 第 `i` 个字节的 N-gram 向量：六阶窗口的桶向量之和除以阶数。
    fn vector(&self, bytes: &[u8], i: usize, out: &mut [S]) {
        out.fill(S::ZERO);
        let mut gram = [0_u32; WIDEST];
        for (order, window) in ORDERS.into_iter().zip(windows(bytes, i)) {
            let gram = &mut gram[..window.len()];
            for (slot, &byte) in gram.iter_mut().zip(window) {
                *slot = u32::from(byte);
            }
            let row = self.table.get(gram, self.seed ^ order as u64);
            for (acc, &value) in out.iter_mut().zip(row) {
                *acc = acc.add(value);
            }
        }
        let orders = S::from_f64(ORDERS.len() as f64);
        for value in out.iter_mut() {
            *value = value.div(orders);
        }
    }

    /// 一个 patch 的表示：以边界字节的向量为查询，对本 patch 内每个字节的向量做注意力加权求和。
    fn pool(&self, vectors: &[S], out: &mut [S]) {
        let dim = self.dim();
        let scale = S::ONE.div(S::from_f64((dim as f64).sqrt()));
        let boundary = &vectors[(vectors.len() / dim - 1) * dim..];
        let mut weights: Vec<S> = vectors
            .chunks_exact(dim)
            .map(|row| dot(boundary, row).mul(scale))
            .collect();
        softmax_row(&mut weights);
        out.fill(S::ZERO);
        for (row, &weight) in vectors.chunks_exact(dim).zip(&weights) {
            accumulate(out, row, weight);
        }
    }
}

/// local decoder 的输出头：patch 表示 → 256-way 字节分布。
#[derive(Debug, Clone, PartialEq)]
pub struct ByteDecoder<S: Scalar = f32> {
    width: usize,
    projection: Vec<S>,
}

impl<S: Scalar> ByteDecoder<S> {
    /// 由表示宽度与投影构造。
    ///
    /// # Errors
    ///
    /// 宽度为零（没有可读的表示维）、投影长度不是 `BYTES × width`、或这个乘积溢出——都返回
    /// `Code::Invalid`。行数或列数错了的话每个通道整体错位，输出的每个数都错却不会有任何一处报错。
    pub fn new(width: usize, projection: &[S]) -> Result<Self> {
        let expected = width.checked_mul(BYTES).ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("BYTES × width 溢出（{BYTES} × {width}）"),
            )
            .with(SITE)
        })?;
        if width == 0 || projection.len() != expected {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "表示宽度必须非零、投影长度必须是 BYTES × width = {expected}，得到宽度 {width}、{} 个元素",
                    projection.len()
                ),
            )
            .with(SITE));
        }
        Ok(Self {
            width,
            projection: projection.to_vec(),
        })
    }

    /// 表示宽度（与 [`ByteEncoder::dim`] 是同一个量）。
    pub fn width(&self) -> usize {
        self.width
    }

    /// 投影，行主序 `[BYTES, width]`。
    pub fn projection(&self) -> &[S] {
        &self.projection
    }

    /// 256 个字节通道的分数。
    ///
    /// # Errors
    ///
    /// 表示宽度不是本层的 `width` 时返回 `Code::Invalid`：宽了是把别的层的数据喂进来，窄了会静默
    /// 少读几维（内积照算不误，结果却是另一个数）。
    pub fn logits(&self, state: &[S]) -> Result<Vec<S>> {
        self.check(state)?;
        Ok(self
            .projection
            .chunks_exact(self.width)
            .map(|row| dot(row, state))
            .collect())
    }

    /// 256-way 分布：分数过一遍 softmax。
    ///
    /// # Errors
    ///
    /// 同 [`ByteDecoder::logits`]。
    pub fn probs(&self, state: &[S]) -> Result<Vec<S>> {
        let mut probs = self.logits(state)?;
        softmax_row(&mut probs);
        Ok(probs)
    }

    /// 贪心字节：分数最大的那个通道，并列时取**最小的字节值**。
    ///
    /// 取分数而不是分布：softmax 单调，argmax 相同，而分数少一次指数运算。并列时不许"随便挑一个"
    /// ——那会让同一份表示在不同运行里给出不同字节，确定性也就没了；`>` 严格比较配上下标顺序，
    /// 第一个最大值留下，就是最小字节值。
    ///
    /// # Errors
    ///
    /// 同 [`ByteDecoder::logits`]。
    pub fn byte(&self, state: &[S]) -> Result<u8> {
        self.pick(state).map(|(byte, _)| byte)
    }

    /// 贪心字节与该字节的**概率**：逐字节生成时一步要走的东西。
    ///
    /// 一次内积同时给出两样：[`ByteDecoder::byte`] 取的就是这里的第一项（argmax 在 softmax 之前
    /// 算，所以两个入口逐位同一），第二项是该字节的 `p`——它的自信息 `−ln p` 就是切分规则的分数
    /// （[`crate::Emitter`]）。于是"生成用哪条分布"与"边界用哪个数"只有一份来源。
    ///
    /// 贪心保证 `p ≥ 1/256`：选中的恒是最大通道，所以 `−ln p ≤ ln 256`，自信息不会溢出成无穷。
    ///
    /// # Errors
    ///
    /// 同 [`ByteDecoder::logits`]。
    pub fn pick(&self, state: &[S]) -> Result<(u8, S)> {
        let mut probs = self.logits(state)?;
        let best = argmax(&probs);
        softmax_row(&mut probs);
        Ok((best, probs[usize::from(best)]))
    }

    /// 入口校验：表示的宽度。
    fn check(&self, state: &[S]) -> Result<()> {
        if state.len() == self.width {
            return Ok(());
        }
        Err(Error::new(
            Code::Invalid,
            format!("表示宽度必须是 {}，得到 {}", self.width, state.len()),
        )
        .with(SITE))
    }
}

/// 通道下标里的最大值：并列时取最小字节值，所以严格大于才算新最优。
fn argmax<S: Scalar>(scores: &[S]) -> u8 {
    let best = scores.iter().enumerate().fold(
        0,
        |best, (b, &score)| {
            if score > scores[best] { b } else { best }
        },
    );
    best as u8
}

#[cfg(test)]
#[path = "../tests/codec_tests.rs"]
mod tests;
