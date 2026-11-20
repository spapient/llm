//! byte-latent 的表示侧与解码侧（§5.17，第 11 款标杆的另外两层）。
//!
//! [`crate::patch`] 只回答第一个问题——"哪些字节该被同一次全局前向一起看"。一次 byte-latent 前向是
//! 三段：**字节 → 动态 patch → latent transformer → 逐字节解码**；切分之外的两段里与字节直接打交道的
//! 那两层在本模块：
//!
//! - [`ByteEncoder`]：local encoder 的表示侧。每个字节先取 `n = 3..=8` 的哈希 N-gram 嵌入（阶闭集
//!   [`ORDERS`]，桶表复用 [`Table`]，哈希与查表不另写一份），再在 patch 内做注意力池化
//!   ——查询取**边界字节**（它就是这条切分的决策点），键值取本 patch 每个字节的 N-gram 向量。
//! - [`ByteDecoder`]：local decoder 的输出头。一个 patch 表示乘一次 `[BYTES, dim]` 投影，给出
//!   256-way 分布（[`ByteDecoder::probs`]）与贪心字节（[`ByteDecoder::byte`]）。
//!
//! 中间那层全局 transformer 不在这里：它就是在 patch 序列上跑的 [`crate::Attention`]，**每 patch 一次**
//! 而不是每字节一次，遍数就是 [`crate::Patches::len`]。
//!
//! 两处与论文口径的差别写在这里，免得读者以为漏了：
//!
//! 1. N-gram 窗口取**已有后缀**，不补零也不补 BOS 符号：补零会让首字节的表示与"前面真有一个 0 字节"
//!    混为一谈（与 [`crate::Bypass`]"历史不足就不加、不拿零向量补位"同一条判据）。窗口是回看的，所以
//!    一个字节的影响是前向的：它进的是自己与之后至多 `WIDEST − 1` 个字节的窗口。
//! 2. local decoder 的其余层（逆向 cross-attention、6–9 层）及全局 transformer 的装配属于装配侧；
//!    逐字节的自回归循环由 [`crate::Emitter`] 串起，它们共用这里的两层。这一层只做"一个 patch 表示
//!    → 一条 256-way 分布"；权重由调用方给，与 [`crate::Mla`] 同一种分工：本模块不持训练状态，也不给
//!    写侧（训练 / 蒸馏）留接口。
//!
//! 纯算术统一走 [`op::scan::Scalar`] 泛型：`f32` 供前向、`f64` 供对拍，不因精度复制两份实现。

use std::ops::Range;

use diagnostic::{Code, Error, Result};
use op::scan::Scalar;

use crate::csa::{accumulate, dot, softmax_row};
use crate::engram::Table;
use crate::patch::Patches;

/// 报错出处：几何与宽度的问题都在入口失败，不留到池化或投影的中途。
const SITE: &str = "nn::codec";

/// 哈希 N-gram 的阶闭集：BLT 取 `n = 3..=8`（§5.17）。
pub const ORDERS: [usize; 6] = [3, 4, 5, 6, 7, 8];

/// 最宽窗口的字节数：阶的上界，从 [`ORDERS`] 派生而不是另写一个字面量。
const WIDEST: usize = ORDERS[ORDERS.len() - 1];

/// 逐字节解码的通道数：一个字节的 256 种取值，通道下标就是字节值。
pub const BYTES: usize = 256;

/// 第 `i` 个字节处的六个窗口，从窄到宽：`bytes[..=i]` 的后缀，长度 `min(n, i + 1)`。
///
/// 历史不足 `n` 时就给那一小段，不补零：补零让"首字节"与"前面真有个 0 字节"变成同一份输入。
fn windows(bytes: &[u8], i: usize) -> [&[u8]; ORDERS.len()] {
    ORDERS.map(|order| &bytes[(i + 1).saturating_sub(order)..=i])
}

/// local encoder 的表示侧：N-gram 桶嵌入 + patch 内注意力池化。
///
/// 表与种子都由调用方给，本层只读：同一句字节在同一张表上永远给出同一份表示——"表示"是确定的，
/// 不是随机的。桶表、哈希与查表沿用 [`Table`] 那套（表长取素数、`O(1)` 槽寻址）。
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
                    "第 {index} 个字节不在 {} 个字节内：本地上下文只取已有字节",
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
    ///
    /// 每阶用 `seed ^ 阶` 当哈希种子，六阶因此不会挤进同一批槽；窗口取 `bytes[..=i]` 的后缀
    /// （[`windows`]），所以这一行只依赖 `0..=i`。
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
    ///
    /// 查询取最后一个字节，而不是外部的可学习查询向量：边界字节就是这条切分的决策点
    /// （[`crate::Boundary`] 在它上面判出"断"），于是表示仍然只用过去，并且不需要再引入一份权重。
    /// 键值只取本 patch 内的字节，所以"一个 patch 一行"不是约定而是寻址本身。
    ///
    /// `vectors` 是本 patch 的逐字节向量（下标相对区间起点，行数即 patch 长度）：区间本身不再作为
    /// 参数，也就不存在"向量与区间指向不同一段字节"这种错位。
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
///
/// 投影按 `[BYTES, dim]` 行主序存放：第 `b` 行是字节 `b` 的输出行，于是"分布的第几个通道"与
/// "哪个字节取值"是同一个下标，不需要额外的码本把通道映射回字节。
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
///
/// 分数与分数过 softmax 后的分布共用这一份：softmax 单调，两个序列上的 argmax 因此是同一个字节，
/// 不因换算式而变。
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
mod tests {
    use super::*;
    use crate::Boundary;

    /// 一张每槽都不同的表：池化与局部性要能区分不同输入，全零表会让它们永远相等。
    fn filled<S: Scalar>(len: usize, dim: usize) -> Table<S> {
        let mut table = Table::new(len, dim);
        for k in 0..table.len() {
            let row: Vec<S> = (0..dim)
                .map(|d| S::from_f64(((k * dim + d) % 7) as f64 - 3.0))
                .collect();
            table.set(&[k as u32], 0, &row);
        }
        table
    }

    /// 测用的编码器：每槽都不同、种子固定。
    fn encoder(len: usize, dim: usize, seed: u64) -> ByteEncoder<f32> {
        ByteEncoder::new(filled(len, dim), seed)
    }

    /// 只在这些下标上给高分：一条可复现的切分（`rise` 放到极大，只留全局阈值）。
    fn scores(len: usize, boundaries: &[usize]) -> Vec<f32> {
        (0..len)
            .map(|i| if boundaries.contains(&i) { 6.0 } else { 0.1 })
            .collect()
    }

    /// 按 [`scores`] 的分数切分。
    fn split(scores: &[f32]) -> Patches {
        Boundary::<f32>::Entropy {
            global: 5.0,
            rise: 1e9,
        }
        .split(scores)
        .expect("参数与分数都合法")
    }

    /// 窗口只取已有后缀：首字节六个窗口都是它自己，历史够长之后长度封顶在阶上。
    #[test]
    fn a_window_is_the_available_suffix() {
        let bytes = *b"abcdefghi";
        let first = windows(&bytes, 0);
        assert!(
            first.iter().all(|w| w.len() == 1),
            "首字节没有历史：六个窗口都只有它自己"
        );
        assert!(first.iter().all(|w| w.iter().all(|&b| b == b'a')));
        assert!(windows(&bytes, 2).iter().all(|w| w.len() == 3));
        let heights: Vec<usize> = windows(&bytes, 5).iter().map(|w| w.len()).collect();
        assert_eq!(heights, vec![3, 4, 5, 6, 6, 6], "阶 6 起被已有长度截住");
        for i in 0..bytes.len() {
            for window in windows(&bytes, i) {
                assert_eq!(window.last(), Some(&bytes[i]), "窗口只回看：每段都以它结尾");
            }
        }
    }

    /// 每个 patch 一行，行数就是全局 transformer 的遍数；空输入没有行。
    #[test]
    fn every_patch_gets_one_row() {
        let encoder = encoder(13, 4, 3);
        let bytes: Vec<u8> = (0..32_u8).map(|b| b.wrapping_add(64)).collect();
        let patches = split(&scores(bytes.len(), &[0, 8, 20]));
        let rows = encoder.encode(&bytes, &patches).expect("切分与输入同源");
        assert_eq!(rows.len(), patches.len() * encoder.dim());
        assert_eq!(patches.len(), 3);
        assert!(
            patches.len() < bytes.len(),
            "全局只跑 {} 遍，不是 {} 遍",
            patches.len(),
            bytes.len()
        );
        assert!(encoder.encode(&[], &split(&[])).expect("空输入").is_empty());
    }

    /// 前缀稳定：追加字节不改变已完成 patch 的表示——表示只看过去。
    #[test]
    fn a_prefix_leaves_finished_patches_alone() {
        let encoder = encoder(13, 4, 5);
        let bytes: Vec<u8> = (0..24_u8).map(|b| b.wrapping_mul(3)).collect();
        let full = split(&scores(bytes.len(), &[0, 8, 16]));
        let reference = encoder.encode(&bytes, &full).expect("同源");
        for cut in 0..=bytes.len() {
            let prefix = split(&scores(cut, &[0, 8, 16]));
            let rows = encoder.encode(&bytes[..cut], &prefix).expect("同源");
            compare_finished(&full, &reference, &rows, cut, encoder.dim());
        }
    }

    /// 在 `cut` 处收尾时，完整落在前缀里的那些 patch 必须逐位不变。
    fn compare_finished(full: &Patches, reference: &[f32], rows: &[f32], cut: usize, dim: usize) {
        for (p, range) in full.ranges().enumerate() {
            if range.end > cut {
                continue;
            }
            let want = &reference[p * dim..(p + 1) * dim];
            let got = &rows[p * dim..(p + 1) * dim];
            assert_eq!(got, want, "截到 {cut} 时第 {p} 个 patch 的表示变了");
        }
    }

    /// 局部性：窗口是回看的，一个字节只改写自己与之后至多 `WIDEST − 1` 个字节的向量。
    #[test]
    fn a_changed_byte_moves_only_the_patches_it_reaches() {
        let encoder = encoder(17, 4, 9);
        let mut bytes: Vec<u8> = (0..32_u8).map(|b| b.wrapping_add(100)).collect();
        let patches = split(&scores(bytes.len(), &[0, 16]));
        let before = encoder.encode(&bytes, &patches).expect("同源");
        // 第 4 个字节的影响面是 `4..=11`，都留在第一个 patch（`0..16`）里。
        bytes[4] ^= 0xFF;
        let after = encoder.encode(&bytes, &patches).expect("同源");
        let dim = encoder.dim();
        assert_ne!(
            &before[..dim],
            &after[..dim],
            "改动的字节在第一个 patch 里，它的表示应当变"
        );
        assert_eq!(
            &before[dim..],
            &after[dim..],
            "第二个 patch 的字节没动：表示也不该动"
        );
    }

    /// 池化是 patch 内的加权和，不是取边界字节、也不是均匀平均。
    #[test]
    fn the_pool_is_an_attention_mix() {
        let encoder = encoder(5, 2, 1);
        let vectors = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0_f32; 2];
        encoder.pool(&vectors, &mut out);
        assert_ne!(out, vec![1.0, 1.0], "池化不是取边界字节");
        assert!(
            out.iter().all(|&v| (0.0..=1.0).contains(&v)),
            "凸组合的每一维落在字节向量的极值之间：{out:?}"
        );
        assert!(
            out.iter().any(|&v| v > 0.0 && v < 1.0),
            "权重不是 0/1：{out:?}"
        );
    }

    /// 相同的字节向量池化回它自己：softmax 的权重和为 1，均匀内容不该被缩放。
    #[test]
    fn a_uniform_patch_pools_back_to_its_vector() {
        let encoder = encoder(5, 3, 2);
        let common = [0.25_f32, -1.5, 3.0];
        let vectors: Vec<f32> = common
            .iter()
            .cycle()
            .take(3 * common.len())
            .copied()
            .collect();
        let mut out = vec![0.0_f32; 3];
        encoder.pool(&vectors, &mut out);
        for (got, want) in out.iter().zip(common) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    /// 切分来自另一次输入就拒绝，而不是拿它去给这一次的字节分行。
    #[test]
    fn a_split_from_another_input_is_rejected() {
        let encoder = encoder(7, 2, 0);
        let other = split(&scores(5, &[0]));
        let error = encoder
            .encode(&[1, 2, 3, 4], &other)
            .expect_err("切分不是这次输入的");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("5"), "{error}");
    }

    /// 只让字节 `A` 那行有分量的投影：分布归一、贪心给出 `A`。
    fn one_hot() -> ByteDecoder<f32> {
        let mut projection = vec![0.0_f32; BYTES];
        projection[b'A' as usize] = 2.0;
        ByteDecoder::new(1, &projection).expect("长度对得上")
    }

    /// 256-way 分布：通道数就是字节取值数，和为 1，贪心与分布一致。
    #[test]
    fn the_decoder_gives_one_distribution_over_every_byte_value() {
        let decoder = one_hot();
        let probs = decoder.probs(&[1.0]).expect("宽度对");
        assert_eq!(probs.len(), BYTES);
        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "分布归一：{total}");
        assert_eq!(decoder.byte(&[1.0]).expect("宽度对"), b'A');
        assert_eq!(
            decoder.logits(&[1.0]).expect("宽度对").len(),
            BYTES,
            "分数也是每字节一个"
        );
    }

    /// 并列时取最小字节值：同一份表示永远给出同一个字节。
    #[test]
    fn ties_go_to_the_smallest_byte_value() {
        let decoder = ByteDecoder::new(1, &vec![0.0_f32; BYTES]).expect("长度对得上");
        assert_eq!(decoder.byte(&[5.0]).expect("宽度对"), 0);
    }

    /// 几何在入口挡：宽度为零、投影长度不对、表示宽度不对，都报 `Invalid`。
    #[test]
    fn degenerate_geometry_fails_at_the_door() {
        assert_eq!(
            ByteDecoder::<f32>::new(0, &[])
                .expect_err("宽度为零")
                .code(),
            Code::Invalid
        );
        assert_eq!(
            ByteDecoder::<f32>::new(2, &[0.0; 3])
                .expect_err("长度不是 BYTES × width")
                .code(),
            Code::Invalid
        );
        let decoder = one_hot();
        for error in [
            decoder.logits(&[1.0, 2.0]).expect_err("宽了"),
            decoder.probs(&[]).expect_err("窄了"),
            decoder.byte(&[1.0, 2.0]).expect_err("宽了"),
        ] {
            assert_eq!(error.code(), Code::Invalid);
        }
    }

    /// `f64` 与 `f32` 是同一份算术，切分与表示只差精度。
    #[test]
    fn the_f64_encoder_agrees_with_the_f32_one() {
        let bytes: Vec<u8> = (0..16_u8).map(|b| b.wrapping_add(7)).collect();
        let patches = split(&scores(bytes.len(), &[0, 8]));
        let coarse = encoder(13, 3, 4).encode(&bytes, &patches).expect("同源");
        let fine = ByteEncoder::<f64>::new(filled(13, 3), 4)
            .encode(&bytes, &patches)
            .expect("同源");
        assert_eq!(coarse.len(), fine.len());
        for (a, b) in coarse.iter().zip(&fine) {
            assert!((f64::from(*a) - *b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    /// 表与种子是读侧口令：桶数、宽度、种子都原样回显，写侧不在这层。
    #[test]
    fn the_encoder_reads_the_table_it_was_given() {
        let encoder = encoder(13, 4, 7);
        assert_eq!(encoder.buckets(), 13);
        assert_eq!(encoder.dim(), 4);
        assert_eq!(encoder.seed(), 7);
        assert_eq!(encoder.table().len(), 13);
    }
}
