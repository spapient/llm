//! byte-latent 的潜在栈（§5.17，第 11 款标杆的中间层与解码层）。
//!
//! 一次前向是三段：**字节 → 动态 patch → latent transformer → 逐字节解码**。本模块补的是中间那两段的
//! **数值层**——[`Global`] 在 patch 序列上跑因果自注意力（每 patch 一次），[`Backread`] 让每个字节回读
//! 它可见的 patch 表示（逆向 cross-attention）。两端已各有主人：[`crate::ByteEncoder`] /
//! [`crate::ByteDecoder`] 把字节送进表示、把表示送进 256-way 分布，[`crate::Emitter`] 管生成循环。
//!
//! # 为什么全局那层跑在列上，而不是字节上
//!
//! [`Global::forward`] 收的是 `[patch, width]` 的列（[`crate::ByteEncoder::encode`] 的输出），一遍的
//! **行数**就是 patch 数：`Patches::len()` 是多少，全局层的遍数就是多少。§5.17 的「免费缩放轴」正是
//! 这一条——把平均 patch 拉长，全局前向的遍数与总成本一起降，而输出投影的宽度不变。本模块不为这条性质
//! 加记账字段：[`crate::Patches::len`] 已经是那个数。
//!
//! # 因果由寻址给出，不由掩码给出
//!
//! 全局层复用 [`crate::BlockTable`] 的块表寻址：位置 `t` 只看得见「整块落在 `t` 之前」的压缩块与
//! `[t+1−window, t]` 的原始行，取不到未来的数。本地层同一条判据换个形式——字节只回读自己所在 patch
//! 及更早的表示，索引上界就是自己的 patch 号。两条都被单测钉住（「改未来，过去逐位不变」）。
//!
//! # 两块各自的花费，写在能力旁边
//!
//! - [`Global`]：`depth · 行数` 次寻址，行数即 patch 数（块因子还能再压）。
//! - [`Backread`]：`depth · 字节数` 次回读，每次回读覆盖「本 patch 及更早」的全部表示。逐字节生成时
//!   可见的正好是已收口的那些（[`crate::Emitter`] 只拿收口的表示当条件），于是可见性不需要第二套规则。
//!
//! 本模块不持训练状态、不注册算子，也不定义前馈与激活：一层在这里就是 q/k/v/o 四张方阵加残差，与
//! [`crate::Moe`]「专家的 FFN 在它的外层 `Contract`」是同一种分工。纯算术统一走 [`op::scan::Scalar`]
//! 泛型（`f32` 前向、`f64` 对拍）；点积、softmax、加权累加与块表寻址都复用 `csa` 的那几份，
//! 不因模块边界重写。

use diagnostic::{Code, Error, Result};
use op::scan::Scalar;

use crate::csa::{BlockTable, accumulate, dot, softmax_row};
use crate::patch::Patches;

/// 报错出处：几何在入口失败，不留到前向中途。
const SITE: &str = "nn::latent";

/// 本地解码层的下限：BLT 的 local decoder 是 6–9 层（§5.17），取闭集而不是任意层数。
pub const LOCAL_MIN: usize = 6;

/// 本地解码层的上限。
pub const LOCAL_MAX: usize = 9;

/// 一层：q / k / v / o 四张行主序方阵，宽度由使用它的栈校验。
///
/// `Layer` 不知道自己有多大——宽度属于**栈**（同一个 `width` 要同时对上列宽、残差与输出投影），
/// 于是校验只有一处（[`Global::new`] / [`Backread::new`]），不会出现「层说宽 4、栈按宽 8 用」这种
/// 只在乘法里才对不上的状态。
#[derive(Debug, Clone, PartialEq)]
pub struct Layer<S: Scalar = f32> {
    q: Vec<S>,
    k: Vec<S>,
    v: Vec<S>,
    o: Vec<S>,
}

impl<S: Scalar> Layer<S> {
    /// 由四张方阵构造，参数顺序即 q / k / v / o。
    pub fn new(q: Vec<S>, k: Vec<S>, v: Vec<S>, o: Vec<S>) -> Self {
        Self { q, k, v, o }
    }

    /// 查询投影。
    pub fn q(&self) -> &[S] {
        &self.q
    }

    /// 键投影。
    pub fn k(&self) -> &[S] {
        &self.k
    }

    /// 值投影。
    pub fn v(&self) -> &[S] {
        &self.v
    }

    /// 输出投影。
    pub fn o(&self) -> &[S] {
        &self.o
    }

    /// 四张方阵都必须是 `[width, width]`：宽了是把别的层的数喂进来，窄了内积照算却是另一个数。
    fn check(&self, width: usize, stride: usize) -> Result<()> {
        for (label, matrix) in [
            ("q", &self.q),
            ("k", &self.k),
            ("v", &self.v),
            ("o", &self.o),
        ] {
            if matrix.len() != stride {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "投影 {label} 有 {} 个元素，应当是 width × width = {stride}（width = {width}）",
                        matrix.len()
                    ),
                )
                .with(SITE));
            }
        }
        Ok(())
    }
}

/// patch 序列上的全局 transformer：块表因果自注意力 + 残差，一层接一层。
///
/// 输入是 `[patch, width]` 的列，输出同形。行数即遍数——这是 §5.17 把「按 patch 前向」和「按字节前向」
/// 分开的那一层；层内不做前馈与归一化（它们在装配侧），于是这里能测的性质只有一条：**因果成立**。
#[derive(Debug, Clone, PartialEq)]
pub struct Global<S: Scalar = f32> {
    width: usize,
    factor: usize,
    window: usize,
    topk: Option<usize>,
    layers: Vec<Layer<S>>,
}

impl<S: Scalar> Global<S> {
    /// 由宽度、块表几何（`factor` / `window`）与逐层投影装配。
    ///
    /// # Errors
    ///
    /// 宽度为零或宽度平方溢出、`factor` / `window` 为零、某层四张方阵长度不等于 `width × width`——
    /// 返回 `Code::Invalid`。零层的栈允许存在且是恒等（消融基线），不是错误。
    pub fn new(width: usize, factor: usize, window: usize, layers: Vec<Layer<S>>) -> Result<Self> {
        let stride = stride(width)?;
        if factor.min(window) == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("块表几何必须非零（factor = {factor}，window = {window}）"),
            )
            .with(SITE));
        }
        for layer in &layers {
            layer.check(width, stride)?;
        }
        Ok(Self {
            width,
            factor,
            window,
            topk: None,
            layers,
        })
    }

    /// 只留每行分数最高的 `topk` 个槽（CSA 的那条压缩轴）；`None` 是全留。
    ///
    /// 阈值掩码而不是重排：掩掉的槽权重为零，槽序不变，于是「分数与值错配」这种错不可能出现。
    #[must_use]
    pub fn with_topk(mut self, topk: Option<usize>) -> Self {
        self.topk = topk;
        self
    }

    /// 列宽。
    pub fn width(&self) -> usize {
        self.width
    }

    /// 层数。
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// 块压缩因子：`1` 是逐行（稠密因果），越大远处压得越狠。
    pub fn factor(&self) -> usize {
        self.factor
    }

    /// 滑窗宽度：最近 `window` 行按原始行参与，不走块表。
    pub fn window(&self) -> usize {
        self.window
    }

    /// 每行的 top-k 槽数。
    pub fn topk(&self) -> Option<usize> {
        self.topk
    }

    /// 把一列 patch 表示跑完整个栈。
    ///
    /// 第 `t` 行的输出只依赖前 `t+1` 行（因果由块表寻址给出），所以「改未来、过去逐位不变」是这条
    /// 路径的性质而不是测试的巧合。空列（零个 patch）返回空列。
    ///
    /// # Errors
    ///
    /// 列长不是宽度的整数倍，或块表拒绝的几何——返回 `Code::Invalid`。
    pub fn forward(&self, reps: &[S]) -> Result<Vec<S>> {
        let rows = rows(reps.len(), self.width)?;
        let mut column = reps.to_vec();
        for layer in &self.layers {
            column = self.attend(layer, &column, rows)?;
        }
        Ok(column)
    }

    /// 一层：`x + W_o · BlockTable(W_k x, W_v x).attend(W_q x, t)`。
    fn attend(&self, layer: &Layer<S>, x: &[S], rows: usize) -> Result<Vec<S>> {
        let query = project(layer.q(), self.width, x);
        let keys = project(layer.k(), self.width, x);
        let values = project(layer.v(), self.width, x);
        let table = BlockTable::new(&keys, &values, rows, self.width, self.factor, self.window)?;
        let mut out = Vec::with_capacity(x.len());
        for t in 0..rows {
            let slot = &query[t * self.width..(t + 1) * self.width];
            let mixed = table.attend(slot, t, self.topk)?;
            push_row(&mut out, x, t, self.width, layer.o(), &mixed);
        }
        Ok(out)
    }
}

/// 字节侧的逆向 cross-attention 栈：每个字节回读「自己所在 patch 及更早」的表示。
///
/// 层数是闭集 [`LOCAL_MIN`]..=[`LOCAL_MAX`]（BLT 的 local decoder 是 6–9 层）。查询是字节自己的状态，
/// 键值是本 patch 与更早 patch 的表示——与 [`Global`] 的方向相反：那里是 patch 读 patch，这里是字节读
/// patch。残差使零权重成为恒等，于是「这一层到底读了什么」可以用差分测出来，不必靠注释承诺。
#[derive(Debug, Clone, PartialEq)]
pub struct Backread<S: Scalar = f32> {
    width: usize,
    layers: Vec<Layer<S>>,
}

impl<S: Scalar> Backread<S> {
    /// 由列宽与逐层投影装配。
    ///
    /// # Errors
    ///
    /// 层数不在 [`LOCAL_MIN`]..=[`LOCAL_MAX`]、宽度为零或平方溢出、某层方阵长度不等于
    /// `width × width`——返回 `Code::Invalid`。
    pub fn new(width: usize, layers: Vec<Layer<S>>) -> Result<Self> {
        let stride = stride(width)?;
        if !(LOCAL_MIN..=LOCAL_MAX).contains(&layers.len()) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "逆向 cross-attention 的层数是闭集 {LOCAL_MIN}..={LOCAL_MAX}，给了 {}",
                    layers.len()
                ),
            )
            .with(SITE));
        }
        for layer in &layers {
            layer.check(width, stride)?;
        }
        Ok(Self { width, layers })
    }

    /// 列宽。
    pub fn width(&self) -> usize {
        self.width
    }

    /// 层数。
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// 逐字节回读：`glyphs` 是每个字节的本地向量（`[bytes, width]`），`reps` 是全局那一列的产物
    /// （`[patches, width]`），输出是同形的逐字节状态，下一步交给 [`crate::ByteDecoder`]。
    ///
    /// 字节 `j` 的可见集由它所属的 patch 给出（`patches` 是「哪些字节同属一次全局前向」的唯一来源，
    /// 这里不重算一遍切分）：自己所在的 patch 及更早。于是「未来 patch 改不动过去的字节」是索引的
    /// 性质，不是掩码的性质。
    ///
    /// # Errors
    ///
    /// 表示行数与 patch 数不等、字节数或本地向量的行数对不上、宽度为零、某层方阵尺寸不符——返回
    /// `Code::Invalid`。宽了是把别的层的数喂进来，窄了内积照算却是另一个数。
    pub fn forward(&self, reps: &[S], glyphs: &[S], patches: &Patches) -> Result<Vec<S>> {
        let visible = rows(reps.len(), self.width)?;
        if visible != patches.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "表示有 {visible} 行，patch 有 {} 个：一行一个 patch",
                    patches.len()
                ),
            )
            .with(SITE));
        }
        if glyphs.len() != glyph_stride(patches, self.width)? {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "本地向量有 {} 个元素，应当是 bytes × width = {} × {} = {}",
                    glyphs.len(),
                    patches.bytes(),
                    self.width,
                    glyph_stride(patches, self.width)?
                ),
            )
            .with(SITE));
        }
        let mut states = glyphs.to_vec();
        for layer in &self.layers {
            states = self.read(layer, &states, reps, patches)?;
        }
        Ok(states)
    }

    /// 一层：每个字节的查询回读可见 patch，读到的值与自己的状态相加（残差）。
    fn read(
        &self,
        layer: &Layer<S>,
        states: &[S],
        reps: &[S],
        patches: &Patches,
    ) -> Result<Vec<S>> {
        let keys = project(layer.k(), self.width, reps);
        let values = project(layer.v(), self.width, reps);
        let mut out = Vec::with_capacity(states.len());
        for (index, range) in patches.ranges().enumerate() {
            let block = &states[range.start * self.width..range.end * self.width];
            for state in block.chunks_exact(self.width) {
                out.extend(self.read_byte(layer, state, &keys, &values, index));
            }
        }
        Ok(out)
    }

    /// 一个字节的回读：`h + W_o · Σ_{i ≤ p} softmax(⟨W_q h, W_k r_i⟩/√d) · W_v r_i`。
    fn read_byte(
        &self,
        layer: &Layer<S>,
        state: &[S],
        keys: &[S],
        values: &[S],
        owner: usize,
    ) -> Vec<S> {
        let query = apply(layer.q(), self.width, state);
        let scale = S::ONE.div(S::from_f64((self.width as f64).sqrt()));
        let mut scores: Vec<S> = (0..=owner)
            .map(|i| {
                let key = &keys[i * self.width..(i + 1) * self.width];
                dot(&query, key).mul(scale)
            })
            .collect();
        softmax_row(&mut scores);
        let mut read = vec![S::ZERO; self.width];
        for (i, &weight) in scores.iter().enumerate() {
            let value = &values[i * self.width..(i + 1) * self.width];
            accumulate(&mut read, value, weight);
        }
        let update = apply(layer.o(), self.width, &read);
        state
            .iter()
            .zip(&update)
            .map(|(&base, &delta)| base.add(delta))
            .collect()
    }
}

/// `width × width`，宽度为零或乘积溢出都是入口错误。
fn stride(width: usize) -> Result<usize> {
    if width == 0 {
        return Err(Error::new(Code::Invalid, "列宽不能为零").with(SITE));
    }
    width
        .checked_mul(width)
        .ok_or_else(|| Error::new(Code::Invalid, format!("{width} × {width} 溢出")))
}

/// 一行一个 patch：列长必须是宽度的整数倍，否则「第几行是哪个 patch」无从谈起。
fn rows(len: usize, width: usize) -> Result<usize> {
    if width == 0 || !len.is_multiple_of(width) {
        return Err(Error::new(
            Code::Invalid,
            format!("表示列有 {len} 个元素，不是宽度 {width} 的整数倍"),
        )
        .with(SITE));
    }
    Ok(len / width)
}

/// 逐字节本地向量的元素数（`bytes × width`），溢出即入口错误。
fn glyph_stride(patches: &Patches, width: usize) -> Result<usize> {
    patches.bytes().checked_mul(width).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("{} 个字节 × {width} 溢出", patches.bytes()),
        )
    })
}

/// 行主序方阵乘向量：不引第三个转置视角，`[r, c]` 就是 `r * width + c`。
fn apply<S: Scalar>(matrix: &[S], width: usize, vector: &[S]) -> Vec<S> {
    (0..width)
        .map(|r| dot(&matrix[r * width..(r + 1) * width], vector))
        .collect()
}

/// 整列投影：每行一次 [`apply`]，行数由列长决定（已校验是整数倍）。
fn project<S: Scalar>(matrix: &[S], width: usize, column: &[S]) -> Vec<S> {
    column
        .chunks_exact(width)
        .flat_map(|row| apply(matrix, width, row))
        .collect()
}

/// 残差落位：第 `t` 行写 `x_t + W_o · mixed`。
fn push_row<S: Scalar>(out: &mut Vec<S>, x: &[S], t: usize, width: usize, o: &[S], mixed: &[S]) {
    let row = &x[t * width..(t + 1) * width];
    let update = apply(o, width, mixed);
    out.extend(
        row.iter()
            .zip(&update)
            .map(|(&base, &delta)| base.add(delta)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::Boundary;

    /// 单位阵：`m[i][i] = 1`。
    fn identity(width: usize) -> Vec<f32> {
        let mut m = vec![0.0_f32; width * width];
        for i in 0..width {
            m[i * width + i] = 1.0;
        }
        m
    }

    /// 零阵：残差只剩自己。
    fn zero(width: usize) -> Vec<f32> {
        vec![0.0_f32; width * width]
    }

    fn layer(width: usize) -> Layer<f32> {
        Layer::new(
            identity(width),
            identity(width),
            identity(width),
            identity(width),
        )
    }

    /// 六层（闭集下界）的非零栈。
    fn backread(width: usize) -> Backread<f32> {
        Backread::new(width, vec![layer(width); LOCAL_MIN]).expect("六层在闭集内")
    }

    /// 列：第 `r` 行的第 `d` 列是 `r + 1 + d/10`，行间可辨、列间可辨。
    fn column(rows: usize, width: usize) -> Vec<f32> {
        (0..rows)
            .flat_map(|r| (0..width).map(move |d| (r + 1) as f32 + d as f32 / 10.0))
            .collect()
    }

    /// 每两个字节切一刀：分数 `2.5` 严格大于阈值 `2`，于是 patch 是 `[0,2)`、`[2,4)`、……
    fn patches(bytes: usize) -> Patches {
        let rule = Boundary::Entropy {
            global: 2.0,
            rise: 1.0e9,
        };
        let scores: Vec<f32> = (0..bytes)
            .map(|i| if i % 2 == 0 { 2.5 } else { 0.0 })
            .collect();
        rule.split(&scores).expect("分数有限、参数合法")
    }

    #[test]
    fn a_row_never_reads_the_rows_after_it() {
        for (factor, window) in [(1, 1), (2, 1), (2, 3), (4, 2)] {
            let global = Global::new(2, factor, window, vec![layer(2)]).expect("几何合法");
            let before = column(5, 2);
            let mut after = before.clone();
            after[4 * 2] += 7.0;
            after[4 * 2 + 1] -= 3.0;
            let seen = global.forward(&before).expect("列长是宽度的整数倍");
            let shifted = global.forward(&after).expect("列长是宽度的整数倍");
            for cell in 0..4 * 2 {
                assert_eq!(
                    seen[cell], shifted[cell],
                    "factor={factor} 第 {cell} 格被未来改了"
                );
            }
            assert_ne!(
                seen[8..],
                shifted[8..],
                "factor={factor} 最后一行没被自己的数影响"
            );
        }
    }

    #[test]
    fn a_single_row_reads_only_itself() {
        let global = Global::new(2, 1, 1, vec![layer(2)]).expect("几何合法");
        let out = global.forward(&[1.0, -2.0]).expect("一行");
        assert_eq!(out, vec![2.0, -4.0], "只有自己可见，softmax 必为 1");
    }

    #[test]
    fn zero_weights_leave_the_column_untouched() {
        let empty = Layer::new(zero(2), zero(2), zero(2), zero(2));
        let global = Global::new(2, 2, 1, vec![empty]).expect("几何合法");
        let reps = column(4, 2);
        assert_eq!(global.forward(&reps).expect("列合法"), reps);
    }

    #[test]
    fn the_column_length_decides_the_pass_count() {
        let bytes = 12;
        let cuts = patches(bytes);
        let global = Global::new(2, 1, 1, vec![layer(2)]).expect("几何合法");
        let reps = column(cuts.len(), 2);
        let out = global.forward(&reps).expect("列合法");
        assert_eq!(out.len() / global.width(), cuts.len());
        assert!(cuts.len() < bytes, "两字节一刀时遍数必须少于字节数");
    }

    #[test]
    fn an_empty_column_has_nothing_to_run() {
        let global = Global::new(2, 1, 1, vec![layer(2)]).expect("几何合法");
        assert!(global.forward(&[]).expect("空列合法").is_empty());
    }

    #[test]
    fn the_global_stack_rejects_impossible_geometry() {
        let none: Vec<Layer<f32>> = Vec::new();
        assert!(Global::new(0, 1, 1, none.clone()).is_err(), "宽度为零");
        assert!(Global::new(2, 0, 1, none.clone()).is_err(), "块因子为零");
        assert!(Global::new(2, 1, 0, none).is_err(), "滑窗为零");
        let short = Layer::new(vec![0.0; 3], zero(4), zero(4), zero(4));
        assert!(Global::new(2, 1, 1, vec![short]).is_err(), "方阵少一格");
        let global = Global::new(2, 1, 1, vec![layer(2)]).expect("几何合法");
        assert!(global.forward(&[1.0, 2.0, 3.0]).is_err(), "列长不是整数倍");
    }

    #[test]
    fn a_byte_never_reads_a_later_patch() {
        let cuts = patches(4);
        let read = backread(2);
        let reps = column(cuts.len(), 2);
        let glyphs = column(4, 2);
        let mut further = reps.clone();
        further[(cuts.len() - 1) * 2] += 5.0;
        further[(cuts.len() - 1) * 2 + 1] -= 1.0;
        let seen = read.forward(&reps, &glyphs, &cuts).expect("几何合法");
        let shifted = read.forward(&further, &glyphs, &cuts).expect("几何合法");
        let head = cuts.ranges().next().expect("首段存在").end * 2;
        assert_eq!(
            seen[..head],
            shifted[..head],
            "前面的字节被后面的 patch 改了"
        );
        assert_ne!(
            seen[head..],
            shifted[head..],
            "最后一段没读到自己那个 patch"
        );
    }

    #[test]
    fn a_byte_does_read_its_own_patch() {
        let cuts = patches(4);
        let read = backread(2);
        let glyphs = column(4, 2);
        let reps = column(cuts.len(), 2);
        let mut changed = reps.clone();
        changed[0] += 4.0;
        let seen = read.forward(&reps, &glyphs, &cuts).expect("几何合法");
        let shifted = read.forward(&changed, &glyphs, &cuts).expect("几何合法");
        assert_ne!(seen[..2], shifted[..2], "首个 patch 的字节读不到自己的表示");
    }

    #[test]
    fn zero_weights_leave_the_local_vectors_untouched() {
        let cuts = patches(4);
        let empty = Layer::new(zero(2), zero(2), zero(2), zero(2));
        let read = Backread::new(2, vec![empty; LOCAL_MIN]).expect("六层在闭集内");
        let glyphs = column(4, 2);
        let reps = column(cuts.len(), 2);
        assert_eq!(
            read.forward(&reps, &glyphs, &cuts).expect("几何合法"),
            glyphs
        );
    }

    #[test]
    fn a_zero_query_reads_the_visible_patches_evenly() {
        let cuts = patches(4);
        let query = zero(2);
        let read = Backread::new(
            2,
            vec![Layer::new(query, identity(2), identity(2), identity(2)); LOCAL_MIN],
        )
        .expect("六层在闭集内");
        let reps = vec![1.0, 0.0, 3.0, 4.0];
        let glyphs = vec![0.0_f32; 8];
        let out = read.forward(&reps, &glyphs, &cuts).expect("几何合法");
        let depth = LOCAL_MIN as f32;
        assert_eq!(out[..2], [depth, 0.0], "首段只有一个可见槽，读到它自己");
        assert_eq!(out[6..], [2.0 * depth, 2.0 * depth], "末段两个槽等权");
    }

    #[test]
    fn the_layer_count_is_the_closed_set() {
        assert_eq!((LOCAL_MIN, LOCAL_MAX), (6, 9));
        let width = 2;
        assert!(Backread::new(width, vec![layer(2); LOCAL_MIN + 1]).is_ok());
        assert!(
            Backread::new(width, vec![layer(2); LOCAL_MAX]).is_ok(),
            "上界本身在闭集内"
        );
        assert!(Backread::new(width, vec![layer(2); LOCAL_MIN - 1]).is_err());
        assert!(Backread::new(width, vec![layer(2); LOCAL_MAX + 1]).is_err());
        assert_eq!(backread(width).depth(), LOCAL_MIN);
    }

    #[test]
    fn the_local_stack_rejects_impossible_geometry() {
        assert!(
            Backread::new(0, vec![layer(2); LOCAL_MIN]).is_err(),
            "宽度为零"
        );
        let short = Layer::new(vec![0.0; 3], identity(2), identity(2), identity(2));
        assert!(
            Backread::new(2, vec![short; LOCAL_MIN]).is_err(),
            "方阵少一格"
        );
        let cuts = patches(4);
        let read = backread(2);
        let glyphs = column(4, 2);
        let reps = column(cuts.len(), 2);
        assert!(
            read.forward(&reps, &glyphs[..6], &cuts).is_err(),
            "字节少了"
        );
        assert!(
            read.forward(&reps[..2], &glyphs, &cuts).is_err(),
            "表示少了一行"
        );
    }

    #[test]
    fn f64_runs_the_same_arithmetic() {
        let cuts32 = patches(4);
        let cuts64 = Boundary::Entropy {
            global: 2.0_f64,
            rise: 1.0e9,
        }
        .split(&[2.5_f64, 0.0, 2.5, 0.0])
        .expect("分数有限、参数合法");
        assert_eq!(cuts32.len(), cuts64.len());
        let width = 2;
        let widen = |m: &Layer<f32>| {
            Layer::new(
                m.q().iter().map(|&v| f64::from(v)).collect(),
                m.k().iter().map(|&v| f64::from(v)).collect(),
                m.v().iter().map(|&v| f64::from(v)).collect(),
                m.o().iter().map(|&v| f64::from(v)).collect(),
            )
        };
        let read = backread(width);
        let double =
            Backread::new(width, read.layers.iter().map(widen).collect()).expect("六层在闭集内");
        let wide_reps = reps64(&cuts64, width);
        let wide_glyphs: Vec<f64> = column(4, width).into_iter().map(f64::from).collect();
        let single = read
            .forward(&column(cuts32.len(), width), &column(4, width), &cuts32)
            .expect("几何合法");
        let wide = double
            .forward(&wide_reps, &wide_glyphs, &cuts64)
            .expect("几何合法");
        for (a, b) in single.iter().zip(&wide) {
            assert!((f64::from(*a) - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    /// `f64` 版的表示列，取值与 [`column`] 相同。
    fn reps64(cuts: &Patches, width: usize) -> Vec<f64> {
        (0..cuts.len() * width)
            .map(|i| {
                let r = i / width;
                (r + 1) as f64 + (i % width) as f64 / 10.0
            })
            .collect()
    }
}
