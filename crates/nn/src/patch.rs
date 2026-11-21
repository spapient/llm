//! byte-latent 的动态 patch 编码（§5.17，第 11 款标杆的编码侧）。
//!
//! byte-latent 不是换 tokenizer，而是换前向：**字节 → 动态 patch → latent transformer → 逐字节解码**。
//! 没有固定词表，于是"哪些字节该被同一次全局前向一起看"必须先由一条**因果**规则决定。本模块只做
//! 这一步：把逐字节的信息量序列切成 patch 区间，并把"全局前向要跑几次"的记账放在结果上。两端的
//! 本地层在 [`crate::codec`]（`Boundary` 的分数与 patch 表示的编码、`Patches` 被逐 patch 编码），
//! 中间那层全局 transformer（复用既有 [`crate::Attention`] / [`crate::BlockTable`]）在装配侧。
//!
//! 两套边界规则都来自 §5.17：
//!
//! - [`Boundary::Entropy`]（BLT）：边界 = `H_i > θ_g` 或 `H_i − H_{i−1} > θ_r`。熵由一个小自回归
//!   模型算，阈值是人工旋钮；`θ_g` / `θ_r` 进配置之日按 §5.16 走 schema 演进，不默默填默认值。
//! - [`Boundary::Rate`]（ByteFlow）：按编码率取 Top-K，**没有人工阈值**。真实 ByteFlow 的 K 是整块
//!   上的，那要用到"还没到的字节"；而 §5.17 把**增量性**列为第一性质（同一前缀在生成的每一步都
//!   必须仍是同一个切分，否则 KV 与 patch 一起重算）。所以这里把 Top-K 落成**尾窗内的秩判定**：
//!   `H_i` 在最近 `window` 个分数里排进前 `top` 名就是边界——同一句话，只用了过去。
//!
//! 这一层有两个契约，都是可测的而非约定的：**覆盖**（区间首尾相接、无空 patch）与**前缀稳定**
//! （对前缀的切分等于完整切分截到该处），后者正是"边界只看过去"的形态。计算量记账也放在
//! [`Patches`] 上：全局 transformer 每个 patch 跑一次，[`Patches::len`] 就是次数，
//! [`Patches::mean_len`] 就是 §5.17 那条"patch 平均 8 字节"的左边。
//!
//! 用 `f32` 跑前向、`f64` 对拍，两档共用同一份泛型实现（[`op::scan::Scalar`]）：分数是熵还是
//! 信息密度由调用方给，本模块不替它选模型。

use std::ops::Range;

use diagnostic::{Code, Error, Result};
use op::scan::Scalar;

use crate::csa::finite;

/// 报错出处：参数与输入的问题都在入口失败，不留到切分中途。
const SITE: &str = "nn::patch";

/// 边界规则闭集：两套机制各自的参数。
///
/// 参数随变体走，于是"规则是熵、却拿着 window"这类状态在类型上就不存在——不是靠校验挡住的。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Boundary<S: Scalar = f32> {
    /// BLT 熵规则：`global` 是全局阈值 `θ_g`，`rise` 是突增阈值 `θ_r`，都非负。
    Entropy {
        /// 全局阈值 `θ_g`：`H_i` 超过它就断。
        global: S,
        /// 突增阈值 `θ_r`：`H_i − H_{i−1}` 超过它就断。
        rise: S,
    },
    /// ByteFlow 编码率规则：尾窗内的秩判定，`top` 名之内断。
    Rate {
        /// 回看宽度：只看最近这么多字节的分数。
        window: usize,
        /// 保留名次数：秩在 `1..=top` 之内就是边界，必须 `top ≤ window`。
        top: usize,
    },
}

impl<S: Scalar> Boundary<S> {
    /// 稳定的短名，写进报告与配置回显。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Entropy { .. } => "entropy",
            Self::Rate { .. } => "rate",
        }
    }

    /// 按这条规则切分逐字节的信息量序列（`scores.len()` 即字节数）。
    ///
    /// 首字节永远是一个 patch 的起点：patch 是区间，不从 0 起步就会有一段字节无处安放。
    ///
    /// 逐字节的判定只有一处（[`Boundary::opens`]），所以「事后一次性切」与「边生成边切」
    /// （[`crate::Emitter`]）不可能给出两种切法。
    ///
    /// # Errors
    ///
    /// 阈值非有限或为负、`window` / `top` 越界、分数里有 `NaN` / `±inf`——都返回
    /// `Code::Invalid`。非有限分数不当"不产生边界"处理：任何比较对它都是 `false`，那会把一整段
    /// 字节静默并进同一个 patch，出错的地方离原因很远。
    pub fn split(self, scores: &[S]) -> Result<Patches> {
        let mut starts = Vec::new();
        for index in 0..scores.len() {
            if self.opens(scores, index)? {
                starts.push(index);
            }
        }
        Ok(Patches {
            starts,
            bytes: scores.len(),
        })
    }

    /// 第 `index` 个字节是否开一个新的 patch——**只看 `0..=index`**，与 `index` 之后的字节无关。
    ///
    /// 这是切分的唯一判定处：[`Boundary::split`] 就是把它对每个下标跑一遍，逐字节生成时也只问它。
    /// 于是前缀稳定性由同一条算式给出：后面的字节再多，也不能把已经定下的起点撤销。
    ///
    /// # Errors
    ///
    /// 参数越界、`index` 不在 `0..scores.len()` 内（流式调用时最常见的错位），或 `index` 与
    /// `index - 1` 的分数非有限——都返回 `Code::Invalid`。
    pub fn opens(self, scores: &[S], index: usize) -> Result<bool> {
        self.probe(scores, index)?;
        Ok(index == 0 || self.breaks(scores, index))
    }

    /// 入口校验：参数、下标与**本字节及前一个字节**的分数都在定义域内。
    ///
    /// 首字节只看自己：它恒为起点，没有「前一个」可问。
    fn probe(self, scores: &[S], index: usize) -> Result<()> {
        self.check()?;
        if index >= scores.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "下标 {index} 越出 {len} 个字节：边界只问已有的字节",
                    len = scores.len()
                ),
            )
            .with(SITE));
        }
        self.window(scores, index)
    }

    /// 判定所需的两个字都必须在定义域内：放行 `NaN` 会让比较全部为假，一整段字节静默并成一个 patch。
    fn window(self, scores: &[S], index: usize) -> Result<()> {
        audit(index, scores[index])?;
        if index == 0 {
            return Ok(());
        }
        audit(index - 1, scores[index - 1])
    }

    /// 参数自检：越界的参数不切分，也不被夹到边界值上。
    fn check(self) -> Result<()> {
        match self {
            Self::Entropy { global, rise } => thresholds(global, rise),
            Self::Rate { window, top } => ranks(window, top),
        }
    }

    /// 第 `i` 个字节（`i ≥ 1`）是否开一个新的 patch。
    ///
    /// 熵规则的两条都是严格大于：阈值是下限而不是上限，取等不算越界。秩判定用"尾窗里严格大于它的
    /// 个数"当名次，取等时名次不后退，于是同一段序列的切分与遍历方向无关。
    fn breaks(self, scores: &[S], i: usize) -> bool {
        let score = scores[i];
        match self {
            Self::Entropy { global, rise } => score > global || score.sub(scores[i - 1]) > rise,
            Self::Rate { window, top } => {
                let from = i.saturating_sub(window);
                scores[from..i]
                    .iter()
                    .filter(|&&older| older > score)
                    .count()
                    < top
            }
        }
    }
}

/// 一次切分的结果：patch 的行区间与记账。
///
/// 只存起点（终点是下一个起点，末段收在总字节数上），所以"区间首尾相接、无空 patch"由构造保证，
/// 不是一条要维护的断言。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Patches {
    starts: Vec<usize>,
    bytes: usize,
}

impl Patches {
    /// patch 个数，也就是全局 transformer 要跑的次数：每 patch 一次，不是每字节一次。
    pub fn len(&self) -> usize {
        self.starts.len()
    }

    /// 是否一个 patch 都没有（输入为空）。
    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }

    /// 输入的字节数。
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// 第 `i` 个 patch 的字节区间 `[start, end)`；越界返回 `None`。
    pub fn get(&self, i: usize) -> Option<Range<usize>> {
        (i < self.starts.len()).then(|| self.span(i))
    }

    /// 全部 patch 的区间，按起点升序、末段收在总字节数上：遍历切分的唯一形态。
    ///
    /// 编码侧按这条遍历走（每个 patch 一行表示），于是"行数等于 patch 数"由单一来源给出。
    pub fn ranges(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        (0..self.starts.len()).map(move |i| self.span(i))
    }

    /// 第 `i` 个 patch 的区间；调用方已确认下标在界内。
    fn span(&self, i: usize) -> Range<usize> {
        self.starts[i]..self.starts.get(i + 1).copied().unwrap_or(self.bytes)
    }

    /// 平均 patch 长度（字节）：`bytes / len`。空输入记 `0` 而不是 `NaN`——没有 patch 时平均值
    /// 不存在，但把它写成 `NaN` 会让上游的求和与比较全部静默传播。
    pub fn mean_len(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.bytes as f64 / self.len() as f64
        }
    }
}

/// 第 `i` 个分数必须是有限值，不是就报出它在序列里的位置。
fn audit<S: Scalar>(i: usize, score: S) -> Result<()> {
    if finite(score) {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!("第 {i} 个字节的分数不是有限值（{score:?}）：熵与信息密度都是实数"),
    )
    .with(SITE))
}

/// 熵阈值必须非负有限：负阈值会让每个字节都成为边界，「门」也就不再是门。
fn thresholds<S: Scalar>(global: S, rise: S) -> Result<()> {
    if non_negative(global) && non_negative(rise) {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!("熵阈值必须非负有限：θ_g = {global:?}，θ_r = {rise:?}"),
    )
    .with(SITE))
}

/// 秩判定要 `1 ≤ top ≤ window`：`window = 0` 时秩无定义，`top > window` 时窗口里凑不齐名次。
fn ranks(window: usize, top: usize) -> Result<()> {
    if top >= 1 && top <= window {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!("编码率规则要 1 ≤ top ≤ window，得到 top = {top}、window = {window}"),
    )
    .with(SITE))
}

/// 非负有限：负值与 `NaN` / `±inf` 一样让阈值失去意义。
fn non_negative<S: Scalar>(value: S) -> bool {
    finite(value) && value >= S::ZERO
}

#[cfg(test)]
#[path = "../tests/patch_tests.rs"]
mod patch_tests;
