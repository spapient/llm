//! byte-latent 的动态 patch 编码（§5.17，第 11 款标杆的编码侧）。

use std::ops::Range;

use crate::invalid;
use diagnostic::Result;
use nn::conv::scan::Scalar;

use crate::csa::finite;

/// 报错出处：参数与输入的问题都在入口失败，不留到切分中途。
const SITE: &str = "algorithm::patch";

/// 边界规则闭集：两套机制各自的参数。
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
        for idx in 0..scores.len() {
            if self.opens(scores, idx)? {
                starts.push(idx);
            }
        }
        Ok(Patches {
            starts,
            bytes: scores.len(),
        })
    }

    /// 第 `idx` 个字节是否开一个新的 patch——**只看 `0..=idx`**，与 `idx` 之后的字节无关。
    ///
    /// 这是切分的唯一判定处：[`Boundary::split`] 就是把它对每个下标跑一遍，逐字节生成时也只问它。
    /// 于是前缀稳定性由同一条算式给出：后面的字节再多，也不能把已经定下的起点撤销。
    ///
    /// # Errors
    ///
    /// 参数越界、`idx` 不在 `0..scores.len()` 内（流式调用时最常见的错位），或 `idx` 与
    /// `idx - 1` 的分数非有限——都返回 `Code::Invalid`。
    pub fn opens(self, scores: &[S], idx: usize) -> Result<bool> {
        self.probe(scores, idx)?;
        Ok(idx == 0 || self.breaks(scores, idx))
    }

    /// 入口校验：参数、下标与**本字节及前一个字节**的分数都在定义域内。
    fn probe(self, scores: &[S], idx: usize) -> Result<()> {
        self.check_patch()?;
        if idx >= scores.len() {
            return err!(
                "下标 {idx} 越出 {len} 个字节：边界只问已有的字节",
                len = scores.len()
            );
        }
        self.window(scores, idx)
    }

    /// 判定所需的两个字都必须在定义域内：放行 `NaN` 会让比较全部为假，一整段字节静默并成一个 patch。
    fn window(self, scores: &[S], idx: usize) -> Result<()> {
        audit(idx, scores[idx])?;
        if idx == 0 {
            return Ok(());
        }
        audit(idx - 1, scores[idx - 1])
    }

    /// 参数自检：越界的参数不切分，也不被夹到边界值上。
    fn check_patch(self) -> Result<()> {
        match self {
            Self::Entropy { global, rise } => thresholds(global, rise),
            Self::Rate { window, top } => ranks(window, top),
        }
    }

    /// 第 `i` 个字节（`i ≥ 1`）是否开一个新的 patch。
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

    accessors! {
        /// 输入的字节数。
        bytes -> usize = bytes,
    }

    /// 第 `i` 个 patch 的字节区间 `[start, end)`；越界返回 `None`。
    pub fn get(&self, i: usize) -> Option<Range<usize>> {
        (i < self.starts.len()).then(|| self.span(i))
    }

    /// 全部 patch 的区间，按起点升序、末段收在总字节数上：遍历切分的唯一形态。
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
    err!("第 {i} 个字节的分数不是有限值（{score:?}）：熵与信息密度都是实数")
}

/// 熵阈值必须非负有限：负阈值会让每个字节都成为边界，「门」也就不再是门。
fn thresholds<S: Scalar>(global: S, rise: S) -> Result<()> {
    if non_negative(global) && non_negative(rise) {
        return Ok(());
    }
    err!("熵阈值必须非负有限：θ_g = {global:?}，θ_r = {rise:?}")
}

/// 秩判定要 `1 ≤ top ≤ window`：`window = 0` 时秩无定义，`top > window` 时窗口里凑不齐名次。
fn ranks(window: usize, top: usize) -> Result<()> {
    if top >= 1 && top <= window {
        return Ok(());
    }
    err!("编码率规则要 1 ≤ top ≤ window，得到 top = {top}、window = {window}")
}

/// 非负有限：负值与 `NaN` / `±inf` 一样让阈值失去意义。
fn non_negative<S: Scalar>(value: S) -> bool {
    finite(value) && value >= S::ZERO
}

#[cfg(test)]
mod patch_tests {
    use super::*;

    /// 熵规则。
    fn entropy(global: f32, rise: f32) -> Boundary<f32> {
        Boundary::Entropy { global, rise }
    }

    /// 编码率规则。
    fn rate(window: usize, top: usize) -> Boundary<f32> {
        Boundary::Rate { window, top }
    }

    /// 切分的起点列表：区间由起点决定，比较切分就是比较这一列。
    fn starts(patches: &Patches) -> Vec<usize> {
        patches.ranges().map(|range| range.start).collect()
    }

    testkit::test_cases! {
    /// 覆盖：区间首尾相接、每个 patch 非空，最后一段收在总字节数上。
    fn every_byte_belongs_to_exactly_one_patch() {
        let scores = [
            0.2_f32, 6.0, 0.3, 0.3, 9.0, 0.1, 0.1, 0.1, 4.0, 0.5, 0.5, 0.5,
        ];
        let patches = entropy(5.0, 2.0).split(&scores).expect("参数与分数都合法");
        assert_eq!(starts(&patches), vec![0, 1, 4, 8]);
        let mut next = 0;
        for i in 0..patches.len() {
            let range = patches.get(i).expect("下标在 range 内");
            assert_eq!(range.start, next, "第 {i} 个 patch 必须接上一个的终点");
            assert!(range.end > range.start, "第 {i} 个 patch 不能是空区间");
            next = range.end;
        }
        assert_eq!(next, scores.len(), "最后一段必须收在总字节数上");
        assert_eq!(patches.bytes(), scores.len());
        assert_eq!(patches.get(patches.len()), None);
        let spans: Vec<Range<usize>> = patches.ranges().collect();
        assert_eq!(spans.len(), patches.len(), "一次遍历就是全部 patch");
        for (i, span) in spans.iter().enumerate() {
            assert_eq!(patches.get(i), Some(span.clone()), "两种读法给出同一个区间");
        }
    }

    /// 前缀稳定：对前缀的切分等于完整切分截到该处——边界只看过去，不看未来。
    ///
    /// 比的是**起点**而不是整段区间：落在前缀末尾的那个 patch 只是被截短了（终点改为前缀长），
    /// 并不会被撤销——能撤销才是"边界看了未来"。
    fn a_prefix_cuts_exactly_where_the_full_sequence_does() {
        let scores = [
            0.1_f32, 0.2, 7.0, 0.1, 0.2, 0.1, 8.0, 0.3, 0.1, 3.0, 0.1, 0.1, 9.0, 0.2, 0.2, 0.1,
        ];
        for rule in [entropy(5.0, 2.0), rate(4, 1)] {
            let full = starts(&rule.split(&scores).expect("参数合法"));
            for cut in 0..=scores.len() {
                let prefix = starts(&rule.split(&scores[..cut]).expect("前缀同样合法"));
                let expected: Vec<usize> = full.iter().copied().filter(|&s| s < cut).collect();
                assert_eq!(prefix, expected, "{}：截到 {cut} 时切分变了", rule.as_str());
            }
        }
    }

    /// 熵超过全局阈值就断（`rise` 放到极大以隔离这条）。
    fn entropy_above_the_global_threshold_starts_a_patch() {
        let patches = entropy(5.0, 1e9)
            .split(&[0.1, 0.2, 5.5, 0.1])
            .expect("参数合法");
        assert_eq!(starts(&patches), vec![0, 2]);
    }

    /// 熵突增就断，即使还没到全局阈值：这是"计算跟随复杂度"的局部形态。
    fn a_rise_starts_a_patch_below_the_global_threshold() {
        let patches = entropy(5.0, 0.8)
            .split(&[0.1, 0.1, 1.0, 1.0, 1.0])
            .expect("参数合法");
        assert_eq!(starts(&patches), vec![0, 2]);
    }

    /// 取等不算越界：阈值是下限，写成 `≥` 会让"θ_g = 最大值"退化成处处是边界。
    fn equality_with_the_threshold_does_not_start_a_patch() {
        let patches = entropy(0.5, 10.0)
            .split(&[0.5, 0.5, 0.4])
            .expect("参数合法");
        assert_eq!(patches.len(), 1);
    }

    /// 编码率只保留尾窗里最好的 `top` 名；`top = 1` 时退化成"新高才断"。
    fn the_rate_rule_keeps_the_best_of_the_trailing_window() {
        let scores = [0.1_f32, 0.4, 0.2, 0.9, 0.3, 0.2, 1.0];
        let patches = rate(3, 1).split(&scores).expect("参数合法");
        assert_eq!(starts(&patches), vec![0, 1, 3, 6]);
    }

    /// 计算跟随复杂度：噪声段的 patch 数多于平坦段，同等字节数下全局前向也就更多次。
    fn a_noisy_stretch_is_cut_finer_than_a_flat_one() {
        let flat = [0.1_f32; 32];
        let mut noisy = flat;
        for (i, slot) in noisy.iter_mut().enumerate() {
            *slot = if i % 2 == 0 { 0.2 } else { 6.0 };
        }
        let flat_len = entropy(5.0, 1.0).split(&flat).expect("参数合法").len();
        let noisy_len = entropy(5.0, 1.0).split(&noisy).expect("参数合法").len();
        assert!(
            noisy_len > flat_len,
            "平坦 {flat_len} 段 vs 噪声 {noisy_len} 段"
        );
    }

    /// 空输入没有 patch：不返回一个空区间，`mean_len` 记 0 而不是 `NaN`。
    fn an_empty_input_has_no_patches() {
        for rule in [entropy(1.0, 1.0), rate(4, 2)] {
            let patches = rule.split(&[]).expect("参数合法");
            assert!(patches.is_empty(), "{} 不该造出 patch", rule.as_str());
            assert_eq!(patches.len(), 0);
            assert_eq!(patches.bytes(), 0);
            assert_eq!(patches.mean_len(), 0.0);
            assert_eq!(patches.get(0), None);
        }
    }

    /// 平均长度就是 `bytes / len`：§5.17 的"patch 平均 8 字节"按这个算式核对。
    fn the_mean_length_is_bytes_over_patches() {
        let whole = entropy(5.0, 10.0).split(&[0.1_f32; 16]).expect("参数合法");
        assert_eq!(whole.len(), 1);
        assert_eq!(whole.mean_len(), 16.0);
        let one_by_one = rate(1, 1)
            .split(&[0.1_f32, 0.2, 0.3, 0.4])
            .expect("参数合法");
        assert_eq!(one_by_one.len(), 4);
        assert_eq!(one_by_one.mean_len(), 1.0);
    }

    /// 参数与输入的问题都在入口失败，不静默退化成"处处是边界"或"只有一段"。
    fn degenerate_parameters_and_scores_are_rejected() {
        for rule in [
            entropy(-1.0, 0.0),
            entropy(f32::NAN, 0.0),
            entropy(f32::INFINITY, 0.0),
            rate(0, 0),
            rate(4, 0),
            rate(4, 5),
        ] {
            assert!(
                rule.split(&[0.1, 0.2]).is_err(),
                "{} 应当被拒",
                rule.as_str()
            );
        }
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let scores = [0.1_f32, bad, 0.2];
            assert!(
                entropy(1.0, 1.0).split(&scores).is_err(),
                "{bad:?} 应当被拒"
            );
            assert!(rate(2, 1).split(&scores).is_err(), "{bad:?} 应当被拒");
        }
    }

    /// 规则短名稳定：写进报告与配置回显。
    fn rules_have_stable_names() {
        assert_eq!(entropy(1.0, 1.0).as_str(), "entropy");
        assert_eq!(rate(4, 2).as_str(), "rate");
    }

    /// `f64` 的同一套规则给出同一切分：精度不是行为差异。
    fn the_f64_engine_cuts_the_same_places() {
        let scores = [0.1_f64, 0.2, 7.0, 0.1, 0.2, 0.1, 8.0, 0.3];
        let fine = Boundary::<f64>::Entropy {
            global: 5.0,
            rise: 2.0,
        }
        .split(&scores)
        .expect("参数合法");
        let narrowed: Vec<f32> = scores.iter().map(|&x| x as f32).collect();
        let coarse = entropy(5.0, 2.0).split(&narrowed).expect("参数合法");
        assert_eq!(starts(&fine), starts(&coarse));
    }
    }
}
