//! 注意力相对残差（AttnRes）：第 10 款标杆（Kimi K3）的深度轴。
//!
//! 深层的残差是**累加**：第 `l` 层的输入是前面所有层输出的和，于是越深，新增的那一层对总和的相对
//! 贡献越小——梯度与信号都被早期层淹没。AttnRes 把「往里加」换成「按注意力取」：深层不再累加，
//! 而是用一次注意力在**前几层的输出**上取一个凸组合，再并回主路。
//!
//! 三件套与它们的边界：
//!
//! - [`AttnRes`] 只做 `Σ w_j · V_j` 的凸组合，权重由 [`crate::csa::softmax_row`] 给出（与块表注意力、
//!   门控同一份 softmax，不因模块边界重写）。它不持参数、不写 KV：`V_j` 是调用方给的前几层输出。
//! - [`Sources`] 是这一层的**来源簿记**：哪些层号喂给本层、主路从哪一层并入。深度改配方时，读的是簿记
//!   而不是重算一遍下标。
//! - [`splice`] 是并回主路的那一步：`main + α·(mixed − main)`。`α = 0` 原样返回主路、`α = 1` 直接替换，
//!   `α ∈ (0, 1)` 是两者之间——`α ≤ 1` 因此保证这次并回**非扩张**。
//!
//! # 为什么来源按层号而不是按位置
//!
//! 位置（`values` 里的第几个）是这一层的私有视角，层号是模型级的名字。用层号记来源，才能让"第 40 层
//! 取第 1/2 层"这件事在装配侧可读、可测、可换配方；换成位置的话，任何一次重排都会悄无声息地改语义。
//! 层号必须严格递增且不与主路重复——重复的层会在一份自注意力里被算两次，等于偷偷改了权重。
//!
//! 空来源被拒：那等于「按注意力取空集」，softmax 无定义，残差也没有东西可并。

use diagnostic::{Code, Error, Result};
use op::scan::Scalar;

use crate::csa::{accumulate, dot, finite, softmax_row};

/// 报错出处：来源簿记、几何与取值都在入口失败，不留到混合中途。
const SITE: &str = "nn::attnres";

/// 注意力相对残差：在前几层输出上按一次注意力取凸组合。
///
/// 不持参数：查询、键、值都由调用方给（真实实现里各是一次投影）。方法对元素类型泛型，同一份几何
/// 既跑 `f32` 也跑 `f64`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnRes {
    width: usize,
}

impl AttnRes {
    /// 由隐藏维 `d` 构造。
    ///
    /// # Errors
    ///
    /// `width` 为零——返回 `Code::Invalid`。
    pub fn new(width: usize) -> Result<Self> {
        if width == 0 {
            return Err(Error::new(Code::Invalid, "宽度必须非零").with(SITE));
        }
        Ok(Self { width })
    }

    /// 隐藏维 `d`。
    pub fn width(&self) -> usize {
        self.width
    }

    /// 一次混合：`Σ w_j · values[j]`，`w = softmax(q·valuesᵀ / √d)`。
    ///
    /// # Errors
    ///
    /// 查询宽度不是 `d`、坐标集为空、坐标集里任一宽度不是 `d`、或任一输入非有限——返回
    /// `Code::Invalid`（放行会让 `NaN` 分数经 softmax 铺满整组权重）。
    pub fn mix<S: Scalar>(&self, query: &[S], values: &[&[S]]) -> Result<Vec<S>> {
        self.check(query, values, "坐标集")?;
        let scores: Vec<S> = values.iter().map(|value| dot(query, value)).collect();
        self.combine(&scores, values)
    }

    /// 门控变体：分数由**注意力日志**给出，而不是查询的内积。
    ///
    /// 这是给装配侧的直通口：真实 K3 层的分数来自注意力日志（`Q·K/√d` 的总和），本模块不该凭空
    /// 重算一遍。`logs` 与 `values` 一一对应，几何与有限性仍在入口判。
    ///
    /// # Errors
    ///
    /// 日志条数与坐标集不等、坐标集为空、任一宽度不是 `d`、任一输入非有限——返回 `Code::Invalid`。
    pub fn gated<S: Scalar>(&self, logs: &[S], values: &[&[S]]) -> Result<Vec<S>> {
        self.check_rows(values, "坐标集")?;
        if logs.len() != values.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "注意力日志必须与坐标集一一对应，得到 {} 条日志对 {} 个坐标",
                    logs.len(),
                    values.len()
                ),
            )
            .with(SITE));
        }
        if let Some(bad) = logs.iter().position(|&log| !finite(log)) {
            return Err(Error::new(
                Code::Invalid,
                format!("注意力日志第 {bad} 条非有限：softmax 会把它摊到整组权重上"),
            )
            .with(SITE));
        }
        self.combine(logs, values)
    }

    /// 分数 → 权重 → 凸组合。两个入口共用这一份，权重只有一处定义。
    fn combine<S: Scalar>(&self, scores: &[S], values: &[&[S]]) -> Result<Vec<S>> {
        // 温度 `√d` 与注意力同一约定：分数随宽度增长，不除的话 `d = 4096` 时 softmax 早被压成 one-hot。
        let temperature = S::from_f64(self.width as f64).sqrt();
        let mut weights: Vec<S> = scores.iter().map(|&score| score.div(temperature)).collect();
        softmax_row(&mut weights);
        let mut out = vec![S::ZERO; self.width];
        for (weight, value) in weights.iter().zip(values) {
            accumulate(&mut out, value, *weight);
        }
        Ok(out)
    }

    /// 入口校验：查询宽度与坐标集。
    fn check<S: Scalar>(&self, query: &[S], values: &[&[S]], what: &str) -> Result<()> {
        self.check_row(query, "查询")?;
        self.check_rows(values, what)
    }

    /// 坐标集非空、每个坐标的宽度正确、取值有限。
    fn check_rows<S: Scalar>(&self, values: &[&[S]], what: &str) -> Result<()> {
        if values.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                format!("{what}为空：softmax 无定义，残差也没有东西可并"),
            )
            .with(SITE));
        }
        for (index, value) in values.iter().enumerate() {
            self.check_row(value, &format!("{what}第 {index} 项"))?;
        }
        Ok(())
    }

    /// 单行几何 + 有限性。
    fn check_row<S: Scalar>(&self, row: &[S], what: &str) -> Result<()> {
        if row.len() != self.width {
            return Err(Error::new(
                Code::Invalid,
                format!("{what}宽度必须是 {}，得到 {}", self.width, row.len()),
            )
            .with(SITE));
        }
        match row.iter().position(|&value| !finite(value)) {
            Some(index) => Err(Error::new(
                Code::Invalid,
                format!("{what}第 {index} 项非有限：softmax 会把 NaN 摊到整组权重上"),
            )
            .with(SITE)),
            None => Ok(()),
        }
    }
}

/// 并回主路：`main + α·(mixed − main)`，`α ∈ [0, 1]`。
///
/// `α = 0` 原样返回主路、`α = 1` 直接替换。系数在 `[0, 1]` 内时并回是**凸组合**，因此非扩张：
/// 混合值有多大，主路就最多走那么远，不会把激活放大。
///
/// # Errors
///
/// 主路与混合值宽度不等、任一含非有限值、或系数不在 `[0, 1]`——返回 `Code::Invalid`。
pub fn splice<S: Scalar>(main: &[S], mixed: &[S], alpha: S) -> Result<Vec<S>> {
    if main.len() != mixed.len() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "主路宽 {} 与混合值宽 {} 不等：并回是把同一维度的两段合起来",
                main.len(),
                mixed.len()
            ),
        )
        .with(SITE));
    }
    if !finite(alpha) || alpha < S::ZERO || alpha > S::ONE {
        return Err(Error::new(
            Code::Invalid,
            format!("系数是 {alpha:?}：必须在 [0, 1] 内（0 保留主路，1 换成混合值）"),
        )
        .with(SITE));
    }
    for (index, (a, b)) in main.iter().zip(mixed).enumerate() {
        if !finite(*a) || !finite(*b) {
            return Err(Error::new(
                Code::Invalid,
                format!("第 {index} 项非有限：并回会把 NaN 带进主路"),
            )
            .with(SITE));
        }
    }
    Ok(main
        .iter()
        .zip(mixed)
        .map(|(&keep, &mixed)| keep.add(alpha.mul(mixed.sub(keep))))
        .collect())
}

/// 一层的来源簿记：喂本层的前几层层号 + 主路取自哪一层。
///
/// 层号是模型级的名字（位置是这一层的私有视角）：装配侧读得到自己能解释的东西，重排 `values` 也不会
/// 改语义。构造期就判——来源必须**严格递增**且**都不等于主路**；重复的层会在一份自注意力里被算两次，
/// 等于偷偷改了权重。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sources {
    layers: Vec<usize>,
    main: usize,
}

impl Sources {
    /// 由来源层号与主路层号构造。
    ///
    /// # Errors
    ///
    /// 来源为空、来源不是严格递增、或任一来源等于主路——返回 `Code::Invalid`。
    pub fn new(layers: Vec<usize>, main: usize) -> Result<Self> {
        if layers.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                "来源为空：这一层没有可取的前层输出",
            )
            .with(SITE));
        }
        if let Some(gap) = layers.windows(2).position(|pair| pair[0] >= pair[1]) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "来源第 {} 与第 {} 项不是严格递增（{} ≥ {}）：重复的层会被算两次",
                    gap,
                    gap + 1,
                    layers[gap],
                    layers[gap + 1]
                ),
            )
            .with(SITE));
        }
        if layers.contains(&main) {
            return Err(Error::new(
                Code::Invalid,
                format!("主路第 {main} 层同时出现在来源里：同一条输出不能取两次"),
            )
            .with(SITE));
        }
        Ok(Self { layers, main })
    }

    /// 来源层号（严格递增）。
    pub fn layers(&self) -> &[usize] {
        &self.layers
    }

    /// 主路取自第几层。
    pub fn main(&self) -> usize {
        self.main
    }

    /// 来源层数：凸组合里的项数。
    pub fn depth(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4 维的层。
    fn layer() -> AttnRes {
        AttnRes::new(4).expect("几何合法")
    }

    /// 四个互不相同的坐标：内积各不相同，权重一眼看得出谁大。
    fn rows() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ]
    }

    /// 查询在首维：第 0 项内积最大。
    fn query() -> [f64; 4] {
        [2.0, 0.0, 0.0, 0.0]
    }

    /// 混合权重是 softmax：先按 `√d` 降温，再在整组上归一。
    #[test]
    fn the_weights_are_the_softmax_of_scaled_scores() {
        let rows = rows();
        let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let got = layer().mix(&query(), &refs).expect("几何合法");
        let scores = [2.0_f64, 1.0, 0.0, 0.0];
        let mut weights: Vec<f64> = scores.iter().map(|score| score / 2.0).collect();
        softmax_row(&mut weights);
        let want: Vec<f64> = (0..4)
            .map(|index| {
                (0..4)
                    .map(|row| weights[row] * refs[row][index])
                    .sum::<f64>()
            })
            .collect();
        for (index, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-12, "第 {index} 维 {a} 不是手算的 {b}");
        }
    }

    /// 混合值落在坐标集的凸包内：不全为零的坐标上，结果不会跑出它们的取值范围。
    #[test]
    fn the_mixed_value_stays_in_the_convex_hull() {
        let rows = rows();
        let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let got = layer().mix(&query(), &refs).expect("几何合法");
        for (index, value) in got.iter().enumerate() {
            let low = refs
                .iter()
                .map(|row| row[index])
                .fold(f64::INFINITY, f64::min);
            let high = refs
                .iter()
                .map(|row| row[index])
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                *value >= low - 1e-12 && *value <= high + 1e-12,
                "第 {index} 维 {value} 越出 [{low}, {high}]"
            );
        }
    }

    /// 只有一个坐标时，混合就是它本身：凸组合退化成取值。
    #[test]
    fn a_single_coordinate_is_itself() {
        let row = [0.25, -1.5, 3.0, 0.0];
        let got = layer().mix(&query(), &[&row]).expect("几何合法");
        assert_eq!(got, row.to_vec());
    }

    /// 权重大小随查询方向走：把查询指向第 2 项，结果更靠近第 2 项。
    #[test]
    fn the_query_direction_decides_the_pick() {
        let rows = rows();
        let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let near_second = layer()
            .mix(&[0.0, 2.0, 0.0, 0.0], &refs)
            .expect("几何合法");
        let far = (near_second[1] - refs[0][1]).abs();
        let close = (near_second[1] - refs[2][1]).abs();
        assert!(close < far, "指向第 2 项时结果该更靠近它：{near_second:?}");
    }

    /// 门控变体用注意力日志当分数：日志最高的那一项权重最大，结果离它最近。
    #[test]
    fn the_gated_variant_takes_the_logs_as_scores() {
        let rows = rows();
        let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let logs = [0.0_f64, 0.0, 8.0, 0.0];
        let got = layer().gated(&logs, &refs).expect("几何合法");
        // 日志只决定权重，不决定坐标：结果仍落在凸包内，且离日志最高的第 2 项最近。
        let distance = |row: &[f64]| {
            row.iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
        };
        let nearest = (0..4)
            .min_by(|&a, &b| distance(&rows[a]).total_cmp(&distance(&rows[b])))
            .expect("四个坐标");
        assert_eq!(nearest, 2, "离结果最近的是第 2 项，得到第 {nearest} 项");
        assert!(
            distance(&rows[2]) < 0.1,
            "日志领先 4 个标准差（8 / √4）时权重应当集中：距离 {}",
            distance(&rows[2])
        );
    }

    /// 日志条数与坐标集不等时拒绝：错位一分就是另一组权重。
    #[test]
    fn a_log_count_mismatch_is_refused() {
        let rows = rows();
        let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let refused = layer()
            .gated(&[1.0, 2.0, 3.0], &refs)
            .expect_err("三条日志对四个坐标");
        assert_eq!(refused.code(), Code::Invalid);
    }

    /// 保真并回：`α = 0` 原样、`α = 1` 替换、`α = 0.5` 取中点。
    #[test]
    fn the_delta_splice_keeps_the_main_path_at_zero() {
        let main = [1.0, 2.0, 3.0];
        let mixed = [-1.0, 0.0, 5.0];
        assert_eq!(splice(&main, &mixed, 0.0).expect("系数合法"), main.to_vec());
        assert_eq!(splice(&main, &mixed, 1.0).expect("系数合法"), mixed.to_vec());
        let half = splice(&main, &mixed, 0.5).expect("系数合法");
        assert_eq!(half, vec![0.0, 1.0, 4.0]);
    }

    /// `α ≤ 1` 时并回非扩张：结果的每一维都在主路与混合值之间。
    #[test]
    fn the_splice_never_widens_the_range() {
        let main = [1.0, -4.0, 0.5];
        let mixed = [3.0, 2.0, -7.0];
        for step in 0..=10 {
            let alpha = f64::from(step) / 10.0;
            let got = splice(&main, &mixed, alpha).expect("系数合法");
            for (index, value) in got.iter().enumerate() {
                let low = main[index].min(mixed[index]);
                let high = main[index].max(mixed[index]);
                assert!(
                    *value >= low - 1e-12 && *value <= high + 1e-12,
                    "α = {alpha} 时第 {index} 维 {value} 越出 [{low}, {high}]"
                );
            }
        }
    }

    /// 系数在 `[0, 1]` 之外被拒：`α > 1` 会让并回放大主路。
    #[test]
    fn a_splice_coefficient_outside_the_unit_interval_is_refused() {
        let main = [1.0, 2.0];
        let mixed = [3.0, 4.0];
        for bad in [-0.5, 1.5, f64::NAN] {
            let refused = splice(&main, &mixed, bad).expect_err("系数越界");
            assert_eq!(refused.code(), Code::Invalid);
        }
    }

    /// 并回两侧的几何与取值都在入口判。
    #[test]
    fn the_splice_checks_both_sides() {
        assert_eq!(
            splice(&[1.0, 2.0], &[1.0], 0.5)
                .expect_err("宽度不等")
                .code(),
            Code::Invalid
        );
        assert_eq!(
            splice(&[1.0, f64::NAN], &[1.0, 1.0], 0.5)
                .expect_err("主路非有限")
                .code(),
            Code::Invalid
        );
        assert_eq!(
            splice(&[1.0, 2.0], &[1.0, f64::INFINITY], 0.5)
                .expect_err("混合值非有限")
                .code(),
            Code::Invalid
        );
    }

    /// 来源簿记：读回来就是构造时给的层号与主路。
    #[test]
    fn the_source_reads_back_what_it_was_given() {
        let source = Sources::new(vec![0, 1, 2], 5).expect("来源合法");
        assert_eq!(source.layers(), &[0, 1, 2][..]);
        assert_eq!(source.main(), 5);
        assert_eq!(source.depth(), 3);
    }

    /// 来源必须严格递增、不与主路重复、且非空。
    #[test]
    fn a_malformed_source_is_refused() {
        for layers in [vec![], vec![1, 1], vec![2, 1], vec![0, 2, 2]] {
            let refused = Sources::new(layers.clone(), 9).expect_err("来源不合法");
            assert_eq!(refused.code(), Code::Invalid, "{layers:?}");
        }
        let overlap = Sources::new(vec![3, 4], 4).expect_err("主路也在来源里");
        assert_eq!(overlap.code(), Code::Invalid);
    }

    /// 构造期与入口的名义几何：零宽度、空坐标集、宽度不符、非有限值一律拒绝。
    #[test]
    fn geometry_is_checked_at_the_door() {
        assert_eq!(
            AttnRes::new(0).expect_err("宽度为零").code(),
            Code::Invalid
        );
        let layer = layer();
        let row = [1.0, 0.0, 0.0, 0.0];
        assert_eq!(
            layer.mix(&query(), &[]).expect_err("空坐标集").code(),
            Code::Invalid
        );
        assert_eq!(
            layer
                .mix(&[1.0, 0.0], &[&row])
                .expect_err("查询窄了两格")
                .code(),
            Code::Invalid
        );
        assert_eq!(
            layer
                .mix(&query(), &[&[1.0, 2.0]])
                .expect_err("坐标窄了两格")
                .code(),
            Code::Invalid
        );
        assert_eq!(
            layer
                .mix(&[f64::NAN, 0.0, 0.0, 0.0], &[&row])
                .expect_err("查询非有限")
                .code(),
            Code::Invalid
        );
        assert_eq!(
            layer
                .mix(&query(), &[&[f64::NAN, 0.0, 0.0, 0.0]])
                .expect_err("坐标非有限")
                .code(),
            Code::Invalid
        );
    }

    /// `f32` 与 `f64` 同一套算术：宽窄差别只在精度。
    #[test]
    fn f32_and_f64_agree() {
        let rows = rows();
        let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let wide = layer().mix(&query(), &refs).expect("几何合法");
        let rows32: Vec<Vec<f32>> = rows
            .iter()
            .map(|row| row.iter().map(|&value| value as f32).collect())
            .collect();
        let refs32: Vec<&[f32]> = rows32.iter().map(Vec::as_slice).collect();
        let query32 = [2.0_f32, 0.0, 0.0, 0.0];
        let narrow = layer().mix(&query32, &refs32).expect("几何合法");
        for (index, (a, b)) in wide.iter().zip(&narrow).enumerate() {
            assert!(
                (a - f64::from(*b)).abs() < 1e-6,
                "第 {index} 维 f64 是 {a}、f32 是 {b}"
            );
        }
    }
}
