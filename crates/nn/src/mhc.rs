//! 双随机残差混合（mHC，§5.8.2）。
//!
//! 一条残差流变多条（Engram-27B 与 V4 取 `M = 4`），层与层之间用一张混合矩阵 `B_l` 把 M 条流重新组合：
//! `x_{l+1} = B_l x_l`。mHC 对 `B_l` 的要求是**双随机**且 `‖B_l‖₂ ≤ 1`——前者让每条流都是输入各流的
//! 凸组合（不放大、不歪偏，也不需要再学一组缩放），后者保证混合是**非扩张**的：流之间怎么混，
//! 激活都不会随层数指数增长。
//!
//! 归一化不另写：[`sinkhorn`] 与 CSA 的块权重是同一份实现；`‖B‖₂` 由 [`spectral_norm`] 实测后留在
//! [`Mhc::sigma`] 上——"`≤ 1`"因此是一条可复核的测量值，而不是注释里的承诺。
//!
//! 非负性是**入口硬条件**，不是清理，理由有两条：
//!
//! 1. Sinkhorn 只做行/列缩放，负号原地不动，而带负号的矩阵即使行和列和都为 1 也可能把信号放大——
//!    `[[1.5, −0.5], [−0.5, 1.5]]` 的双随机性成立、谱范数却是 `1.5 + 0.5 = 2`。
//! 2. 更隐蔽的一层：`spectral_norm` 的幂法从**全 1** 起步，而对上面那个矩阵，全 1 正是特征值 1 的
//!    特征方向——实测报的是 `1.0`，比真值小一半。非负矩阵没有这个风险：由 Perron–Frobenius，主特征
//!    向量各分量同号，全 1 起步不会正交到主方向上。
//!
//! 非负搭上了，`‖B‖₂ ≤ 1` 才由 Birkhoff–von Neumann 定理保证（双随机阵是置换阵的凸组合，每个
//! 置换阵谱范数为 1），而不是靠实测碰运气。
//!
//! 分工写清楚：mHC 混的是**流**（同一条残差里的多个副本），[`crate::mix`] 合的是**通道**（连续 / 事件
//! 两套表示），两者都是在给 `x` 做加权和一个矩阵乘法，但前者不持门控标量、后者不持矩阵。

use diagnostic::{Code, Error, Result};
use op::scan::Scalar;

use crate::csa::{accumulate, finite, sinkhorn, spectral_norm};

/// 报错出处：几何与取值都在构造期失败，不留到混合的中途。
const SITE: &str = "nn::mhc";

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
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "混合矩阵必须是 [{streams}, {streams}] = {size} 个元素，得到 {}",
                    mix.len()
                ),
            )
            .with(SITE));
        }
        check_entries(mix)?;
        let mut normalized = mix.to_vec();
        if !sinkhorn(&mut normalized, streams, ITERS, S::from_f64(TOL)) {
            return Err(Error::new(
                Code::Invalid,
                "Sinkhorn 归一后仍不是双随机：矩阵里有行和或列和为零（归一时原地不动），这张矩阵没有分布可归一",
            )
            .with(SITE));
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
        let width = self.check(streams)?;
        Ok((0..self.streams)
            .map(|r| self.blend(r, streams, width))
            .collect())
    }

    /// 入口校验：流条数与各条宽度一致，返回公共宽度。
    fn check(&self, streams: &[Vec<S>]) -> Result<usize> {
        if streams.len() != self.streams {
            return Err(Error::new(
                Code::Invalid,
                format!("必须是 {} 条流，得到 {} 条", self.streams, streams.len()),
            )
            .with(SITE));
        }
        let width = streams.first().map_or(0, Vec::len);
        if width == 0 {
            return Err(Error::new(Code::Invalid, "流宽度不能为零").with(SITE));
        }
        if let Some(bad) = streams.iter().position(|row| row.len() != width) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "第 {bad} 条流宽度是 {}，与第 0 条的 {width} 不一致",
                    streams[bad].len()
                ),
            )
            .with(SITE));
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
        Some(0) | None => Err(Error::new(
            Code::Invalid,
            format!("流数必须非零且平方不溢出，得到 {streams}"),
        )
        .with(SITE)),
        Some(size) => Ok(size),
    }
}

/// 每个元素都必须有限且非负。
fn check_entries<S: Scalar>(mix: &[S]) -> Result<()> {
    let Some(bad) = mix.iter().position(|&v| !finite(v) || v < S::ZERO) else {
        return Ok(());
    };
    Err(Error::new(
        Code::Invalid,
        format!(
            "混合矩阵第 {bad} 项是 {:?}：必须是有限非负数——负号不会被 Sinkhorn 消掉，而带负号的矩阵即使双随机也可能把信号放大",
            mix[bad]
        ),
    )
    .with(SITE))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csa::is_doubly_stochastic;

    /// 三条流、宽四：每条的每个位置取不同值，混合结果可逐项核对。
    fn streams() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![-5.0, -10.0, -15.0, -20.0],
        ]
    }

    /// 单位混合不改任何一条流。
    #[test]
    fn identity_leaves_the_streams_alone() {
        let mhc = Mhc::identity(3).expect("三条流");
        assert_eq!(mhc.mix_streams(&streams()).expect("几何合法"), streams());
    }

    /// 单流时混合是空操作：`M = 1` 的矩阵归一后必然是 `[[1]]`。
    #[test]
    fn one_stream_is_a_no_op() {
        let mhc = Mhc::new(&[2.5], 1).expect("单流");
        assert_eq!(mhc.mix(), &[1.0]);
        assert!((mhc.sigma() - 1.0).abs() < 1e-12);
        assert_eq!(
            mhc.mix_streams(&[vec![0.25, -0.5]]).expect("几何合法"),
            vec![vec![0.25, -0.5]]
        );
    }

    /// 每条输出流都是输入各流的凸组合：逐元素的取值必落在输入的最小值与最大值之间。
    #[test]
    fn a_row_is_a_convex_combination_of_the_input_rows() {
        let mix = [
            4.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0, 4.0,
        ];
        let mhc = Mhc::new(&mix, 4).expect("四流");
        let input: Vec<Vec<f64>> = (0..4)
            .map(|r| (0..3).map(|c| (r * 7 + c) as f64 - 5.0).collect())
            .collect();
        let out = mhc.mix_streams(&input).expect("几何合法");
        for (r, row) in out.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                let column: Vec<f64> = input.iter().map(|s| s[c]).collect();
                let low = column.iter().copied().fold(f64::INFINITY, f64::min);
                let high = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                assert!(
                    v >= low - 1e-12 && v <= high + 1e-12,
                    "第 {r} 条流第 {c} 维 {v} 逸出凸包 [{low}, {high}]"
                );
            }
        }
    }

    /// 混合不放大 L1 范数：`‖B ⊗ I‖₁ = 1`（列和为 1），于是整个多流张量的 L1 只会变小——
    /// 这正是「非扩张」的可测形态。
    #[test]
    fn mixing_does_not_expand_the_l1_norm() {
        let mix = [3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0];
        let mhc = Mhc::new(&mix, 3).expect("三流");
        let input = streams();
        let budget: f64 = input.iter().flatten().map(|v| v.abs()).sum();
        let out = mhc.mix_streams(&input).expect("几何合法");
        let total: f64 = out.iter().flatten().map(|v| v.abs()).sum();
        assert!(total <= budget + 1e-12, "输出 L1 {total} 超过输入 {budget}");
    }

    /// 任意正矩阵经 Sinkhorn 变成双随机，且与 CSA 用的是同一份归一化。
    #[test]
    fn sinkhorn_turns_a_positive_matrix_doubly_stochastic() {
        let raw = [3.0, 1.0, 0.5, 2.0];
        let mhc = Mhc::new(&raw, 2).expect("两流");
        let mut expected = raw;
        assert!(sinkhorn(&mut expected, 2, ITERS, TOL));
        assert_eq!(mhc.mix(), expected.as_slice());
        assert!(
            is_doubly_stochastic(mhc.mix(), 2, TOL),
            "行和、列和都必须是 1"
        );
    }

    /// 实测谱范数留在结果上且不超 1。
    #[test]
    fn the_measured_spectral_norm_stays_within_one() {
        let permutation = [0.0, 1.0, 1.0, 0.0];
        let blend = [0.7, 0.3, 0.3, 0.7];
        for mix in [
            permutation.to_vec(),
            blend.to_vec(),
            vec![1.0, 1.0, 1.0, 1.0],
        ] {
            let mhc = Mhc::new(&mix, 2).expect("两流");
            assert!(
                mhc.sigma() <= 1.0 + 1e-9,
                "{mix:?} 的实测谱范数 {} 超过 1",
                mhc.sigma()
            );
        }
    }

    /// 置换阵本来就是双随机：它只换流的位置，不做加权。
    #[test]
    fn a_permutation_swaps_the_streams() {
        let mhc = Mhc::new(&[0.0, 1.0, 1.0, 0.0], 2).expect("两流");
        let input = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(
            mhc.mix_streams(&input).expect("几何合法"),
            vec![vec![3.0, 4.0], vec![1.0, 2.0]]
        );
    }

    /// 负项被拒：Sinkhorn 不消符号，且幂法会被全 1 起步骗过。
    ///
    /// 这个反例双随机、特征值是 `1.5 ± 0.5`（即 1 与 2），真谱范数是 `2`；但 `spectral_norm` 从全 1
    /// 起步，全 1 正是特征值 1 的特征方向，于是它报出 `1.0`——「实测 ≤ 1」会假通过。
    #[test]
    fn a_negative_entry_is_refused() {
        let signed = [1.5, -0.5, -0.5, 1.5];
        assert!(
            is_doubly_stochastic(&signed, 2, TOL),
            "这个反例确实双随机：行和、列和都是 1"
        );
        let deduced = 1.5 + 0.5;
        let estimated = spectral_norm(&signed, 2, ITERS);
        assert!(
            (estimated - 1.0).abs() < 1e-9,
            "全 1 起步落在特征值 1 的特征方向上，估计值应当离真值 {deduced} 很远，实测 {estimated}"
        );
        assert_eq!(
            Mhc::new(&signed, 2).expect_err("负项必须被拒").code(),
            Code::Invalid
        );
    }

    /// 全零阵没有分布可归一：行/列和为零时归一化原地不动，永远到不了双随机。
    #[test]
    fn an_all_zero_matrix_has_no_distribution() {
        assert_eq!(
            Mhc::<f64>::new(&[0.0; 9], 3).expect_err("全零").code(),
            Code::Invalid
        );
    }

    /// 非有限项被拒：`NaN` 会静默穿过行/列缩放。
    #[test]
    fn a_non_finite_entry_is_refused() {
        assert_eq!(
            Mhc::new(&[1.0, f64::NAN, 1.0, 1.0], 2)
                .expect_err("NaN")
                .code(),
            Code::Invalid
        );
    }

    /// 几何问题一律在入口失败。
    #[test]
    fn geometry_is_checked_at_the_entry() {
        assert_eq!(
            Mhc::<f64>::new(&[], 0).expect_err("零流").code(),
            Code::Invalid
        );
        assert_eq!(
            Mhc::new(&[1.0, 3.0], 2).expect_err("长度不是 4").code(),
            Code::Invalid
        );
        let mhc = Mhc::identity(3).expect("三条流");
        let ragged = vec![vec![1.0, 2.0], vec![3.0], vec![4.0, 5.0]];
        assert_eq!(
            mhc.mix_streams(&ragged).expect_err("宽度不一致").code(),
            Code::Invalid
        );
        let empty = vec![Vec::<f64>::new(), vec![], vec![]];
        assert_eq!(
            mhc.mix_streams(&empty).expect_err("宽度为零").code(),
            Code::Invalid
        );
        let two = vec![vec![1.0], vec![2.0]];
        assert_eq!(
            mhc.mix_streams(&two).expect_err("条数不符").code(),
            Code::Invalid
        );
    }
}
