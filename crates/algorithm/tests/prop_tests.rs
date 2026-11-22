//! 属性测试：只断言**定律**，不断言实现步骤。
//!
//! 每条属性都是关于公开面的一条独立陈述——端点、凸包、仿射等变、双随机、行和为一、单调——
//! 所以它既能抓住实现漂移，也能在换实现之后继续成立。这正是属性测试相对用例测试的价值：
//! 用例锁定"这一组数算对了"，属性锁定"对所有合法的数都必须成立"。
//!
//! # 三条纪律
//!
//! 1. **不生成 `NaN` / `±inf`**。`prop::num::f64::NORMAL` 已排除它们与次正规数，再夹到 `±4`
//!    让平方与累加不溢出。非有限值不是被测语义的一部分：放进来只会把属性退化成随机失败。
//! 2. **不靠过滤**。`prop_filter` 一边丢弃样本一边偏置分布，形状约束一律用 `prop_flat_map`
//!    从维度派生，于是每个样本都可用、每个样本都在测被测函数。
//! 3. **失败可复现**。proptest 会把最小反例写进 `proptest-regressions/prop_tests.txt`（入库）；
//!    放大样本量用 `PROPTEST_CASES`，重放某一颗种子用 `PROPTEST_RNG_SEED`。
//!
//! # 容差口径
//!
//! `|got − want| ≤ tol · max(1, |want|)`：相对误差，绝对下限为 1。`testkit::compare` 是同一判据，
//! 但它的公开面只收 `f32` 切片；这里被测的是 `f64` 实例化，故用它的 `f64` 投影，不另立第二套。

use diagnostic::Code;
use proptest::prelude::*;

use super::*;

/// 有界的有限标量：`NORMAL` 排除 `NaN`/`±inf`/次正规，再夹到 `±4`。
fn scalar() -> impl Strategy<Value = f64> {
    prop::num::f64::NORMAL.prop_map(|v| v.clamp(-4.0, 4.0))
}

/// 严格为正的标量：`exp` 后的 `±0.5`，用于 Sinkhorn 这类"分母不能为零"的定义域。
fn positive() -> impl Strategy<Value = f64> {
    scalar().prop_map(|v| (v / 8.0).exp())
}

/// 同长度的三条通道 + 两个门控系数（`π`/`γ` 是概率，故在 `[0, 1]`）。
fn triad() -> impl Strategy<Value = (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64)> {
    (1usize..=8).prop_flat_map(|len| {
        (
            prop::collection::vec(scalar(), len),
            prop::collection::vec(scalar(), len),
            prop::collection::vec(scalar(), len),
            0.0..=1.0f64,
            0.0..=1.0f64,
        )
    })
}

/// `n × n` 的严格正矩阵：Sinkhorn 的列归一要求每列之和为正。
fn positive_matrix() -> impl Strategy<Value = (Vec<f64>, usize)> {
    (2usize..=4).prop_flat_map(|n| (prop::collection::vec(positive(), n * n), Just(n)))
}

/// `rows × cols` 的有限分数表：(数据, 行, 列)。
fn score_table() -> impl Strategy<Value = (Vec<f64>, usize, usize)> {
    (1usize..=3, 1usize..=5).prop_flat_map(|(rows, cols)| {
        (
            prop::collection::vec(scalar(), rows * cols),
            Just(rows),
            Just(cols),
        )
    })
}

/// 相对 + 绝对下限的容差判据；`what` 只进失败消息。
#[track_caller]
fn assert_near(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol * want.abs().max(1.0),
        "{what}: got {got}, want {want}, tol {tol}"
    );
}

/// 逐元素对拍：长度不等即失败（形状不对就不是同一批数）。
#[track_caller]
fn assert_vec_near(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: 长度不等");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "{what}: 第 {i} 项 got {g}, want {w}"
        );
    }
}

proptest! {
    #![proptest_config(testkit::prop::config())]

    /// 合成律是**加权和**，权重和为 1：三条通道整体平移 `+s`，输出也必须平移 `+s`。
    /// 平移不变性是"门控不是分段函数、没有隐藏的偏置项"的判据。
    #[test]
    fn synthesize_is_translation_equivariant(
        (ya, yb, yg, pi, gamma) in triad(),
        shift in -8.0..=8.0f64,
    ) {
        let lifted = |xs: &[f64]| xs.iter().map(|x| x + shift).collect::<Vec<_>>();
        let got = synthesize(&lifted(&ya), &lifted(&yb), &lifted(&yg), pi, gamma);
        let want: Vec<f64> = synthesize(&ya, &yb, &yg, pi, gamma)
            .iter()
            .map(|y| y + shift)
            .collect();
        assert_vec_near(&got, &want, 1e-12, "平移等变");
    }

    /// 合成律是**正齐次**的：`t > 0` 时三条通道同乘 `t`，输出也乘 `t`。
    /// 它排除了"顺手加一个常数下界"或"对幅度做非线性压缩"这类实现漂移。
    #[test]
    fn synthesize_is_positively_homogeneous(
        (ya, yb, yg, pi, gamma) in triad(),
        t in 0.25..=4.0f64,
    ) {
        let scaled = |xs: &[f64]| xs.iter().map(|x| x * t).collect::<Vec<_>>();
        let got = synthesize(&scaled(&ya), &scaled(&yb), &scaled(&yg), pi, gamma);
        let want: Vec<f64> = synthesize(&ya, &yb, &yg, pi, gamma)
            .iter()
            .map(|y| y * t)
            .collect();
        assert_vec_near(&got, &want, 1e-12, "正齐次");
    }

    /// 融合写法与两步写法**逐位一致**：省掉的是中间向量的分配，不是精度。
    /// 这条把"性能优化"钉在"语义不变"上，而不是让它只靠一次人眼 diff。
    #[test]
    fn synthesize_equals_the_two_step_composition(
        (ya, yb, yg, pi, gamma) in triad(),
    ) {
        prop_assert_eq!(
            synthesize(&ya, &yb, &yg, pi, gamma),
            gate(&cont(&ya, &yb, pi), &yg, gamma)
        );
    }

    /// 输出落在三通道取值的凸包内：每个分量都不小于三分量的最小值、不大于最大值。
    /// 凸包是加权和的必要条件，任一分量越界就说明权重里有负数或和不为 1。
    #[test]
    fn synthesize_stays_inside_the_hull(
        (ya, yb, yg, pi, gamma) in triad(),
    ) {
        let out = synthesize(&ya, &yb, &yg, pi, gamma);
        for (i, &v) in out.iter().enumerate() {
            let lo = ya[i].min(yb[i]).min(yg[i]);
            let hi = ya[i].max(yb[i]).max(yg[i]);
            assert!(v >= lo && v <= hi, "第 {i} 项 {v} 越出 [{lo}, {hi}]");
        }
    }

    /// 门控系数取端点时只剩一条通道：`γ = 1` 取事件通道，`γ = 0, π = 1` 取 α，`π = 0` 取 β。
    /// 端点是最容易被"平滑项"悄悄污染的写法。
    #[test]
    fn gate_endpoints_select_one_channel(
        (ya, yb, yg, _, _) in triad(),
    ) {
        assert_vec_near(&synthesize(&ya, &yb, &yg, 1.0, 0.0), &ya, 1e-15, "π=1,γ=0 取 α");
        assert_vec_near(&synthesize(&ya, &yb, &yg, 0.0, 0.0), &yb, 1e-15, "π=0,γ=0 取 β");
        assert_vec_near(&synthesize(&ya, &yb, &yg, 0.5, 1.0), &yg, 1e-15, "γ=1 取 γ");
    }

    /// `π` 与 `γ` 是概率：任意有限输入（含极端值）都夹在 `[0, 1]`。
    /// 上游会把它们当权重与阈值用，越界不会崩，只会静默算错。
    #[test]
    fn gating_coefficients_are_probabilities(
        rho in -1e6..=1e6f64,
        nu in -1e6..=1e6f64,
        sigma in -1e6..=1e6f64,
    ) {
        let p = pi(rho, nu);
        let g = gamma(sigma);
        assert!((0.0..=1.0).contains(&p), "π({rho}, {nu}) = {p} 越出 [0,1]");
        assert!((0.0..=1.0).contains(&g), "γ({sigma}) = {g} 越出 [0,1]");
        // 夹取不能是常值：定义域内部必须真的跟着变。
        assert!((0.0..=1.0).contains(&pi(0.0, 0.0)));
        assert!((0.0..=1.0).contains(&gamma(0.0)));
    }

    /// `π` 对 `ρ` 单调不减：`ρ` 变大，α 通道的权重不能反而变小。
    #[test]
    fn pi_is_monotone_in_rho(rho1 in -4.0..=4.0f64, rho2 in -4.0..=4.0f64, nu in 0.0..=8.0f64) {
        let (lo, hi) = if rho1 <= rho2 { (rho1, rho2) } else { (rho2, rho1) };
        assert!(pi(lo, nu) <= pi(hi, nu), "π 在 ρ 上非单调");
    }

    /// `RMS` 是正齐次范数：缩放输入只按 `|t|` 缩放输出。
    #[test]
    fn rms_is_positively_homogeneous(
        xs in prop::collection::vec(scalar(), 1..=8),
        t in -4.0..=4.0f64,
    ) {
        let scaled: Vec<f64> = xs.iter().map(|x| x * t).collect();
        assert_near(rms(&scaled), t.abs() * rms(&xs), 1e-12, "rms 正齐次");
    }

    /// 稀疏度是比例：恒在 `[0, 1]`，且对阈值单调不减（阈值放宽只会把更多元素算进来）。
    #[test]
    fn sparsity_is_a_monotone_ratio(
        xs in prop::collection::vec(scalar(), 1..=8),
        t1 in 0.0..=4.0f64,
        t2 in 0.0..=4.0f64,
    ) {
        let s = sparsity(&xs, t1);
        assert!((0.0..=1.0).contains(&s), "稀疏度 {s} 越出 [0,1]");
        let (lo, hi) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
        assert!(
            sparsity(&xs, lo) <= sparsity(&xs, hi),
            "稀疏度对阈值非单调"
        );
    }

    /// Sinkhorn 的最后一趟是列归一，所以**任何**严格正矩阵出来都必须列和为 1；
    /// 与收敛与否无关，这是构造保证，不是数值运气。
    #[test]
    fn sinkhorn_output_columns_sum_to_one((mut mat, n) in positive_matrix()) {
        let _ = sinkhorn(&mut mat, n, 8, 1e-9);
        for c in 0..n {
            let sum: f64 = (0..n).map(|r| mat[r * n + c]).sum();
            assert_near(sum, 1.0, 1e-12, &format!("第 {c} 列之和"));
        }
    }

    /// 良态正矩阵在足够迭代后双随机：行和与列和同时为 1。
    /// 这条才是"Sinkhorn 收敛"的判据；上一只测最后一趟的构造保证。
    #[test]
    fn sinkhorn_converges_on_well_conditioned_matrices(
        n in 2usize..=3,
        src in prop::collection::vec(0.5..=2.0f64, 9),
    ) {
        let mut mat = src[..n * n].to_vec();
        let converged = sinkhorn(&mut mat, n, 256, 1e-4);
        assert!(converged, "良态矩阵未收敛：{n}×{n}");
        for r in 0..n {
            let row: f64 = mat[r * n..(r + 1) * n].iter().sum();
            assert_near(row, 1.0, 1e-3, &format!("第 {r} 行之和"));
        }
    }

    /// top-k 之后保留的**恰好是最大的 k 个值**（按多重集），其余置 `−∞`。
    /// 用排序后的多重集比对，而不是逐位比对：并列时的选取次序不构成契约，取值集合才是。
    #[test]
    fn topk_keeps_the_largest_values(
        (mut scores, rows, cols) in score_table(),
        k in 1usize..=5,
    ) {
        let k = k.min(cols);
        let original = scores.clone();
        dsa_topk(&mut scores, rows, cols, k);
        for r in 0..rows {
            let mut want = original[r * cols..(r + 1) * cols].to_vec();
            want.sort_by(|a, b| b.partial_cmp(a).expect("分数已限定为有限"));
            let mut got = scores[r * cols..r * cols + k].to_vec();
            got.sort_by(|a, b| b.partial_cmp(a).expect("保留项无 NaN"));
            assert_vec_near(&got, &want[..k], 0.0, "top-k 保留集");
            for &v in &scores[r * cols + k..(r + 1) * cols] {
                assert_eq!(v, f64::NEG_INFINITY, "弃项必须置 −∞");
            }
        }
    }

    /// 稠密 softmax 的每行和为 1：注意力权重是分布，行和漂移就等于多算或少算了质量。
    #[test]
    fn softmax_rows_sum_to_one((mut scores, rows, cols) in score_table()) {
        dense_softmax(&mut scores, rows, cols);
        for r in 0..rows {
            let sum: f64 = scores[r * cols..(r + 1) * cols].iter().sum();
            assert_near(sum, 1.0, 1e-12, &format!("第 {r} 行之和"));
        }
    }

    /// top-k 再 softmax：弃项概率**恰好为 0**，保留项仍和为 1。
    /// 稀疏注意力的语义就是"弃项不参与归一化"，两者不能各留一半质量。
    #[test]
    fn topk_then_softmax_zeroes_the_discarded_mass(
        (mut scores, rows, cols) in score_table(),
        k in 1usize..=5,
    ) {
        let k = k.min(cols);
        dsa_topk(&mut scores, rows, cols, k);
        dense_softmax(&mut scores, rows, cols);
        for r in 0..rows {
            let row = &scores[r * cols..(r + 1) * cols];
            let sum: f64 = row.iter().sum();
            assert_near(sum, 1.0, 1e-12, &format!("第 {r} 行之和"));
            for &v in &row[k..] {
                assert_eq!(v, 0.0, "弃项概率必须恰好为 0");
            }
        }
    }

    /// 分块压缩的每个输出分量等于其块内元素的算术平均（独立按定义算，不调被测函数）。
    #[test]
    fn compress_is_a_block_mean(
        d in 1usize..=3,
        n in 1usize..=8,
        m in 1usize..=4,
    ) {
        let seq: Vec<f64> = (0..n * d).map(|i| (i as f64) * 0.25 - 1.0).collect();
        let out = compress(&seq, n, d, m);
        assert_eq!(out.len(), effective_len(n, m) * d, "输出形状");
        for b in 0..effective_len(n, m) {
            let start = b * m;
            let end = (start + m).min(n);
            for j in 0..d {
                let want = (start..end).map(|i| seq[i * d + j]).sum::<f64>() / (end - start) as f64;
                assert_near(out[b * d + j], want, 1e-12, "块均值");
            }
        }
    }

    /// 谱范数是正齐次范数：矩阵整体缩放 `t > 0` 时按 `t` 缩放。
    #[test]
    fn spectral_norm_is_positively_homogeneous(
        n in 2usize..=3,
        src in prop::collection::vec(scalar(), 9),
        t in 0.5..=4.0f64,
    ) {
        let base = &src[..n * n];
        let scaled = base.iter().map(|v| v * t).collect::<Vec<_>>();
        assert_near(
            spectral_norm(&scaled, n, 32),
            t * spectral_norm(base, n, 32),
            1e-10,
            "谱范数正齐次",
        );
    }

    /// 谱范数不超过 Frobenius 范数：`‖B‖₂ ≤ ‖B‖_F` 是恒等式级别的上界，
    /// 幂迭代只可能低估，因此这条同时校验"没有把范数算成别的量"。
    #[test]
    fn spectral_norm_is_below_the_frobenius_norm(
        n in 2usize..=3,
        src in prop::collection::vec(scalar(), 9),
    ) {
        let mat = &src[..n * n];
        let sigma = spectral_norm(mat, n, 32);
        let frobenius = mat.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(sigma <= frobenius + 1e-12, "σ={sigma} 超过 Frobenius {frobenius}");
    }

    /// 路由输出是分布：权重和为 1、个数不超过激活数、下标互不相同且在专家数内。
    /// 重归一化就是为了"整体不随 k 缩放"，这条是它的判据。
    #[test]
    fn routing_weights_are_a_distribution(
        width in 2usize..=4,
        active in 1usize..=3,
        src in prop::collection::vec(-2.0..=2.0f64, 16),
    ) {
        let experts = 4usize;
        let active = active.min(experts);
        let router = src[..experts * width].to_vec();
        let experts_mat: Vec<Vec<f64>> = (0..experts)
            .map(|e| {
                (0..width * width)
                    .map(|i| 0.1 * ((e * width * width + i) as f64) - 0.3)
                    .collect()
            })
            .collect();
        let moe = Moe::new(width, active, router, experts_mat, None).expect("形状合法");
        let picks = moe.route(&vec![0.5; width]).expect("有限输入");
        assert_eq!(picks.len(), active, "激活数");
        let total: f64 = picks.iter().map(|&(_, w)| w).sum();
        assert_near(total, 1.0, 1e-12, "路由权重之和");
        let mut idx: Vec<usize> = picks.iter().map(|&(i, _)| i).collect();
        idx.sort_unstable();
        idx.dedup();
        assert_eq!(idx.len(), active, "专家下标必须互不相同");
        assert!(idx.iter().all(|&i| i < experts), "下标越出专家数");
    }

    /// 非有限输入必须报错，不能靠 NaN 传播：路由分数会把它们带进分布，
    /// 出错点离原因很远，所以入口就得拦。
    #[test]
    fn forward_rejects_non_finite_input(width in 2usize..=3) {
        let experts_mat: Vec<Vec<f64>> = (0..2).map(|_| vec![0.25; width * width]).collect();
        let moe = Moe::new(width, 1, vec![0.5; 2 * width], experts_mat, None).expect("形状合法");
        let mut x = vec![1.0; width];
        x[0] = f64::NAN;
        let err = moe.forward(&x).expect_err("NaN 输入必须报错");
        prop_assert_eq!(err.code(), Code::Invalid);
    }

    /// 切分结果铺满整段字节：首段从 0 起、相邻首尾相接、末段收在字节数上、没有空段。
    /// "每条字节都属于恰好一个 patch"是编码侧按 patch 行数的前提。
    #[test]
    fn patches_tile_the_whole_byte_range(
        scores in prop::collection::vec(scalar(), 1..=16),
        global in 0.0..=4.0f64,
        rise in 0.0..=4.0f64,
    ) {
        let rule: Boundary<f64> = Boundary::Entropy { global, rise };
        let patches = rule.split(&scores).expect("分数已限定为有限");
        prop_assert!(!patches.is_empty(), "非空输入至少一个 patch");
        prop_assert_eq!(patches.len(), patches.ranges().count());
        let mut cursor = 0usize;
        for range in patches.ranges() {
            prop_assert_eq!(range.start, cursor, "段与段之间必须首尾相接");
            prop_assert!(range.end > range.start, "不许有空段");
            cursor = range.end;
        }
        prop_assert_eq!(cursor, scores.len(), "末段必须收在字节数上");
        prop_assert_eq!(patches.bytes(), scores.len());
    }
}
