use super::*;
use diagnostic::Code;
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
