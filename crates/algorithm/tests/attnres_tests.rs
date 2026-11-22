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
        .map(|idx| {
            (0..4)
                .map(|row| weights[row] * refs[row][idx])
                .sum::<f64>()
        })
        .collect();
    for (idx, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!((a - b).abs() < 1e-12, "第 {idx} 维 {a} 不是手算的 {b}");
    }
}

/// 混合值落在坐标集的凸包内：不全为零的坐标上，结果不会跑出它们的取值范围。
#[test]
fn the_mixed_value_stays_in_the_convex_hull() {
    let rows = rows();
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let got = layer().mix(&query(), &refs).expect("几何合法");
    for (idx, value) in got.iter().enumerate() {
        let low = refs
            .iter()
            .map(|row| row[idx])
            .fold(f64::INFINITY, f64::min);
        let high = refs
            .iter()
            .map(|row| row[idx])
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            *value >= low - 1e-12 && *value <= high + 1e-12,
            "第 {idx} 维 {value} 越出 [{low}, {high}]"
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
    let near_second = layer().mix(&[0.0, 2.0, 0.0, 0.0], &refs).expect("几何合法");
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
    assert_eq!(
        splice(&main, &mixed, 1.0).expect("系数合法"),
        mixed.to_vec()
    );
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
        for (idx, value) in got.iter().enumerate() {
            let low = main[idx].min(mixed[idx]);
            let high = main[idx].max(mixed[idx]);
            assert!(
                *value >= low - 1e-12 && *value <= high + 1e-12,
                "α = {alpha} 时第 {idx} 维 {value} 越出 [{low}, {high}]"
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
    assert_eq!(AttnRes::new(0).expect_err("宽度为零").code(), Code::Invalid);
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
    for (idx, (a, b)) in wide.iter().zip(&narrow).enumerate() {
        assert!(
            (a - f64::from(*b)).abs() < 1e-6,
            "第 {idx} 维 f64 是 {a}、f32 是 {b}"
        );
    }
}
