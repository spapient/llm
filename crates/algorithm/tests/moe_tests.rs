use super::*;
use diagnostic::Code;

/// 专家 `e` 是 `(e + 1)·Id`：输出可直接按系数手算，路由对不对一眼可见。
fn experts<S: Scalar>(count: usize, width: usize) -> Vec<Vec<S>> {
    (0..count)
        .map(|e| {
            let mut matrix = vec![S::ZERO; width * width];
            for i in 0..width {
                matrix[i * width + i] = S::from_f64((e + 1) as f64);
            }
            matrix
        })
        .collect()
}

/// 路由第 `e` 行在首列取 `count − e`：分数 `(count − e)·x[0]` 互不相同且按专家次序递减。
fn router<S: Scalar>(count: usize, width: usize) -> Vec<S> {
    let mut out = vec![S::ZERO; count * width];
    for e in 0..count {
        out[e * width] = S::from_f64((count - e) as f64);
    }
    out
}

/// 两个专家的层，`k` 个激活；`shared` 为 `None`。
fn moe(active: usize) -> Moe<f64> {
    Moe::new(4, active, router(3, 4), experts(3, 4), None).expect("几何合法")
}

/// 固定输入：首项为 1，其余递减。
fn input() -> Vec<f64> {
    vec![1.0, 0.5, 0.25, 0.125]
}

/// 路由权重就是决策端口给出的分布，只在选中的 `k` 个上重新归一。
#[test]
fn the_route_is_the_decision_heads_distribution() {
    let moe = moe(2);
    let x = input();
    let scores: Vec<f64> = (0..3).map(|e| (3 - e) as f64 * x[0]).collect();
    let dist = Calib::Spherical.to_dist(&scores);
    let total = dist[0] + dist[1];
    let picks = moe.route(&x).expect("几何合法");
    assert_eq!(picks.len(), 2);
    assert_eq!(picks[0].0, 0, "分数最大的是专家 0");
    assert_eq!(picks[1].0, 1);
    assert!((picks[0].1 - dist[0] / total).abs() < 1e-12);
    assert!((picks[1].1 - dist[1] / total).abs() < 1e-12);
}

/// 换校准名只换端口，不改 TopK 的位置：softmax 下同样选前两个。
#[test]
fn the_calibration_name_only_changes_the_distribution() {
    let moe = moe(2).with_calib(Calib::Softmax);
    let x = input();
    let scores = [3.0_f64, 2.0, 1.0];
    let dist = Calib::Softmax.to_dist(&scores);
    let total = dist[0] + dist[1];
    let picks = moe.route(&x).expect("几何合法");
    assert_eq!(picks[0].0, 0);
    assert!((picks[0].1 - dist[0] / total).abs() < 1e-12);
}

/// 选中集合上的权重和为 1：输出不随 `k` 缩放。
#[test]
fn the_picked_weights_sum_to_one() {
    for active in 1..=3 {
        let total: f64 = moe(active)
            .route(&input())
            .expect("几何合法")
            .iter()
            .map(|&(_, w)| w)
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "k = {active} 时权重和为 {total}"
        );
    }
}

/// 权重降序、下标互不相同、`k` 个。
#[test]
fn the_pick_is_ordered_and_distinct() {
    let picks = moe(3).route(&input()).expect("几何合法");
    assert_eq!(
        picks.iter().map(|&(e, _)| e).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    let ordered = picks.windows(2).all(|w| w[0].1 >= w[1].1);
    assert!(ordered, "权重必须降序：{picks:?}");
}

/// 前向就是选中专家的加权和：手算 `Σ wₑ·(e+1)·x`。
#[test]
fn forward_is_the_weighted_sum_of_the_picked_experts() {
    let moe = moe(2);
    let x = input();
    let picks = moe.route(&x).expect("几何合法");
    let want: Vec<f64> = (0..4)
        .map(|i| {
            picks
                .iter()
                .map(|&(e, w)| w * ((e + 1) as f64 * x[i]))
                .sum::<f64>()
        })
        .collect();
    let got = moe.forward(&x).expect("几何合法");
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!((a - b).abs() < 1e-12, "第 {i} 维 {a} 不是手算的 {b}");
    }
}

/// 未选中的专家改了也不影响输出：稀疏激活是真的稀疏。
#[test]
fn an_unselected_expert_cannot_move_the_output() {
    let mut witnesses = expert_matrices();
    let before = moe(2).forward(&input()).expect("几何合法");
    witnesses[2] = vec![9.0; 16];
    let after = Moe::new(4, 2, router(3, 4), witnesses, None)
        .expect("几何合法")
        .forward(&input())
        .expect("几何合法");
    assert_eq!(before, after, "专家 2 没被选中，它的权重不该影响这次前向");
}

/// 共享专家恒参与、只算一次、且不被路由权重缩放。
#[test]
fn the_shared_expert_runs_once_and_unscaled() {
    let shared = experts::<f64>(1, 4).remove(0);
    let with = Moe::new(4, 2, router(3, 4), experts(3, 4), Some(shared.clone())).expect("几何合法");
    let x = input();
    let baseline = moe(2).forward(&x).expect("几何合法");
    for (i, (a, b)) in with
        .forward(&x)
        .expect("几何合法")
        .iter()
        .zip(&baseline)
        .enumerate()
    {
        assert!(
            (a - b - x[i]).abs() < 1e-12,
            "第 {i} 维：共享专家的贡献就是 1·(1·x)"
        );
    }
    assert_eq!(with.shared(), Some(shared.as_slice()));
}

/// `k` 等于专家数时每个专家都跑：任一专家改动都会改变输出。
#[test]
fn a_full_route_runs_every_expert() {
    let before = moe(3).forward(&input()).expect("几何合法");
    let mut touched = expert_matrices();
    touched[2] = vec![7.0; 16];
    let after = Moe::new(4, 3, router(3, 4), touched, None)
        .expect("几何合法")
        .forward(&input())
        .expect("几何合法");
    assert_ne!(before, after, "k = 3 时专家 2 也参与，改它必须改变输出");
}

/// 分数内积溢出时拒绝，而不是把 `NaN` 带进前向。
#[test]
fn an_overflowing_route_is_refused() {
    let moe = Moe::new(1, 1, vec![1e308_f64], vec![vec![1.0]], None).expect("几何合法");
    assert_eq!(
        moe.route(&[1e308])
            .expect_err("内积溢出，分布非有限")
            .code(),
        Code::Invalid
    );
}

/// 非有限输入在入口被拒：放行会让整条分布变成 `NaN`。
#[test]
fn a_non_finite_input_is_refused() {
    let moe = moe(2);
    assert_eq!(
        moe.route(&[f64::NAN, 0.0, 0.0, 0.0])
            .expect_err("输入非有限")
            .code(),
        Code::Invalid
    );
    assert_eq!(
        moe.forward(&[1.0, 2.0]).expect_err("宽度不符").code(),
        Code::Invalid
    );
}

/// 并列按最小下标：同一份输入在每次运行里都选中同一组专家。
#[test]
fn ties_keep_the_lower_idx() {
    let router = vec![1.0_f64, 0.0, 1.0, 0.0, 0.5, 0.0];
    let moe = Moe::new(2, 1, router, experts(3, 2), None).expect("几何合法");
    let picks = moe.route(&[1.0, 0.0]).expect("几何合法");
    assert_eq!(picks[0].0, 0, "并列时取下标 0，不是下标 1");
}

/// 几何非法一律在构造期失败。
#[test]
fn a_rejected_geometry_is_refused() {
    let good = experts::<f64>(2, 2);
    let cases: Vec<(&str, Result<Moe<f64>>)> = vec![
        ("宽度为零", Moe::new(0, 1, vec![], good.clone(), None)),
        ("没有专家", Moe::new(2, 1, vec![], vec![], None)),
        (
            "激活数为零",
            Moe::new(2, 0, vec![0.0; 4], good.clone(), None),
        ),
        (
            "激活数超过专家数",
            Moe::new(2, 3, vec![0.0; 4], good.clone(), None),
        ),
        (
            "路由矩阵尺寸不符",
            Moe::new(2, 1, vec![0.0; 3], good.clone(), None),
        ),
        (
            "专家不是方阵",
            Moe::new(
                2,
                1,
                vec![0.0; 4],
                vec![vec![0.0; 3], good[1].clone()],
                None,
            ),
        ),
        (
            "共享专家不是方阵",
            Moe::new(2, 1, vec![0.0; 4], good.clone(), Some(vec![0.0; 5])),
        ),
    ];
    for (why, case) in cases {
        let error = case.expect_err(why);
        assert_eq!(error.code(), Code::Invalid, "{why}");
    }
}

/// 一组可改动的专家矩阵副本。
fn expert_matrices() -> Vec<Vec<f64>> {
    experts(3, 4)
}
