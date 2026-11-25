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

/// 覆盖：区间首尾相接、每个 patch 非空，最后一段收在总字节数上。
#[test]
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
#[test]
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
#[test]
fn entropy_above_the_global_threshold_starts_a_patch() {
    let patches = entropy(5.0, 1e9)
        .split(&[0.1, 0.2, 5.5, 0.1])
        .expect("参数合法");
    assert_eq!(starts(&patches), vec![0, 2]);
}

/// 熵突增就断，即使还没到全局阈值：这是"计算跟随复杂度"的局部形态。
#[test]
fn a_rise_starts_a_patch_below_the_global_threshold() {
    let patches = entropy(5.0, 0.8)
        .split(&[0.1, 0.1, 1.0, 1.0, 1.0])
        .expect("参数合法");
    assert_eq!(starts(&patches), vec![0, 2]);
}

/// 取等不算越界：阈值是下限，写成 `≥` 会让"θ_g = 最大值"退化成处处是边界。
#[test]
fn equality_with_the_threshold_does_not_start_a_patch() {
    let patches = entropy(0.5, 10.0)
        .split(&[0.5, 0.5, 0.4])
        .expect("参数合法");
    assert_eq!(patches.len(), 1);
}

/// 编码率只保留尾窗里最好的 `top` 名；`top = 1` 时退化成"新高才断"。
#[test]
fn the_rate_rule_keeps_the_best_of_the_trailing_window() {
    let scores = [0.1_f32, 0.4, 0.2, 0.9, 0.3, 0.2, 1.0];
    let patches = rate(3, 1).split(&scores).expect("参数合法");
    assert_eq!(starts(&patches), vec![0, 1, 3, 6]);
}

/// 计算跟随复杂度：噪声段的 patch 数多于平坦段，同等字节数下全局前向也就更多次。
#[test]
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
#[test]
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
#[test]
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
#[test]
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
#[test]
fn rules_have_stable_names() {
    assert_eq!(entropy(1.0, 1.0).as_str(), "entropy");
    assert_eq!(rate(4, 2).as_str(), "rate");
}

/// `f64` 的同一套规则给出同一切分：精度不是行为差异。
#[test]
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
