//! `latent` 的用例。
//!
//! 生产文件只留执行与判据，用例集中于此；N3 行数门限不再受用例增删影响。

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
