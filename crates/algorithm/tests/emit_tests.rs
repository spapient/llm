//! `emit` 的用例。
//!
//! 生产文件只留执行与判据，用例集中于此；N3 行数门限不再受用例增删影响。

use super::*;
use crate::codec::{BYTES, ORDERS};
use nn::Table;

/// 一张每槽都不同的表：全零表会让"哪个字节"永远分不出来，测试也就测不出错位。
fn table<S: Scalar>(len: usize, dim: usize) -> Table<S> {
    let mut table = Table::new(len, dim);
    for k in 0..table.len() {
        let row: Vec<S> = (0..dim)
            .map(|d| S::from_f64(((k * (d + 3)) % 11) as f64 - 4.0))
            .collect();
        table.set(&[k as u32], 0, &row);
    }
    table
}

/// 测用的表示侧：8 维、每槽不同、种子固定。
fn encoder(dim: usize) -> ByteEncoder<f32> {
    ByteEncoder::new(table(23, dim), 7)
}

/// 测用的输出头：第 `b` 行只在 `b % dim` 这一维取 `1`，于是"选哪个字节"由条件的某一维决定，
/// 同维的字节并列时取最小字节值——确定可复现，不需要训练。
fn decoder(dim: usize) -> ByteDecoder<f32> {
    let mut projection = vec![0.0_f32; BYTES * dim];
    for b in 0..BYTES {
        projection[b * dim + b % dim] = 1.0;
    }
    ByteDecoder::new(dim, &projection).expect("投影长度对")
}

/// 按全局阈值切分：`rise` 放到极大，只留 `−ln p > global` 这一条（与 [`crate::Boundary`] 的
/// 熵规则同义）。
fn rule<S: Scalar>(global: f64) -> Boundary<S> {
    Boundary::Entropy {
        global: S::from_f64(global),
        rise: S::from_f64(1e9),
    }
}

/// 装好的循环：8 维表示、阈值 2 nat。
fn emitter() -> Emitter<f32> {
    Emitter::new(rule(2.0), encoder(8), decoder(8)).expect("两侧宽度对")
}

/// 预算用尽就停：产出正好 `ceiling` 个字节，并记 `Ceiling`。
#[test]
fn the_loop_stops_at_the_ceiling() {
    let emission = emitter().run(9, None).expect("状态自洽");
    assert_eq!(emission.len(), 9);
    assert_eq!(emission.halt(), Halt::Ceiling);
    assert_eq!(
        emission.bytes(),
        emitter().run(9, None).expect("状态自洽").bytes(),
        "同一参数两次生成必须逐字节相同"
    );
}

/// 零预算不生成，也不读成"不限"。
#[test]
fn a_zero_budget_emits_nothing() {
    let emission = emitter().run(0, Some(0)).expect("状态自洽");
    assert!(emission.is_empty());
    assert_eq!(emission.halt(), Halt::Ceiling);
}

/// 产出停止字节的那一步就停，停止字节含在产出里。
#[test]
fn a_stop_byte_halts_on_the_step_that_emits_it() {
    let first = emitter().run(1, None).expect("状态自洽").bytes()[0];
    let emission = emitter().run(16, Some(first)).expect("状态自洽");
    assert_eq!(emission.halt(), Halt::Stop);
    assert_eq!(emission.bytes(), &[first]);
}

/// 边生成边判与事后一次性切是同一份结果。
#[test]
fn the_incremental_split_equals_a_whole_sequence_split() {
    let mut emitter = emitter();
    emitter.run(24, None).expect("状态自洽");
    let patches = emitter.patches().expect("分数恒有限");
    let replay = rule::<f32>(2.0)
        .split(emitter.scores())
        .expect("分数恒有限");
    assert_eq!(patches, replay);
    assert_eq!(patches.bytes(), emitter.bytes().len());
    assert!(!patches.is_empty(), "有字节就至少有一个 patch");
}

/// 已收口的表示只算一次：它们与"整条切分一次编码"的那些行逐位相同，末段（未收口）不在里面。
#[test]
fn closed_patches_are_encoded_exactly_once() {
    let encoder = encoder(8);
    let mut emitter = Emitter::new(rule(0.5), encoder.clone(), decoder(8)).expect("宽度对");
    emitter.run(20, None).expect("状态自洽");
    let patches = emitter.patches().expect("分数恒有限");
    assert!(patches.len() > 1, "阈值 0.5 nat 下应当切出多条 patch");
    let all = encoder.encode(emitter.bytes(), &patches).expect("同源");
    assert_eq!(
        emitter.reprs().len(),
        (patches.len() - 1) * emitter.dim(),
        "末段还在长，不该有表示"
    );
    assert_eq!(emitter.reprs(), &all[..all.len() - emitter.dim()]);
}

/// 追加只往后长：再走几步，已经产出的字节与已经算出的表示都不变。
#[test]
fn a_prefix_is_never_rewritten() {
    let mut emitter = emitter();
    emitter.run(12, None).expect("状态自洽");
    let bytes = emitter.bytes().to_vec();
    let reprs = emitter.reprs().to_vec();
    emitter.run(12, None).expect("状态自洽");
    assert_eq!(&emitter.bytes()[..bytes.len()], &bytes[..], "字节被改写了");
    assert_eq!(
        &emitter.reprs()[..reprs.len()],
        &reprs[..],
        "已收口的表示被重算了"
    );
}

/// 空历史的条件是全零：还没有记忆可加，也不拿零向量假装有一条。
#[test]
fn an_empty_history_conditions_on_zero() {
    let emitter = emitter();
    assert!(
        emitter
            .condition()
            .expect("空状态自洽")
            .iter()
            .all(|&v| v == 0.0)
    );
    assert!(emitter.bytes().is_empty());
    assert!(emitter.reprs().is_empty());
}

/// 单 patch 时条件只剩本地上下文：没有收口的表示可加。
#[test]
fn a_single_open_patch_conditions_on_the_last_byte_only() {
    let encoder = encoder(8);
    let mut emitter = Emitter::new(rule(2.0), encoder.clone(), decoder(8)).expect("宽度对");
    emitter.seed(b"ab", &[0.5, 0.5]).expect("分数有限");
    assert!(emitter.reprs().is_empty(), "唯一那条 patch 还没收口");
    assert_eq!(
        emitter.condition().expect("状态自洽"),
        encoder.glyph(b"ab", 1).expect("下标在界内")
    );
}

/// 收口之后就换成"上一条 patch 的表示 + 末字节向量"。
#[test]
fn a_closed_patch_joins_the_condition() {
    let encoder = encoder(8);
    let mut emitter = Emitter::new(rule(2.0), encoder.clone(), decoder(8)).expect("宽度对");
    // 每字节 3 nat 都超过阈值：四个字节各成一条 patch，前三条立刻收口。
    emitter
        .seed(b"abcd", &[3.0, 3.0, 3.0, 3.0])
        .expect("分数有限");
    assert_eq!(emitter.reprs().len(), 3 * emitter.dim());
    let closed = encoder.patch(b"abcd", 2..3).expect("区间合法");
    let local = encoder.glyph(b"abcd", 3).expect("下标在界内");
    let want: Vec<f32> = closed.iter().zip(&local).map(|(&a, &b)| a + b).collect();
    assert_eq!(emitter.condition().expect("状态自洽"), want);
}

/// 预填充是替换而不是追加：换 prompt 不该留下上一次的切分与表示。
#[test]
fn seeding_replaces_the_history() {
    let mut emitter = emitter();
    emitter.seed(b"old", &[1.0, 1.0, 1.0]).expect("分数有限");
    emitter.seed(b"new", &[1.0, 1.0, 1.0]).expect("分数有限");
    assert_eq!(emitter.bytes(), b"new");
}

/// 阈值越紧切得越碎：同一段字节在低阈值下不会切得更少。
#[test]
fn a_tighter_threshold_cuts_at_least_as_often() {
    let mut loose = Emitter::new(rule(4.0), encoder(8), decoder(8)).expect("宽度对");
    let mut tight = Emitter::new(rule(0.2), encoder(8), decoder(8)).expect("宽度对");
    for _ in 0..16 {
        loose.step().expect("状态自洽");
        tight.step().expect("状态自洽");
    }
    let (loose, tight) = (
        loose.patches().expect("分数有限"),
        tight.patches().expect("分数有限"),
    );
    assert!(
        tight.len() >= loose.len(),
        "0.2 nat 切出 {} 条，4 nat 切出 {} 条",
        tight.len(),
        loose.len()
    );
}

/// 分数就是自信息：与字节一一对应、非负、且不超过 `ln BYTES`（贪心取的是最大通道）。
#[test]
fn the_scores_are_the_self_information_of_the_picked_byte() {
    let mut emitter = emitter();
    let produced = emitter.run(10, None).expect("状态自洽").bytes().to_vec();
    assert_eq!(emitter.scores().len(), produced.len());
    let ceiling = (BYTES as f64).ln();
    for (idx, &score) in emitter.scores().iter().enumerate() {
        assert!(
            score >= 0.0 && f64::from(score) <= ceiling + 1e-6,
            "第 {idx} 个字节的分数 {score} 越出 [0, {ceiling}]"
        );
    }
}

/// 停止原因是闭集，短名稳定。
#[test]
fn the_halt_names_are_stable() {
    assert_eq!(
        [Halt::Ceiling.as_str(), Halt::Stop.as_str()],
        ["ceiling", "stop"]
    );
}

/// 两侧宽度不等在装配时就拒：喂不进这个头。
#[test]
fn a_width_mismatch_is_refused_at_assembly() {
    let refused = Emitter::new(rule::<f32>(1.0), encoder(8), decoder(4)).expect_err("宽度不等");
    assert_eq!(refused.code(), Code::Invalid);
}

/// 预填充的长度或分数不合就拒，并且旧历史原样保留。
#[test]
fn a_bad_seed_is_refused_without_touching_the_history() {
    let mut emitter = emitter();
    emitter
        .seed(b"keep", &[0.5, 0.5, 0.5, 0.5])
        .expect("分数有限");
    let before = emitter.bytes().to_vec();
    assert_eq!(
        emitter
            .seed(b"ab", &[0.5, 0.5, 0.5])
            .expect_err("长度不等")
            .code(),
        Code::Invalid
    );
    assert_eq!(
        emitter
            .seed(b"ab", &[0.5, f32::NAN])
            .expect_err("分数非有限")
            .code(),
        Code::Invalid
    );
    assert_eq!(emitter.bytes(), &before[..], "失败不该推进一半状态");
}

/// 更宽的标量走同一个循环：`f64` 下遍数与覆盖范围与 `f32` 一致。
#[test]
fn a_wider_scalar_runs_the_same_loop() {
    let encoder = ByteEncoder::<f64>::new(table(23, 8), 7);
    let mut projection = vec![0.0_f64; BYTES * 8];
    for b in 0..BYTES {
        projection[b * 8 + b % 8] = 1.0;
    }
    let decoder = ByteDecoder::new(8, &projection).expect("投影长度对");
    let mut emitter = Emitter::new(rule::<f64>(0.0), encoder, decoder).expect("宽度对");
    let emission = emitter.run(6, None).expect("状态自洽");
    assert_eq!(emission.len(), 6, "阈值 0 下最碎：{} 阶", ORDERS.len());
    assert_eq!(
        emitter.patches().expect("分数有限").bytes(),
        emission.len(),
        "切分必须覆盖全部已产出字节"
    );
    assert!(
        emitter.reprs().len() < emission.len() * 8,
        "含末段的行不该全收口"
    );
}
