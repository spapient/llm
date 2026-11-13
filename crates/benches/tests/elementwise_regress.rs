//! elementwise 性能回退闭环：真实测量 → regress 基线判定 → 回退拦截。
//!
//! 这是 `llm-benches`（测量）与 `llm-tool` 的 `regress` 模块（判定）的接线点。测量对象是 elementwise 骨架按
//! [`Add`] 单态化的 `forward`，指标取吞吐（elems/s，越大越好）。
//!
//! # 为什么基线在同进程锚定，而不是硬编码绝对数
//!
//! 跨机器的绝对吞吐不可比（CI 的 bench job 因此只 `--no-run` 编译，不判定）。这里先用同机实测
//! 锚定基线，再分别验证三件事：同条件实测落在预算内、注入的回退被拦截、硬下界能挡住灾难性
//! 内核。固定硬件上把基线换成持久化的锚定值（见 `Baseline`），同一套 `judge` 即成为跨提交
//! 门禁，无需改判定逻辑。

use std::hint::black_box;
use std::time::Instant;

use op::add_f32::Add;
use op::elementwise;
use tool::regress::{Baseline, Case, Direction, Judgement};

/// 每次测量的元素数。
const LEN: usize = 1 << 16;
/// 计时前的预热轮数（摊薄首次指令缓存与页错误）。
const WARMUP: u32 = 8;
/// 计时轮数。
const TIMED: u32 = 64;

/// 实测 elementwise `Add` 的吞吐：单位 elems/s（越大越好）。
fn throughput() -> f64 {
    let a: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.5 - 1.0).collect();
    let b: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.25 + 0.5).collect();
    let mut out = vec![0.0f32; LEN];

    for _ in 0..WARMUP {
        elementwise::forward::<Add>(&a, &b, &mut out).expect("缓冲等长");
    }

    let start = Instant::now();
    for _ in 0..TIMED {
        elementwise::forward::<Add>(black_box(&a), black_box(&b), black_box(&mut out))
            .expect("缓冲等长");
    }
    let elapsed = start.elapsed().as_secs_f64();
    (LEN as u64 * TIMED as u64) as f64 / elapsed
}

/// 同机同条件的实测应落在预算内：证明"测量 → 基线 → 判定"整条链路接通。
#[test]
fn the_measurement_feeds_the_baseline_and_passes() {
    let observed = throughput();
    let case = Case::new("elementwise/add_f32", observed, 50.0, Direction::Higher).expect("场景");
    let baseline = Baseline::new(vec![case]).expect("基线");

    let judgement = baseline
        .judge("elementwise/add_f32", observed)
        .expect("判定");
    assert!(judgement.is_pass(), "同条件实测应在预算内：{judgement:?}");
}

/// 注入的回退必须被拦截，翻倍的改进不报警：门禁对方向敏感。
#[test]
fn the_gate_catches_a_regression_and_rewards_an_improvement() {
    let observed = throughput();
    let case = Case::new("elementwise/add_f32", observed, 10.0, Direction::Higher).expect("场景");
    let baseline = Baseline::new(vec![case]).expect("基线");

    let regressed = baseline
        .judge("elementwise/add_f32", observed * 0.1)
        .expect("判定");
    assert!(
        matches!(regressed, Judgement::Regressed { .. }),
        "吞吐掉到 10% 必须拦截：{regressed:?}"
    );
    assert!(!regressed.is_pass(), "回退是唯一的失败态");

    let improved = baseline
        .judge("elementwise/add_f32", observed * 2.0)
        .expect("判定");
    assert!(
        matches!(improved, Judgement::Improved { .. }),
        "吞吐翻倍应记为改进：{improved:?}"
    );
    assert!(improved.is_pass(), "改进不阻断门禁");
}

/// 硬下界挡住灾难性内核：任何现代 CPU 对向量化加法都应远超该吞吐。
#[test]
fn the_hard_floor_blocks_a_catastrophic_kernel() {
    // 1e6 elems/s 是刻意保守的下限，给跨机器留足余量，只拦"彻底坏掉"的退化。
    let floor = Case::new(
        "elementwise/add_f32/floor",
        1_000_000.0,
        50.0,
        Direction::Higher,
    )
    .expect("场景");
    let baseline = Baseline::new(vec![floor]).expect("基线");

    let judgement = baseline
        .judge("elementwise/add_f32/floor", throughput())
        .expect("判定");
    assert!(
        judgement.is_pass(),
        "未达硬下界，疑似灾难性退化（如逐元素分配）：{judgement:?}"
    );
}

/// 未知场景是配置错误而非性能回退：基线名对不上时直接报错。
#[test]
fn an_unknown_scenario_is_a_configuration_error() {
    let case =
        Case::new("elementwise/add_f32", throughput(), 10.0, Direction::Higher).expect("场景");
    let baseline = Baseline::new(vec![case]).expect("基线");

    let error = baseline
        .judge("elementwise/mul_f32", 1.0)
        .expect_err("未登记场景");
    assert!(
        error.to_string().contains("elementwise/add_f32"),
        "错误信息列出已登记场景，便于排查配置：{error}"
    );
}
