//! [`llm_testkit::prop`] 的自测：环境变量收口、生成器值域、精确相等断言的边界。
//!
//! 这里验的是**装置本身**——装置错了，四个 crate 的性质测试会一起假装绿。

use crate::prop::{self, DEFAULT_CASES, DEFAULT_SEED, MAX_SHRINK_ITERS};
use proptest::prelude::*;
use proptest::test_runner::RngSeed;

#[test]
fn absent_or_blank_cases_fall_back_to_the_default() {
    for raw in [None, Some(""), Some("   ")] {
        assert_eq!(prop::cases_from(raw), DEFAULT_CASES, "缺省取默认样本数");
    }
}

#[test]
fn a_positive_case_count_is_honoured() {
    assert_eq!(prop::cases_from(Some("512")), 512, "环境变量覆盖生效");
    assert_eq!(prop::cases_from(Some(" 64 ")), 64, "两侧空白不算配置错误");
}

/// `0` 会让每条性质"零次通过"：门禁看起来绿，实际什么都没验——必须响亮失败。
#[test]
#[should_panic(expected = "PROPTEST_CASES 必须是正整数")]
fn zero_cases_is_rejected_instead_of_silently_passing() {
    prop::cases_from(Some("0"));
}

#[test]
#[should_panic(expected = "PROPTEST_CASES 必须是正整数")]
fn a_malformed_case_count_is_rejected_instead_of_falling_back() {
    prop::cases_from(Some("many"));
}

#[test]
fn absent_seed_falls_back_to_the_fixed_default() {
    assert_eq!(prop::seed_from(None), DEFAULT_SEED, "默认种子固定");
    assert_eq!(prop::seed_from(Some("42")), 42, "种子可显式覆盖");
}

#[test]
#[should_panic(expected = "PROPTEST_RNG_SEED 必须是 u64")]
fn a_malformed_seed_is_rejected() {
    prop::seed_from(Some("0x2a"));
}

/// 配置必须真的接到两个解析器上，否则 `PROPTEST_CASES` 就成了无效旋钮。
#[test]
fn the_shared_config_is_wired_to_both_resolvers() {
    let config = prop::config();
    assert_eq!(config.cases, prop::cases(), "样本数来自 `cases()`");
    assert_eq!(config.max_shrink_iters, MAX_SHRINK_ITERS, "收缩上限统一");
    assert_eq!(
        config.rng_seed,
        RngSeed::Fixed(prop::seed()),
        "种子来自 `seed()`"
    );
}

proptest! {
    #![proptest_config(prop::config())]

    /// 生成器不能自己产出 `NaN`/`±inf`：否则每条性质都会在无意义的输入上失败。
    #[test]
    fn the_bounded_generators_stay_finite_and_in_range(value in prop::finite(), wide in prop::wide()) {
        prop_assert!(value.is_finite() && value.abs() < 1.0, "{value} 越出 (−1, 1)");
        prop_assert!(wide.is_finite() && wide != 0.0, "{wide} 不是非零有限值");
    }

    #[test]
    fn the_unit_generator_covers_the_closed_unit_interval(value in prop::unit()) {
        prop_assert!((0.0..=1.0).contains(&value), "{value} 越出 [0, 1]");
    }

    /// 对数均匀正数：严格为正，且落在 `e^{−4}..=e^{4}`。跨五个数量级才是它存在的理由。
    #[test]
    fn the_positive_generator_is_strictly_positive_and_log_symmetric(value in prop::positive()) {
        prop_assert!(value > 0.0 && value.is_finite(), "{value} 不是严格正有限值");
        prop_assert!((-4.0f32).exp() <= value && value <= 4.0f32.exp(), "{value} 越出 e^±4");
    }

    /// 宽动态范围生成器的**存在价值**是跨量级：每 10^k 都要真落到 f32 上。
    #[test]
    fn the_wide_generator_reaches_both_ends_of_the_f32_range(exponent in -30.0f32..=30.0) {
        prop_assert!(10.0f32.powf(exponent).is_finite(), "10^{exponent} 溢出");
    }
}

/// 取反律、网格落点这类断言里 `-0.0` 与 `0.0` 必须视为相等，否则合法实现会被判错。
#[test]
fn exact_equality_treats_signed_zeroes_as_equal() {
    prop::assert_exact("±0", &[-0.0, 0.0, 1.0], &[0.0, -0.0, 1.0]);
}

#[test]
#[should_panic(expected = "第 1 项")]
fn exact_equality_reports_the_first_mismatch() {
    prop::assert_exact("反例", &[1.0, 2.0], &[1.0, 3.0]);
}

/// 长度不等是"不是同一批数"，必须在比数值之前就拦下。
#[test]
#[should_panic(expected = "长度 2 ≠ 3")]
fn exact_equality_rejects_length_mismatch_before_comparing() {
    prop::assert_exact("长度", &[1.0, 2.0], &[1.0, 2.0, 3.0]);
}
