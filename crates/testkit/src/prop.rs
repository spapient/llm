//! 数值性质测试装置：**一份**配置、**一套**受约束生成器、**一条**容差口径。
//!
//! 三条纪律（原先 `nn`/`algorithm` 各写一份，现在只在这里定义）：
//!
//! 1. **不生成 `NaN` / `±inf` / 次正规**：一律从 [`prop::num::f32::NORMAL`] 出发，再用
//!    `v / (1 + |v|)` 平滑压进 `(−limit, limit)`。不用 `clamp`：夹逼把样本堆在边界上，最小反例
//!    退化成 `±limit`，收缩失去意义；除法是双射，边界不带质量。
//! 2. **不靠 `prop_filter`**：被丢弃的样本记在 `max_global_rejects` 上，命中上限就是 flaky。
//!    一切约束在生成端闭合（区间、位宽、`prop_flat_map` 派生长度）。
//! 3. **失败可复现**：种子固定（`PROPTEST_RNG_SEED` 可覆盖），反例语料落在 `proptest-regressions/`
//!    并入库；同一棵树两次跑必须给同一个反例。
//!
//! 环境变量只有两个，且只在本模块收口：`PROPTEST_CASES`、`PROPTEST_RNG_SEED`。取值非法时
//! **响亮失败**而不是退回默认——静默降级会让门禁假装通过。
//!
//! 第四条纪律由 `prop_discipline` 这个门盯着：**任何 crate 都不得自建 [`ProptestConfig`]**，
//! 有 `proptest` 依赖的 crate 必须也依赖本装置并真的调用它。手抄配置不会让任何测试变红，
//! 它只会让同一性质在不同 crate 里用不同的样本数与收缩预算——静默地变弱。
//! 深挖反例时用 `PROPTEST_CASES=4096 cargo test`，不必改代码。

use proptest::prelude::*;
use proptest::test_runner::RngSeed;
use std::ops::RangeInclusive;

/// 默认样本数：一条性质跑这么多次。CI 与本地同值，不设两套。
pub const DEFAULT_CASES: u32 = 256;

/// 默认固定种子：性质测试必须可复现，随机种子只由 `PROPTEST_RNG_SEED` 显式引入。
pub const DEFAULT_SEED: u64 = 0x5EED_1DEA_0000_0001;

/// 收缩上限：反例必须缩到最小，否则失败报告指向"某个大矩阵"而不是"这一项"。
pub const MAX_SHRINK_ITERS: u32 = 4_096;

/// `PROPTEST_CASES` 的解析：空/缺省取 [`DEFAULT_CASES`]，非正整数一律 panic。
///
/// `0` 被拦下是因为它会让每条性质都"零次通过"——门禁看起来是绿的，实际什么都没验。
///
/// # Panics
///
/// 取值不是正整数时 panic，消息带上原始取值。
#[must_use]
pub fn cases_from(raw: Option<&str>) -> u32 {
    match raw.map(str::trim) {
        None | Some("") => DEFAULT_CASES,
        Some(text) => match text.parse::<u32>() {
            Ok(cases) if cases > 0 => cases,
            _ => panic!("PROPTEST_CASES 必须是正整数，当前是 {text:?}"),
        },
    }
}

/// `PROPTEST_RNG_SEED` 的解析：空/缺省取 [`DEFAULT_SEED`]。
///
/// # Panics
///
/// 取值不是 `u64` 时 panic，消息带上原始取值。
#[must_use]
pub fn seed_from(raw: Option<&str>) -> u64 {
    match raw.map(str::trim) {
        None | Some("") => DEFAULT_SEED,
        Some(text) => text
            .parse::<u64>()
            .unwrap_or_else(|_| panic!("PROPTEST_RNG_SEED 必须是 u64，当前是 {text:?}")),
    }
}

/// 生效的样本数（读 `PROPTEST_CASES`）。
#[must_use]
pub fn cases() -> u32 {
    cases_from(std::env::var("PROPTEST_CASES").ok().as_deref())
}

/// 生效的种子（读 `PROPTEST_RNG_SEED`）。
#[must_use]
pub fn seed() -> u64 {
    seed_from(std::env::var("PROPTEST_RNG_SEED").ok().as_deref())
}

/// 全部性质测试共用的 `ProptestConfig`：`proptest!` 里写
/// `#![proptest_config(llm_testkit::prop::config())]`。
///
/// 未在此显式设置的字段保持 proptest 默认，且都是刻意的：`failure_persistence` 落盘（反例
/// 语料入库、下次自动重放）、`fork = false`（同进程，不吃 CI 的子进程预算）、每例 5 s 超时
/// （数值核慢于此本身就是缺陷）。
#[must_use]
pub fn config() -> ProptestConfig {
    ProptestConfig {
        cases: cases(),
        max_shrink_iters: MAX_SHRINK_ITERS,
        rng_seed: RngSeed::Fixed(seed()),
        ..ProptestConfig::default()
    }
}

/// 严格小于 1 的最大 `f32`（`1 − 2^-24`）。`v = f32::MAX` 时 `v / (1 + |v|)` 会经舍入落到
/// `1.0`，与"严格小于 `limit`"的契约不符，所以上界压到这里。
const BELOW_ONE: f32 = 1.0 - f32::EPSILON / 2.0;

/// 有界 squash：`limit · |v| / (1 + |v|)`，幅值压在 [`BELOW_ONE`]，再按 `v` 的符号还原。
///
/// 对**全部** `f32` 位模式都返回 `(−limit, limit)` 内的有限值：`±inf` 与 `NaN` 经
/// `f32::min` 的"忽略 NaN"语义落回 `BELOW_ONE`，于是生成器不必先滤掉它们。
#[must_use]
pub fn squash(limit: f32, v: f32) -> f32 {
    let magnitude = (v.abs() / (1.0 + v.abs())).min(BELOW_ONE);
    let squashed = limit * magnitude;
    if v.is_sign_negative() {
        -squashed
    } else {
        squashed
    }
}

/// 有界有限标量：`NORMAL` 平滑压进 `(−limit, limit)`。
pub fn bounded(limit: f32) -> impl Strategy<Value = f32> {
    prop::num::f32::NORMAL.prop_map(move |v| squash(limit, v))
}

/// 最常用档：`(−1, 1)`，够覆盖符号、排序与抵消，又不至于把累加推到溢出边缘。
pub fn finite() -> impl Strategy<Value = f32> {
    bounded(1.0)
}

/// `[0, 1]` 均匀：概率、权重、插值系数这类有定义域约束的输入。
pub fn unit() -> impl Strategy<Value = f32> {
    0.0f32..=1.0
}

/// 对数均匀正数：`e^k`、`k ∈ [−4, 4]`，即 `[0.018, 54.6]`。严格为正且跨四个数量级，
/// 用来抓"只在某个量级才对"的错误。
pub fn positive() -> impl Strategy<Value = f32> {
    (-4.0f32..=4.0).prop_map(f32::exp)
}

/// 宽动态范围：`±10^k`、`k ∈ [−30, 30]`。看的是溢出、下溢与灾难性抵消——
/// `finite()` 覆盖不到的失效模式全在这里。
pub fn wide() -> impl Strategy<Value = f32> {
    (-30.0f32..=30.0, any::<bool>()).prop_map(|(exp, negative)| {
        let magnitude = 10.0f32.powf(exp);
        if negative { -magnitude } else { magnitude }
    })
}

/// 定长/变长向量，元素取自 [`finite`]。
pub fn vector(len: RangeInclusive<usize>) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(finite(), len)
}

/// [`wide`] 的向量版本：数值稳定性用例专用。
pub fn wide_vector(len: RangeInclusive<usize>) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(wide(), len)
}

pub use crate::compare::all_finite;
pub use crate::rel::{assert_within_rel, within_rel};

/// 精确相等断言：逐元素 `a == b`（`±0` 视为相等，`NaN` 与他人不等）。
///
/// 用于"容差没有意义"的性质——整数累加、取反律、`pos = 0` 的恒等、量化网格落点。容差断言
/// 一律走 [`assert_within_rel`]，这里不另立第二套口径。
///
/// # Panics
///
/// 长度不同或任一项不等时 panic，消息给出首个不符项的标号与两侧取值。
#[track_caller]
pub fn assert_exact(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}：长度 {} ≠ {}",
        got.len(),
        want.len()
    );
    if let Some(index) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!("{what}：第 {index} 项 {} ≠ {}", got[index], want[index]);
    }
}
