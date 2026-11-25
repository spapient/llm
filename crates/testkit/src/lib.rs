//! 测试约定：浮点对拍、真机门控、临时目录、随机数、位级 CRC 参考。**只作 dev-dependency**。
//!
//! **拥有**：
//! - `assert_eq`：结构化相等断言，失败时给行级 diff；判定仍是 `PartialEq`，
//!   不引入浮点容差（浮点对拍归 [`compare`]，字节快照归 [`hash`]）；
//! - [`compare`]：逐元素与方向判据 `approx_eq`、`close`、`assert_close`、`all_finite`、
//!   `cosine_similarity`、`COSINE_GUARD`，以及整批相对误差口径 `max_rel_err`、`Scale`、
//!   `assert_within_rel`、`assert_rel_with`（失败读数与最差元素的载体留在 crate 内部）；
//! - [`fixture`]：格式夹具 `safetensors_bytes`、`pack_codes`；
//! - [`gating`]：真机门控 `device_enabled`、`device_or_skip`、`skip`、`skip_none`、`skip_count`、
//!   `skipped_reasons`；
//! - [`hash`]：位级 CRC-32 参考 `crc32_bitwise` 与 `assert_table_matches_bitwise`；
//! - [`rng`]：确定性伪随机 `Rng` 与确定性样本 `sample`；
//! - [`scratch`]：`scratch_dir`、`scratch_clean`；
//! - [`witness`]：`witness::run`、`Call`、`Reading`、`MAX_OUT`——设备 crate 的见证源码在本机
//!   编译执行并与期望值对拍；
//! - [`device`]、[`evidence`]：`Contract`、`Provision`、`Witness` 与 `check_expectations`、
//!   `check_witnesses`、`check_self_contained`、`TOLERANCE`——六台设备共用的片段层契约：签名、
//!   见证入口、档案供给三者必须闭合，且每条见证在本机算得出档案写的值；
//! - [`suite`]：`device_suite!` 把九条判据摊成设备模块的 `#[test]` 函数，于是"一台设备一遍"
//!   不必复制九次。
//!
//! **不拥有**：任何生产逻辑——它是测试夹具，不进任何二进制。
//! **依赖**：`proptest`（仅 [`prop`]：性质测试装置）——夹具不拉被测 crate，复用测试库则比在
//! 四个成员里各写一份配置更省，且 testkit 只作 dev-dependency，此依赖不进任何二进制。
//! **被依赖**：各成员的 `[dev-dependencies]`。
//!
//! 这里的真机门控变量是**测试门控，不是产品配置**：它不进 `cli::settings` 的键表，
//! 生产代码不得读取它。真机用例的形态固定为"默认跳过、显式开门才跑"：
//!
//! ```
//! if llm_testkit::device_enabled("nvidia") {
//!     // 需要 nvcc + 真卡的用例
//! }
//! ```
//!
//! 前提不满足的用例必须走 [`skip`] 登记，禁止裸 `return` 空转：跳过原因进进程内计数与
//! 结构化日志，设置 `LLM_TEST_SKIP_DIR` 时还会追加到该目录的 `skips.log`，供 CI 汇总。

#![deny(unsafe_code)]

pub mod compare;
pub mod device;
pub mod evidence;
pub mod fixture;
pub mod gating;
pub mod hash;
/// 标量参考数学与数值对拍共用 `compare` 的实现，这里保留原公共路径。
pub mod math {
    pub use crate::compare::{dot, mean, variance};
}
pub mod prop;
pub mod rng;
pub mod scratch;
pub mod suite;
pub mod witness;

// 表驱动宏直接定义在 crate 根，避免仅为宏声明单开口袋文件。

/// 逐条执行命名用例：`cases!(闭包; "用例名" => 输入, …)`。
///
/// 表驱动配置化：用例是数据，判据只有一份。闭包接收 `(用例名, 输入)` 两个参数，每条表项顺序
/// 调用一次；用例名应写进断言消息，失败时即可定位到具体表项。
///
/// ```
/// fn double(x: i32) -> i32 { x * 2 }
/// llm_testkit::cases!(|name, (input, want): (i32, i32)| {
///     assert_eq!(double(input), want, "{name}");
/// };
/// "零" => (0, 0),
/// "正数" => (3, 6),
/// "负数" => (-2, -4),
/// );
/// ```
#[macro_export]
macro_rules! cases {
    ($run:expr; $($name:literal => $input:expr),+ $(,)?) => {{
        // `&mut` 绑定同时容纳 `Fn` 与 `FnMut` 闭包（如捕获 `&mut` 计数器），无需可变性警告豁免。
        let run = &mut $run;
        $( run($name, $input); )+
    }};
}

/// 把一组同形用例各生成一个独立 `#[test]`：`test_cases!(模块名 { 测试名 => 输入, … }, 判据)`。
///
/// 与 [`cases!`] 的差别是**一条表项一个测试函数**：测试报告里每条用例单独计数、单独并行，
/// 单条失败不拖累同组其他用例的执行。判据闭包接收 `(输入)`，测试名即表项标识。
///
/// ```
/// llm_testkit::test_cases!(parity {
///     even_zero => 0,
///     even_two => 2,
///     even_ten => 10,
/// }, |input: i32| {
///     assert_eq!(input % 2, 0);
/// });
/// ```
#[macro_export]
macro_rules! test_cases {
    ($module:ident { $($name:ident => $input:expr),+ $(,)? }, $run:expr) => {
        #[cfg(test)]
        mod $module {
            // 判据闭包常要调用外层的装配 helper；`use super::*` 把它们带进生成模块。
            #[allow(unused_imports)]
            use super::*;
            $(
                #[test]
                fn $name() {
                    let run = $run;
                    run($input);
                }
            )+
        }
    };
}

#[cfg(test)]
mod cases {
    #[test]
    fn cases_run_every_row_and_pass_the_case_name() {
        let mut seen = Vec::new();
        crate::cases!(|name, input: i32| {
            seen.push((name, input * 2));
        };
        "一" => 1,
        "二" => 2,
        "三" => 3,
        );
        assert_eq!(seen, [("一", 2), ("二", 4), ("三", 6)]);
    }

    crate::test_cases!(generated {
        zero_is_even => 0,
        two_is_even => 2,
        four_is_even => 4,
    }, |input: i32| {
        assert_eq!(input % 2, 0, "{input} 应为偶数");
    });

    // 第二组：钉住「同一条判据被多个生成的测试复用」，含负值输入。
    crate::test_cases!(signs {
        positive => 1,
        negative => -1,
    }, |input: i32| {
        assert_ne!(input, 0);
    });

    /// 外层装配 helper：验证生成模块经 `use super::*` 能调到它。
    fn triple(x: i32) -> i32 {
        x * 3
    }

    crate::test_cases!(uses_outer_helper {
        one => 1,
        five => 5,
    }, |input: i32| {
        assert_eq!(triple(input), input * 3);
    });
}

// 工作区级的跨层对拍与契约门禁。它们只靠 `dev-dependencies` 里的被测层编译，所以每条都在
// `cfg(test)` 之下——`use llm_testkit` 的运行时构建看不到其中任何一行。
#[cfg(test)]
mod architecture;
#[cfg(test)]
mod boundary;
#[cfg(test)]
mod contracts;
#[cfg(test)]
mod e2e;
// 三条外部 `#[path]` 用户的取值来源（`cutile` 单测、`device` 单测、`device` 的 schedule 集成
// 测试）已在 `crates/testkit/tests/kernel.rs`：它只被那三处编译，`src/` 里不留测试夹具。
#[cfg(test)]
mod layer;
#[cfg(test)]
mod mirror;
#[cfg(test)]
mod ownership;
// 性质测试装置的自测：环境变量收口、生成器值域、精确相等断言的边界。
#[cfg(test)]
#[path = "../tests/prop_config.rs"]
mod prop_config;
// 性质测试的工作区纪律：任何 crate 都不得自建 `ProptestConfig`（唯一来源是 `prop`）。
#[cfg(test)]
#[path = "../tests/prop_discipline.rs"]
mod prop_discipline;
// L3 合成律的跨层对拍：只收「一层算不出来」的事实——`bound` 量真实内核上的界是否罩住实测误差，
// `ruling` 量真实容器的逐位往返，`measure` 是两者共用的夹具（定理侧递推由 `weights::chain` 自测）。
#[cfg(test)]
#[path = "../tests/chain_bound.rs"]
mod chain_bound;
#[cfg(test)]
#[path = "../tests/chain_measure.rs"]
mod chain_measure;
#[cfg(test)]
#[path = "../tests/chain_ruling.rs"]
mod chain_ruling;

pub use compare::{
    COSINE_GUARD, Scale, all_finite, approx_eq, assert_close, assert_near, assert_rel_with,
    assert_vec_near, assert_within_rel, close, cosine_similarity, max_rel_err, mean_rel_err,
    mean_rel_err_with,
};
pub use fixture::{assert_rejects_length_mismatch, pack_codes, safetensors_bytes};
pub use gating::{device_enabled, device_or_skip, skip, skip_count, skip_none, skipped_reasons};
pub use hash::{assert_table_matches_bitwise, crc32_bitwise};
pub use math::{dot, mean, variance};
pub use rng::{Rng, sample, sample_with};
pub use scratch::{ScratchDir, scratch_clean, scratch_dir};
pub use similar_asserts::assert_eq;
