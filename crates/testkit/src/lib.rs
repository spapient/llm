//! 测试约定：浮点对拍、真机门控、临时目录、随机数、位级 CRC 参考。**只作 dev-dependency**。
//!
//! **拥有**：`approx_eq`、`close`、`all_finite`、`cosine_similarity`、`COSINE_GUARD`
//! （[`compare`]）；`device_enabled`、`device_or_skip`、`skip`、`skip_none`、`skip_count`、
//! `skipped_reasons`（[`gating`]）；`scratch_dir`、`scratch_clean`（[`scratch`]）；
//! [`Rng`]（[`rng`]）；`crc32_bitwise`（[`hash`]）。**不拥有**：任何生产逻辑——它是测试
//! 夹具，不进任何二进制。**依赖**：无（夹具不该把被测 crate 拉进来）。
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
pub mod gating;
pub mod hash;
pub mod rng;
pub mod scratch;

pub use compare::{COSINE_GUARD, all_finite, approx_eq, close, cosine_similarity};
pub use gating::{device_enabled, device_or_skip, skip, skip_count, skip_none, skipped_reasons};
pub use hash::{assert_table_matches_bitwise, crc32_bitwise};
pub use rng::Rng;
pub use scratch::{scratch_clean, scratch_dir};
