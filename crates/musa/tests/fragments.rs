//! `musa` 片段层与档案的对账：左边是 `build.rs` 编出来的三张表，右边是
//! `crates/prof/devices/musa.toml` 声称具备的能力，中间的任何错位都是缺陷。
//!
//! 判据本身在 `testkit::device`（六台设备共用一套），用例由 `testkit::device_suite!`
//! 摊开，这里只交出设备行。输出语言是 MUSA。

use super::*;

testkit::device_suite!();
