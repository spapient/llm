//! `ascend` 片段层与档案的对账：左边是 `build.rs` 编出来的三张表，右边是
//! `crates/prof/devices/ascend.toml` 声称具备的能力，中间的任何错位都是缺陷。
//!
//! 判据本身在 `testkit::device`（六台设备共用一套），用例由 `testkit::device_suite!`
//! 摊开，这里只交出设备行与 P0 闭集。语言门不在这里：`ccec` 只随 CANN 安装，昇腾前端的
//! 解析由 `cutile::gate` 单独登记跳过。

use super::*;

/// P0 闭集：`crates/op/ops/*.k` 真的会 `@` 到的九个原语。片上同步的两条在这台设备上是
/// 空操作，但名字必须留着——模板按名字取片段，缺一条就拼不出内核。
const P0: [&str; 9] = [
    "bar",
    "commit_wait",
    "cvt",
    "dot",
    "load",
    "load_async",
    "quant",
    "reduce",
    "store",
];

testkit::device_suite!(P0);
