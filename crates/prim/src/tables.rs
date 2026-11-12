//! 构建期生成的表：片段（`BUILTIN`）、见证源码（`WITNESS`）与设备镜像（`DEVICES`）。
//!
//! 内容由 [`build.rs`](../build.rs) 扫描 `<vendor>/<prim>.<ext>` 与 `<vendor>/witness.<ext>` 得到，
//! 写进 `$OUT_DIR/tables.rs` 后在这里 `include!`。本模块只有数据：类型定义在 [`crate::device`] 与
//! [`crate::Builtin`]，所以生成代码不进所有权表，"一个类型只有一个主人"照旧成立。

include!(concat!(env!("OUT_DIR"), "/tables.rs"));
