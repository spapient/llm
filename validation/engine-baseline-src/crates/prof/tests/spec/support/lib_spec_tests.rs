//! 原先内联在 `src/lib.rs` 的测试。
//!
//! 用 `#[path]` 挂回原模块，模块树位置不变，故 `super::`/`crate::` 路径照旧。

use super::same_bytes;

#[test]
fn same_bytes_compares_content_not_identity() {
    assert!(same_bytes("gemv_i4", "gemv_i4"));
    assert!(!same_bytes("gemv_i4", "gemv_i8"));
    assert!(!same_bytes("gemv", "gemv_i4"));
    assert!(same_bytes("", ""));
}
