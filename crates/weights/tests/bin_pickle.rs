//! `bin::pickle` 的单元测试：栈式 pickle 虚拟机。

use super::*;

#[test]
fn an_odd_setitems_is_an_error_not_a_panic() {
    let error = Vm::run(b"}(K\x01u.").expect_err("奇数项必须报错");
    assert!(error.to_string().contains("奇数"), "{error}");
}

#[test]
fn a_huge_memo_index_is_rejected_not_resized() {
    let error = Vm::run(b"K\x01r\xff\xff\xff\xff.").expect_err("memo 索引必须报错");
    assert!(error.to_string().contains("memo"), "{error}");
}
