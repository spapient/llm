//! `pickle` 的单元测试：栈式 pickle 虚拟机。
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

/// 一段 `_rebuild_tensor_v2` 的实参：storage key、offset、size、stride。
fn rebuild(key: &str, size: (u8, u8), stride: (u8, u8)) -> Vec<u8> {
    let mut out = vec![b'('];
    out.extend(b"\x8c\x07storage");
    out.extend(b"c__main__\nFloatStorage\n");
    out.push(b'\x8c');
    out.push(key.len() as u8);
    out.extend(key.as_bytes());
    out.push(0x87);
    out.push(b'Q');
    out.push(b'K');
    out.push(0);
    for dim in [size.0, size.1] {
        out.push(b'K');
        out.push(dim);
    }
    out.push(0x86);
    for step in [stride.0, stride.1] {
        out.push(b'K');
        out.push(step);
    }
    out.push(0x86);
    out.push(0x89);
    out.push(b'}');
    out.push(b't');
    out.push(b'R');
    out
}

#[test]
fn a_non_contiguous_tensor_is_refused_instead_of_read_as_contiguous() {
    // 2×2 的连续 stride 是 (2, 1)；(1, 2) 是转置布局，字节区间不等于「4 个元素 × 4 字节」。
    let mut pkl = b"\x80\x02".to_vec();
    pkl.extend(rebuild("0", (2, 2), (1, 2)));
    pkl.push(b'.');
    let error = Vm::run(&pkl).expect_err("非连续张量必须报错");
    assert!(error.to_string().contains("连续"), "{error}");
}

#[test]
fn a_contiguous_tensor_still_parses() {
    let mut pkl = b"\x80\x02".to_vec();
    pkl.extend(rebuild("0", (2, 2), (2, 1)));
    pkl.push(b'.');
    let tensors = Vm::run(&pkl).expect("连续布局");
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].shape(), &[2, 2]);
}

#[test]
fn a_tensor_without_a_dict_key_gets_a_distinct_positional_name() {
    let mut pkl = b"\x80\x02\x28".to_vec();
    pkl.extend(rebuild("a", (1, 1), (1, 1)));
    pkl.extend(rebuild("b", (1, 1), (1, 1)));
    pkl.push(0x86);
    pkl.push(b'.');
    let tensors = Vm::run(&pkl).expect("元组里的两个张量");
    assert_eq!(tensors.len(), 2);
    assert_ne!(tensors[0].name(), tensors[1].name());
    assert!(!tensors[0].name().is_empty());
}
