//! `Allocator` / `CpuAllocator` 的单元测试：分配、清零、可写、零长度与拷贝长度校验。

use super::*;

#[test]
fn cpu_allocator_allocates_zeroed_buf() {
    let mut alloc = CpuAllocator::new();
    let buf = alloc.alloc(5).expect("分配");
    assert_eq!(buf.len(), 5);
    assert_eq!(buf.bytes(), &[0; 5]);
    alloc.free(buf);
}

#[test]
fn allocator_buf_is_writable() {
    let mut alloc = CpuAllocator::new();
    let mut buf = alloc.alloc(3).expect("分配");
    buf.bytes_mut().copy_from_slice(&[1, 2, 3]);
    assert_eq!(buf.bytes(), &[1, 2, 3]);
    let bytes = buf.into_bytes();
    assert_eq!(bytes, vec![1, 2, 3]);
}

#[test]
fn zero_allocation_yields_an_empty_buf() {
    let mut alloc = CpuAllocator::new();
    let buf = alloc.alloc(0).expect("零长度");
    assert!(buf.is_empty());
}

#[test]
fn copy_into_matches_lengths_and_rejects_mismatch() {
    let alloc = CpuAllocator::new();
    let mut dst = vec![0u8; 4];
    alloc.copy_into(&[1, 2, 3, 4], &mut dst).expect("等长拷贝");
    assert_eq!(dst, [1, 2, 3, 4]);
    let err = alloc.copy_into(&[1, 2], &mut dst).unwrap_err();
    assert_eq!(err.code(), Code::Invalid);
}

#[test]
fn an_unaffordable_allocation_is_an_error_not_an_abort() {
    let mut alloc = CpuAllocator::new();
    let error = alloc.alloc(usize::MAX).expect_err("地址空间装不下");
    assert_eq!(error.code(), Code::Invalid);
}
