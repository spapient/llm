use super::with_panels;

/// 同一线程两次借用拿到同一块地址：复用生效，且旧内容不会越界泄漏到新长度之外。
#[test]
fn a_second_borrow_reuses_the_same_buffer() {
    let first = with_panels(1024, |buf| buf.as_ptr() as usize);
    let second = with_panels(512, |buf| (buf.as_ptr() as usize, buf.len()));
    assert_eq!(first, second.0);
    assert_eq!(second.1, 512);
}

/// 扩容时新增段清零：面板补零列依赖这一点。
#[test]
fn growing_the_buffer_clears_the_new_tail() {
    with_panels(16, |buf| buf.fill(1.0));
    with_panels(64, |buf| {
        assert!(buf[..16].iter().all(|v| *v == 1.0) || buf[..16].iter().all(|v| *v == 0.0));
        assert!(buf[16..].iter().all(|v| *v == 0.0), "新增段必须清零");
    });
}

/// 重入：借用期间再借一次不 panic，内层拿到的是另一块。
#[test]
fn a_nested_borrow_falls_back_to_a_temporary() {
    let outer = with_panels(8, |buf| {
        buf.fill(3.0);
        let inner = with_panels(8, |inner| (inner.as_ptr() as usize, inner.to_vec()));
        assert_ne!(buf.as_ptr() as usize, inner.0, "重入必须换一块");
        assert!(inner.1.iter().all(|v| *v == 0.0));
        buf[0]
    });
    assert_eq!(outer, 3.0);
}
