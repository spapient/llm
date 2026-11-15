use super::*;

#[test]
fn a_batch_must_match_its_shape() {
    let tokens = [1, 2, 3, 4];
    let batch = Batch::new(&tokens, 2, 2).expect("2 x 2");
    assert_eq!(batch.tokens(), tokens);
    assert_eq!(batch.shape().dims(), [2, 2]);
    assert_eq!(
        Batch::new(&[1, 2, 3], 2, 2).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(Batch::new(&[], 0, 2).unwrap_err().code(), Code::Invalid);
}

#[test]
fn a_batch_must_reject_an_overflowing_shape() {
    let tokens = [1, 2];
    // 溢出必须报错，而不是回绕成一个恰好等于 `tokens.len()` 的形状：后者会让一个非法的
    // 2 x usize::MAX 被当成合法批，形状从此不再可信。
    assert_eq!(
        Batch::new(&tokens, usize::MAX, 2).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        Batch::new(&tokens, 2, usize::MAX).unwrap_err().code(),
        Code::Invalid
    );
}

#[test]
fn a_batch_borrows_the_callers_buffer() {
    let tokens = [7, 8, 9, 10, 11, 12];
    for (rows, cols) in [(1, 6), (2, 3), (3, 2), (6, 1)] {
        let batch = Batch::new(&tokens, rows, cols).expect("形状合法");
        assert_eq!(
            batch.tokens().as_ptr(),
            tokens.as_ptr(),
            "批必须借用调用方的缓冲而不是拷贝"
        );
        assert_eq!(rows * cols, batch.tokens().len());
    }
}
