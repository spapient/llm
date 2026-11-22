use super::*;

#[test]
fn a_batch_must_match_its_shape() {
    let tokens = [1, 2, 3, 4];
    let batch = Batch::new(&tokens, 2, 2).expect("2 x 2");
    assert_eq!(batch.tokens(), tokens);
    assert_eq!((batch.rows(), batch.cols()), (2, 2));
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
fn a_batch_borrows_the_callers_buf() {
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

#[test]
fn a_chunks_stream_rejects_bad_geometry() {
    let tokens = [1, 2, 3, 4];
    for (rows, cols) in [(1, 0), (0, 1), (usize::MAX, 2)] {
        assert_eq!(
            Chunks::new(&tokens, rows, cols).unwrap_err().code(),
            Code::Invalid,
            "rows = {rows}、cols = {cols} 必须在构造期被拒"
        );
    }
}

#[test]
fn both_constructors_read_geometry_in_the_same_order() {
    let tokens: [u32; 6] = [0, 1, 2, 3, 4, 5];
    // `(2, 3)` 在两个构造函数里都必须表示「2 行 3 列」。此前 `Chunks::new` 收的是 `(cols, rows)`，
    // 于是同一对数字换个构造函数就是一次静默转置——而两种顺序都能通过校验，没有一条用例会红。
    let batch = Batch::new(&tokens, 2, 3).expect("2 x 3");
    let mut dataset = Chunks::new(&tokens, 2, 3).expect("2 x 3");
    let first = dataset.next_batch().expect("6 个 token 恰好是一批");
    assert_eq!(
        (first.rows(), first.cols()),
        (batch.rows(), batch.cols()),
        "两个构造函数必须读出同一形状"
    );
    assert_eq!(first.tokens(), batch.tokens());
    assert_eq!(
        (first.rows(), first.cols()),
        (2, 3),
        "顺序反了会得到 (3, 2)"
    );
    assert_eq!(dataset.remaining_batches(), 0);
}

#[test]
fn an_empty_stream_yields_no_batch() {
    let mut dataset = Chunks::new(&[], 1, 2).expect("空流也是合法输入");
    assert_eq!(dataset.cols(), 2);
    assert_eq!(dataset.batches(), 0);
    assert_eq!(dataset.remaining_batches(), 0);
    assert!(dataset.next_batch().is_none());
}

#[test]
fn chunks_drop_the_partial_tail() {
    let tokens = [0, 1, 2, 3, 4, 5, 6];
    let mut dataset = Chunks::new(&tokens, 1, 2).expect("数据");
    assert_eq!(dataset.batches(), 3, "7 个 token 按 2 个一行只能凑 3 批");
    assert_eq!(dataset.remaining_batches(), 3);
    for expected in [&tokens[0..2], &tokens[2..4], &tokens[4..6]] {
        assert_eq!(dataset.next_batch().expect("整批").tokens(), expected);
    }
    assert!(
        dataset.next_batch().is_none(),
        "只剩 1 个 token，凑不满一行"
    );
    assert_eq!(dataset.remaining_batches(), 0);
}

#[test]
fn consuming_advances_both_counters() {
    let tokens: [u32; 6] = [0, 1, 2, 3, 4, 5];
    let mut dataset = Chunks::new(&tokens, 1, 2).expect("数据");
    assert_eq!(
        (
            dataset.cols(),
            dataset.batches(),
            dataset.remaining_batches()
        ),
        (2, 3, 3)
    );
    let batch = dataset.next_batch().expect("第一批");
    assert_eq!(batch.tokens(), [0, 1]);
    assert_eq!(dataset.remaining_batches(), 2);
    assert_eq!(
        (
            batch.rows(),
            batch.cols(),
            dataset.batches(),
            dataset.cols()
        ),
        (1, 2, 3, 2),
        "每批几何、批总数、列宽都不随消费推进"
    );
}

#[test]
fn two_batches_can_be_held_at_once() {
    let tokens: [u32; 6] = [0, 1, 2, 3, 4, 5];
    let mut dataset = Chunks::new(&tokens, 1, 2).expect("数据");
    let first = dataset.next_batch().expect("第一批");
    let second = dataset.next_batch().expect("第二批");
    // 两批的寿命来自 token 缓冲而不是数据集，所以可以同时存活且不重叠。
    assert_eq!(first.tokens(), [0, 1]);
    assert_eq!(second.tokens(), [2, 3]);
    assert_eq!(
        first.tokens().as_ptr().wrapping_add(first.tokens().len()),
        second.tokens().as_ptr(),
        "两批必须落在缓冲里相邻但不重叠的位置"
    );
}

#[test]
fn iterating_yields_exactly_the_declared_batches() {
    let tokens: [u32; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let mut dataset = Chunks::new(&tokens, 1, 3).expect("数据");
    assert_eq!(dataset.len(), 3, "`len()` 就是还没取走的批数");
    assert_eq!(dataset.size_hint(), (3, Some(3)));
    let collected: Vec<_> = dataset.by_ref().collect();
    assert_eq!(collected.len(), 3, "迭代条数必须等于 `batches()`");
    assert_eq!(collected[2].tokens(), [6, 7, 8]);
    assert_eq!(dataset.len(), 0);
    // 取空之后 `len()` 与两种取法都不再变化（`FusedIterator`）。
    assert!(dataset.next().is_none());
    assert!(dataset.next_batch().is_none());
    assert_eq!(dataset.size_hint(), (0, Some(0)));
}

#[test]
fn next_and_next_batch_advance_the_same_cursor() {
    let tokens: [u32; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let mut dataset = Chunks::new(&tokens, 1, 3).expect("数据");
    assert_eq!(dataset.next().expect("第一条").tokens(), [0, 1, 2]);
    assert_eq!(dataset.next_batch().expect("第二条").tokens(), [3, 4, 5]);
    assert_eq!(
        dataset.by_ref().count(),
        1,
        "两种取法共用同一个游标，不会跳批"
    );
    assert_eq!(dataset.remaining_batches(), 0);
}

#[test]
fn the_trait_object_reports_and_yields_the_same_stream() {
    let tokens: [u32; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let mut chunks = Chunks::new(&tokens, 1, 3).expect("数据");
    // 走 trait 对象路径：三个计数与取批都由 trait 分发，验证对象安全契约。
    let ds: &mut dyn Dataset<'_> = &mut chunks;
    assert_eq!(ds.cols(), 3);
    assert_eq!(ds.batches(), 3);
    assert_eq!(ds.remaining_batches(), 3);
    for expected in [&tokens[0..3], &tokens[3..6], &tokens[6..9]] {
        assert_eq!(ds.next_batch().expect("整批").tokens(), expected);
    }
    assert_eq!(ds.remaining_batches(), 0);
    assert!(ds.next_batch().is_none(), "取空后 trait 路径同样终止");
}
