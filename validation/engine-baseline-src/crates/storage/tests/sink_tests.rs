use super::*;
use proptest::prelude::*;
use testkit::prop;

#[test]
fn count_matches_vec_len() {
    let payload = b"hello world";
    let mut count = Count::default();
    let mut bytes = Vec::new();
    count.put(payload);
    bytes.put(payload);
    assert_eq!(count.0, bytes.len());
}

#[test]
fn slice_sink_writes_within_capacity() {
    let mut buf = [0u8; 8];
    let mut sink = SliceSink::new(&mut buf);
    sink.put(b"abc");
    sink.put(b"def");
    assert_eq!(sink.written(), 6);
    assert!(!sink.overflow());
    assert_eq!(&buf[..6], b"abcdef");
}

#[test]
fn slice_sink_flags_overflow() {
    let mut buf = [0u8; 3];
    let mut sink = SliceSink::new(&mut buf);
    sink.put(b"abcd");
    assert!(sink.overflow());
}

#[test]
fn sealed_matches_standalone_crc() {
    let mut downstream = Vec::new();
    let mut sealed = Sealed::new(&mut downstream);
    sealed.put(b"checksum this");
    sealed.put(b" and this");
    assert_eq!(sealed.checksum(), crc::crc32(b"checksum this and this"));
    assert_eq!(downstream, b"checksum this and this");
}

#[test]
fn file_sink_streams_to_writer() {
    let mut target = Vec::new();
    let mut sink = FileSink::new(&mut target);
    sink.put(b"streamed");
    assert!(sink.error().is_none());
    assert_eq!(sink.written(), 8);
    let back = sink.into_inner().expect("flush");
    assert_eq!(*back, b"streamed");
}

proptest! {
    #![proptest_config(prop::config())]

    /// 固定缓冲只报告溢出、不越界：溢出是**整块拒绝**，已经写下的那段字节不动，写出位置之外保持初始零。
    #[test]
    fn a_slice_sink_refuses_the_whole_chunk_that_does_not_fit(
        capacity in 0usize..=8,
        payload in proptest::collection::vec(any::<u8>(), 0..=16),
    ) {
        let mut buf = vec![0u8; capacity];
        let (written, overflow) = {
            let mut sink = SliceSink::new(&mut buf);
            sink.put(&payload);
            (sink.written(), sink.overflow())
        };
        prop_assert_eq!(overflow, payload.len() > capacity);
        if overflow {
            prop_assert_eq!(written, 0, "整块拒绝时不该写下一半");
        } else {
            prop_assert_eq!(written, payload.len());
            prop_assert_eq!(&buf[..written], payload.as_slice());
        }
        prop_assert!(buf[written..].iter().all(|byte| *byte == 0));
    }
}
