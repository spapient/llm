    use super::*;

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
