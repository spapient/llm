//! 黑盒契约：`wire` 的公开面只有"任意字节 → 结论"，因此这里只经公共 API 断言对外承诺。
//!
//! 内联测试覆盖编码器内部算术（分块、位宽、溢出）；本文件回答另一个问题：把 `leb` / `crc` /
//! `sink` 三个模块当作第三方依赖使用时，长度承诺、规范性与错误类别是否真的成立。唯一实现细节
//! 依赖是 `Count` 与 `Vec` 必须走同一条编码路径——那正是 `Sink` 抽象存在的理由。

use llm_wire::crc;
use llm_wire::leb::{Cursor, write};
use llm_wire::sink::{Count, FileSink, Measured, Sealed, Sink, SliceSink};

#[test]
fn counting_and_writing_agree_byte_for_byte() {
    let values = [0u64, 7, 128, u32::MAX as u64, u64::MAX];
    let mut counted = Count::default();
    for value in values {
        write(&mut counted, value);
    }
    let mut written: Vec<u8> = Vec::new();
    for value in values {
        write(&mut written, value);
    }
    assert_eq!(
        counted.0,
        written.len(),
        "算长度与实际写出必须走同一条路径，否则长度承诺会漂移"
    );
}

#[test]
fn a_sealed_sink_covers_every_byte_it_passes_through() {
    let payload: &[u8] = b"frame-with-trailer";
    let mut out: Vec<u8> = Vec::new();
    let checksum = {
        let mut sealed = Sealed::new(&mut out);
        sealed.put(payload);
        sealed.checksum()
    };
    assert_eq!(
        checksum,
        crc::crc32(payload),
        "密封的校验范围就是它往下游放行的每个字节"
    );
    assert_eq!(out, payload, "密封只更新校验，不改字节");
}

#[test]
fn a_slice_sink_flags_overflow_instead_of_truncating() {
    let mut buffer = [0u8; 4];
    let mut sink = SliceSink::new(&mut buffer);
    sink.put(b"ab");
    sink.put(b"cdefg");
    assert!(sink.overflow(), "放不下必须置标志，不能静默截断");
    assert_eq!(
        sink.written(),
        2,
        "溢出后不再写入，已写位置停在上一次成功的长度"
    );
    assert_eq!(buffer[..2], *b"ab", "越界位置必须保持原值");
}

#[test]
fn a_measured_sink_reports_the_same_length_as_its_downstream() {
    let mut inner: Vec<u8> = Vec::new();
    let (len, empty) = {
        let mut measured = Measured::new(&mut inner);
        measured.put(b"xyz");
        (measured.len(), measured.is_empty())
    };
    assert_eq!(len, 3);
    assert!(!empty);
    assert_eq!(inner.len(), 3);
}

#[test]
fn a_file_sink_reports_the_first_write_error_instead_of_panicking() {
    /// 一个一次也写不进去的写入器：模拟磁盘满或已关闭的句柄。
    struct Failing;

    impl std::io::Write for Failing {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("磁盘满"))
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let mut sink = FileSink::new(Failing);
    sink.put(b"abc");
    sink.put(b"def");
    assert_eq!(sink.written(), 0, "写失败不计入已写字节");
    let error = sink.take_error().expect("第一次失败必须留下记录");
    assert_eq!(error.to_string(), "磁盘满");
    assert!(sink.take_error().is_none(), "错误只留第一次，取走即清");
}

#[test]
fn a_cursor_consumes_exactly_what_it_was_asked_for() {
    let mut bytes: Vec<u8> = Vec::new();
    write(&mut bytes, 300u64);
    write(&mut bytes, 7u64);
    let mut cursor = Cursor::new(&bytes);
    assert_eq!(cursor.read::<u32>().expect("规范编码"), 300);
    assert_eq!(cursor.read::<u8>().expect("规范编码"), 7);
    assert_eq!(cursor.remaining().len(), 0, "取值即消耗，剩下零字节");
    assert!(
        cursor.read::<u8>().is_err(),
        "读尽之后再读必须报错，而不是返回默认值"
    );
}

#[test]
fn a_non_canonical_encoding_is_rejected_with_a_category() {
    // 0 的规范编码是单字节 0x00；补一个零字节就成了过长表示。
    let error = Cursor::new(&[0x80, 0x00])
        .read::<u64>()
        .expect_err("过长表示必须拒绝");
    assert_eq!(error.code().as_str(), "invalid", "错误类别是对外契约");
    assert!(!error.message().trim().is_empty(), "拒绝必须说清为什么");
}
