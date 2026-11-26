//! `safetensors` 模块的单元测试。
use super::*;

/// 按 safetensors 的布局拼一个文件：8 字节长度前缀 + 头部 + 数据段。
fn file(json: &str, data: &[u8]) -> Vec<u8> {
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();

    bytes.extend_from_slice(json.as_bytes());

    bytes.extend_from_slice(data);

    bytes
}

testkit::test_cases! {
fn two_entries_are_read_back_byte_for_byte() {
    let json = r#"{"a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},

               "b": {"dtype": "F16", "shape": [4], "data_offsets": [24, 32]}}"#;

    let data: Vec<u8> = (0..32).collect();

    let bytes = file(json, &data);

    let archive = Archive::parse(&bytes).expect("合法");

    let names: Vec<&str> = archive.header().entries().iter().map(Entry::name).collect();

    assert_eq!(names, ["a", "b"], "条目按数据段偏移排序");

    assert_eq!(archive.payload_bytes(), 32);

    let a = archive.header().get("a").expect("条目 a");

    assert_eq!(archive.read(a).expect("可读"), &data[..24]);

    let b = archive.header().get("b").expect("条目 b");

    assert_eq!(archive.read(b).expect("可读"), &data[24..]);
}

fn metadata_is_skipped_but_must_still_be_an_object() {
    let json = r#"{"__metadata__": {"format": "pt"}, "w": {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]}}"#;

    let bytes = file(json, &[1, 2]);

    let archive = Archive::parse(&bytes).expect("合法");

    assert_eq!(archive.header().entries().len(), 1);

    let broken =
        r#"{"__metadata__": "pt", "w": {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]}}"#;

    let error = Archive::parse(&file(broken, &[1, 2])).expect_err("元数据不是对象");

    assert!(error.to_string().contains("必须是对象"), "{error}");
}

fn an_unknown_dtype_is_unsupported_rather_than_guessed() {
    let json = r#"{"w": {"dtype": "I16", "shape": [1], "data_offsets": [0, 2]}}"#;

    let error = Archive::parse(&file(json, &[1, 2])).expect_err("I16 不在闭集");

    assert_eq!(error.code(), Code::Unsupported);

    assert!(error.to_string().contains("I16"), "{error}");

    assert!(
        error.to_string().contains("F32"),
        "错误里要列出闭集：{error}"
    );
}

fn an_unsigned_byte_entry_keeps_the_unsigned_eight_bit_type() {
    // 真实的 4 位量化权重存成 `U8`（两个 nibble 一个字节）：符号性保留，不与逐元素 `I8` 混同。

    let json = r#"{"w": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}"#;

    let bytes = file(json, &[1, 2]);

    let archive = Archive::parse(&bytes).expect("U8 是 8 位元素");

    let entry = &archive.header().entries()[0];

    assert_eq!(entry.dtype(), DType::U8);

    assert_eq!(archive.read(entry).expect("可读"), &[1, 2]);
}

fn a_declared_interval_must_match_the_shape() {
    let json = r#"{"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 16]}}"#;

    let error = Archive::parse(&file(json, &[0; 16])).expect_err("少了 8 字节");

    assert_eq!(error.code(), Code::Invalid);

    assert!(error.to_string().contains("推出 24 字节"), "{error}");
}

fn unknown_and_missing_fields_are_rejected() {
    let unknown =
        r#"{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4], "layout": "row"}}"#;

    let error = Archive::parse(&file(unknown, &[0; 4])).expect_err("未知字段");

    assert!(error.to_string().contains("未知字段 `layout`"), "{error}");

    let missing = r#"{"w": {"dtype": "F32", "shape": [1]}}"#;

    let error = Archive::parse(&file(missing, &[])).expect_err("缺字段");

    assert!(
        error.to_string().contains("缺字段 `data_offsets`"),
        "{error}"
    );
}

fn a_data_section_that_does_not_match_the_header_is_rejected() {
    let json = r#"{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}"#;

    for data in [&[0u8; 0][..], &[0u8; 5][..]] {
        let error = Archive::parse(&file(json, data)).expect_err("字节数不符");

        assert!(error.to_string().contains("数据段应为 4 字节"), "{error}");
    }
}

fn a_broken_length_prefix_is_rejected() {
    assert!(Archive::parse(&[0u8; 4]).is_err(), "不足 8 字节");

    let mut oversized = 4096u64.to_le_bytes().to_vec();

    oversized.extend_from_slice(b"{}");

    let error = Archive::parse(&oversized).expect_err("头部超长");

    assert!(error.to_string().contains("超出文件长度"), "{error}");

    assert!(Archive::parse(&file("[]", &[])).is_err(), "顶层必须是对象");
}

fn overlapping_and_reversed_intervals_are_rejected() {
    let overlap = r#"{"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},

                  "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}}"#;

    let error = Archive::parse(&file(overlap, &[0; 12])).expect_err("区间重叠");

    assert!(error.to_string().contains("重叠"), "{error}");

    let reversed = r#"{"w": {"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}}"#;

    let error = Archive::parse(&file(reversed, &[])).expect_err("起点晚于终点");

    assert!(error.to_string().contains("晚于"), "{error}");
}

fn an_empty_archive_is_a_valid_zero_byte_file() {
    let bytes = file("{}", &[]);

    let archive = Archive::parse(&bytes).expect("空头部合法");

    assert!(archive.header().entries().is_empty());

    assert_eq!(archive.payload_bytes(), 0);

    assert!(archive.data().is_empty());
}
}

fn path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("llm-weights-{}-{name}", std::process::id()))
}

/// 落一个数据段长度为 `data_len` 的文件：数据段用 `set_len` 拉开（稀疏文件，不占内存）。
fn sparse(name: &str, json: &str, data_len: usize) -> std::path::PathBuf {
    use std::io::Write as _;

    let target = path(name);

    let mut file = std::fs::File::create(&target).expect("临时文件可写");

    file.write_all(&(json.len() as u64).to_le_bytes())
        .expect("写长度前缀");

    file.write_all(json.as_bytes()).expect("写头部");

    file.set_len(8 + json.len() as u64 + data_len as u64)
        .expect("拉开数据段");

    target
}

testkit::test_cases! {
fn a_gigabyte_of_weights_is_opened_by_reading_only_the_header() {
    // 1 GiB 数据段 + 尾部一个 4 字节条目：打开只读前缀与头部，读小条目只读那 4 字节。

    let json = r#"{"big": {"dtype": "F32", "shape": [268435456], "data_offsets": [0, 1073741824]},

               "__metadata__": {"format": "pt"},

               "tail": {"dtype": "I8", "shape": [4], "data_offsets": [1073741824, 1073741828]}}"#;

    let target = sparse("huge.safetensors", json, 1_073_741_828);

    let file = File::open(&target).expect("头部自证合法");

    assert_eq!(file.header_len(), 8 + json.len(), "只读了前缀与头部");

    assert_eq!(file.file_len(), 8 + json.len() + 1_073_741_828);

    assert_eq!(file.header().entries().len(), 2, "`__metadata__` 不是条目");

    assert_eq!(file.header().data_bytes(), 1_073_741_828);

    let tail = file.header().get("tail").expect("条目 tail");

    assert_eq!(Loader::load(&file, tail).expect("可读"), vec![0u8; 4]);

    // 缓冲区不够时先在长度上拒绝，不会去磁盘上搬 1 GiB。

    let big = file.header().get("big").expect("条目 big");

    let mut short = [0u8; 4];

    let error = file.copy_into(big, &mut short).expect_err("缓冲区太小");

    assert_eq!(error.code(), Code::Invalid);

    assert!(error.to_string().contains("缓冲区 4 字节"), "{error}");

    let _ = std::fs::remove_file(target);
}

fn a_truncated_or_padded_file_is_rejected_when_it_is_opened() {
    let json = r#"{"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}"#;

    let truncated = sparse("truncated.safetensors", json, 16);

    let error = File::open(&truncated).expect_err("少了 8 字节");

    assert_eq!(error.code(), Code::Invalid);

    assert!(error.to_string().contains("文件被截断"), "{error}");

    let _ = std::fs::remove_file(truncated);

    let padded = sparse("padded.safetensors", json, 25);

    let error = File::open(&padded).expect_err("多出 1 字节");

    assert!(error.to_string().contains("尾部有多余字节"), "{error}");

    let _ = std::fs::remove_file(padded);
}

fn a_prefix_that_promises_more_than_the_file_holds_is_rejected() {
    let short = path("short.safetensors");

    std::fs::write(&short, 4096u64.to_le_bytes()).expect("临时文件可写");

    let error = File::open(&short).expect_err("头部超长");

    assert!(error.to_string().contains("头部声明 4096 字节"), "{error}");

    let _ = std::fs::remove_file(short);

    let tiny = path("tiny.safetensors");

    std::fs::write(&tiny, [0u8; 4]).expect("临时文件可写");

    let error = File::open(&tiny).expect_err("不足 8 字节");

    assert!(error.to_string().contains("读不出 8 字节"), "{error}");

    let _ = std::fs::remove_file(tiny);
}

fn a_missing_file_is_an_io_error_not_a_panic() {
    let error = File::open("/nonexistent/llm.safetensors").expect_err("打不开");

    assert_eq!(error.code(), Code::Io);

    assert!(error.to_string().contains("打开文件"), "{error}");
}

fn both_backends_produce_the_same_header_and_the_same_bytes() {
    let json = r#"{"a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},

               "b": {"dtype": "I8", "shape": [4], "data_offsets": [24, 28]}}"#;

    let data: Vec<u8> = (0..28).collect();

    let bytes = file(json, &data);

    let archive = Archive::parse(&bytes).expect("合法");

    let target = path("agree.safetensors");

    std::fs::write(&target, &bytes).expect("临时文件可写");

    let on_disk = File::open(&target).expect("头部自证合法");

    assert_eq!(on_disk.header(), archive.header(), "两个后端解出同一个头部");

    for entry in archive.header().entries() {
        assert_eq!(
            Loader::load(&on_disk, entry).expect("可读"),
            Loader::load(&archive, entry).expect("可读"),
            "条目 `{}`",
            entry.name()
        );
    }

    let _ = std::fs::remove_file(target);
}

fn a_wrong_sized_buf_is_rejected_by_the_slice_backend_too() {
    let json = r#"{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}"#;

    let bytes = file(json, &[0; 4]);

    let archive = Archive::parse(&bytes).expect("合法");

    let entry = &archive.header().entries()[0];

    let mut too_big = [0u8; 8];

    let error = archive
        .copy_into(entry, &mut too_big)
        .expect_err("缓冲区太大");

    assert!(error.to_string().contains("缓冲区 8 字节"), "{error}");

    let mut exact = [9u8; 4];

    archive.copy_into(entry, &mut exact).expect("正好");

    assert_eq!(exact, [0u8; 4]);
}
}

#[cfg(all(feature = "mmap", unix))]
mod mmap_tests {
    use std::path::PathBuf;

    use diagnostic::Code;

    use crate::{MmapSource, Reader};

    /// 按 safetensors 布局拼一份临时文件：8 字节长度前缀 + 头部 JSON + 数据段。
    fn write_file(json: &str, data: &[u8]) -> PathBuf {
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();

        bytes.extend_from_slice(json.as_bytes());

        bytes.extend_from_slice(data);

        let path = std::env::temp_dir().join(format!(
            "llm_mmap_source_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        std::fs::write(&path, &bytes).expect("写临时权重文件");

        path
    }

    testkit::test_cases! {
    fn read_borrows_from_the_mapped_region() {
        let data: Vec<u8> = (0..24u8).collect();

        let json = r#"{"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}"#;

        let path = write_file(json, &data);

        let source = MmapSource::open(&path).expect("映射打开");

        let entry = source.header().get("w").expect("条目在位");

        // 零拷贝证据：返回切片的地址必须落在映射区内部，而不是另拷出来的缓冲。

        let mapped = source.as_slice();

        let map_start = mapped.as_ptr() as usize;

        let map_end = map_start + mapped.len();

        let bytes = source.read(entry).expect("借用读取");

        let bytes_start = bytes.as_ptr() as usize;

        assert!(
            bytes_start >= map_start && bytes_start + bytes.len() <= map_end,
            "借用切片必须落在映射区内"
        );

        assert_eq!(bytes, &data[..]);

        // 借用生命期绑定在源上：源存活期间切片可用，源释放后映射随之归还。

        let again = source.read(entry).expect("二次借用仍有效");

        assert_eq!(again, bytes);

        std::fs::remove_file(&path).expect("清理");
    }

    fn a_missing_file_is_an_io_error() {
        let path = PathBuf::from("/nonexistent/path/llm_mmap_source_no_such_file");

        let error = MmapSource::open(&path).expect_err("缺文件必须报错");

        assert_eq!(error.code(), Code::Io);
    }

    fn a_malformed_header_is_rejected_at_open() {
        let path = write_file(
            r#"{"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 48]}}"#,
            &[0u8; 24],
        );

        let error = MmapSource::open(&path).expect_err("长度不符必须报错");

        assert_eq!(error.code(), Code::Invalid);

        std::fs::remove_file(&path).expect("清理");
    }
    }
}
