//! `mmap` 模块的单元测试。

//! M1-A2：映射源 [`MmapSource`] 借用生命周期与零拷贝证据。

//!

//! 父模块已在 `mmap` feature + Unix 双门控内；其它平台该类型不存在，测试整块剔除，

//! 不提供假装映射的空实现。



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



#[test]

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



#[test]

fn a_missing_file_is_an_io_error() {

    let path = PathBuf::from("/nonexistent/path/llm_mmap_source_no_such_file");

    let error = MmapSource::open(&path).expect_err("缺文件必须报错");

    assert_eq!(error.code(), Code::Io);

}



#[test]

fn a_malformed_header_is_rejected_at_open() {

    let path = write_file(

        r#"{"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 48]}}"#,

        &[0u8; 24],

    );

    let error = MmapSource::open(&path).expect_err("长度不符必须报错");

    assert_eq!(error.code(), Code::Invalid);

    std::fs::remove_file(&path).expect("清理");

}
