//! `Mmap` 的单元测试：读写映射、空文件、缺失文件。

use super::*;
use testkit::ScratchDir;

testkit::test_cases! {
fn maps_file_bytes_and_length() {
    let dir = ScratchDir::new("mmap-bytes");
    let path = dir.join("payload.bin");
    let payload: Vec<u8> = (0..=250).collect();
    std::fs::write(&path, &payload).expect("写文件");

    let map = Mmap::open(&path).expect("映射");
    assert_eq!(map.len(), payload.len());
    assert_eq!(map.as_slice(), payload.as_slice());
    assert_eq!(map.as_ref(), payload.as_slice());
}

fn an_empty_file_maps_to_empty() {
    let dir = ScratchDir::new("mmap-empty");
    let path = dir.join("empty.bin");
    std::fs::write(&path, b"").expect("建空文件");
    let map = Mmap::open(&path).expect("空文件可映射");
    assert!(map.is_empty());
    assert_eq!(map.as_slice(), b"");
}

fn a_missing_file_is_an_error() {
    let map = Mmap::open(Path::new("/nonexistent/path/llm_mmap_no_such_file"));
    assert!(map.is_err());
}
}
