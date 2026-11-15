use super::*;

#[test]
fn maps_file_bytes_and_length() {
    let path = std::env::temp_dir().join(format!(
        "llm_mmap_{}_{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let payload: Vec<u8> = (0..=250).collect();
    std::fs::write(&path, &payload).expect("写文件");

    let map = Mmap::open(&path).expect("映射");
    assert_eq!(map.len(), payload.len());
    assert_eq!(map.as_slice(), payload.as_slice());
    assert_eq!(map.as_ref(), payload.as_slice());

    std::fs::remove_file(&path).expect("清理");
}

#[test]
fn an_empty_file_maps_to_empty() {
    let path = std::env::temp_dir().join(format!(
        "llm_mmap_empty_{}.bin",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&path, b"").expect("建空文件");
    let map = Mmap::open(&path).expect("空文件可映射");
    assert!(map.is_empty());
    assert_eq!(map.as_slice(), b"");
    std::fs::remove_file(&path).expect("清理");
}

#[test]
fn a_missing_file_is_an_error() {
    let map = Mmap::open(Path::new("/nonexistent/path/llm_mmap_no_such_file"));
    assert!(map.is_err());
}
