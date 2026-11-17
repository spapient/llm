use super::*;
use crate::testutil::real_bin_bytes;

#[test]
fn parses_real_torch_zip() {
    let Some(raw) = real_bin_bytes() else {
        return;
    };
    let zip = Zip::parse(&raw).expect("解析");
    assert!(zip.get_suffix("data.pkl").is_some());
    assert!(zip.get_suffix("/data/0").is_some());
    assert_eq!(zip.data_suffix("/data/0").expect("数据").len(), 96);
}

#[test]
fn rejects_non_zip() {
    assert!(Zip::parse(b"not a zip archive at all").is_err());
}

#[test]
fn slices_with_overflowing_offsets_are_errors_not_panics() {
    assert!(slice(b"abc", usize::MAX, 4).is_err());
    assert!(slice(b"abc", 1, usize::MAX).is_err());
}

#[test]
fn zip64_fields_must_fit_the_declared_size() {
    let mut bytes = vec![0x01, 0x00, 8, 0];
    bytes.extend(7u64.to_le_bytes());
    bytes.extend(0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
    assert_eq!(zip64_extra(&bytes, 0, 12, 0).ok(), Some(7));
    assert!(zip64_extra(&bytes, 0, 12, 1).is_err());
}

#[test]
fn zip64_extra_handles_absurd_lengths_without_overflow() {
    let bytes = vec![0x01, 0x00, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    assert!(zip64_extra(&bytes, 0, usize::MAX, 0).is_ok());
    assert!(zip64_extra(&bytes, usize::MAX - 2, 12, 0).is_err());
}
