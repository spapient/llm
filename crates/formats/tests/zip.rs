//! `zip` 的单元测试：ZIP_STORED 归档解析。
use super::*;
use crate::torch_sample::real_bin_bytes;

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

/// 本地头与中央目录共用的固定字段：版本号、标志、压缩方法、修改时间
/// （4 个零 u16）与 CRC32（零 u32）。
fn header_fixed(out: &mut Vec<u8>) {
    out.extend(45u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(0u32.to_le_bytes());
}

/// 最小 ZIP：一条 STORED 条目，中央目录里 `csize`/`usize`/本地头偏移全写成
/// `0xFFFF_FFFF`，真值按 APPNOTE 顺序（原长、压缩长、本地头偏移）放进 ZIP64 extra。
fn zip64_archive(name: &str, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(0x0403_4B50u32.to_le_bytes());
    header_fixed(&mut out);
    out.extend((payload.len() as u32).to_le_bytes());
    out.extend((payload.len() as u32).to_le_bytes());
    out.extend((name.len() as u16).to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(name.as_bytes());
    out.extend(payload);
    let cd_start = out.len();
    out.extend(0x0201_4B50u32.to_le_bytes());
    out.extend(45u16.to_le_bytes());
    header_fixed(&mut out);
    out.extend(0xFFFF_FFFFu32.to_le_bytes());
    out.extend(0xFFFF_FFFFu32.to_le_bytes());
    out.extend((name.len() as u16).to_le_bytes());
    out.extend(24u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(0u32.to_le_bytes());
    out.extend(0xFFFF_FFFFu32.to_le_bytes());
    out.extend(name.as_bytes());
    out.extend(0x0001u16.to_le_bytes());
    out.extend(24u16.to_le_bytes());
    out.extend((payload.len() as u64).to_le_bytes());
    out.extend((payload.len() as u64).to_le_bytes());
    out.extend(0u64.to_le_bytes());
    let cd_size = out.len() - cd_start;
    out.extend(0x0605_4B50u32.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend((cd_size as u32).to_le_bytes());
    out.extend((cd_start as u32).to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out
}

#[test]
fn zip64_fields_are_indexed_by_presence_not_by_field_number() {
    let raw = zip64_archive("/data/0", &[0xAA, 0xBB, 0xCC, 0xDD]);
    let zip = Zip::parse(&raw).expect("解析 ZIP64 归档");
    assert_eq!(
        zip.data_suffix("/data/0").expect("数据"),
        &[0xAA, 0xBB, 0xCC, 0xDD],
        "本地头偏移是 extra 里的第 3 个（槽位 2），不是槽位 0"
    );
}
