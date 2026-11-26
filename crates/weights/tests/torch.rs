//! `torch` 的单元测试：torch.save 检查点（`Tensor` / `Checkpoint`）的只读解析。
use super::*;
use crate::testutil::real_bin_bytes;
use dtype::DType;

testkit::test_cases! {
fn parses_real_torch_bin() {
    let Some(raw) = real_bin_bytes() else {
        return;
    };
    let file = Checkpoint::parse(&raw).expect("解析");
    assert_eq!(file.tensors().len(), 4);
    const CASES: [(&str, DType, usize); 4] = [
        ("layer.weight", DType::F32, 96),
        ("layer.bias", DType::F32, 16),
        ("embed", DType::F16, 24),
        ("count", DType::I64, 24),
    ];
    for (name, dtype, bytes) in CASES {
        let tensor = file.tensor(name).expect("张量");
        assert_eq!(tensor.name(), name);
        assert_eq!(tensor.dtype(), dtype, "{name}");
        assert_eq!(file.data(tensor).expect("字节").len(), bytes, "{name}");
    }
    assert_eq!(file.tensor("layer.weight").expect("权重").shape(), [4, 6]);
}

fn rejects_a_non_archive() {
    assert!(Checkpoint::parse(b"definitely not a torch bin").is_err());
}
}

// ---- pickle 虚拟机 ----

#[test]
fn an_odd_setitems_is_an_error_not_a_panic() {
    let error = Vm::run(b"}(K\x01u.").expect_err("奇数项必须报错");
    assert!(error.to_string().contains("奇数"), "{error}");
}

/// 一段协议 4 的流：`PROTO`/`FRAME`、`STACK_GLOBAL`（模块名先压、类名后压）、
/// `BININT`/`BININT2`/`LONG1`/`LONG4` 四种整数编码与 `BINFLOAT`，最后重建一个 1×1 张量。
fn protocol4_stream() -> Vec<u8> {
    let mut out = b"\x80\x04\x95\x00\x00\x00\x00\x00\x00\x00\x00".to_vec();
    out.extend(b"ctorch._utils\n_rebuild_tensor_v2\n(");
    out.extend(b"\x8c\x07storage");
    out.extend(b"\x8c\x08__main__");
    out.extend(b"\x8c\x0cFloatStorage");
    out.extend(b"\x93\x8c\x010\x87Q");
    out.extend(b"J\x00\x00\x00\x00");
    out.extend(b"M\x01\x00\x8a\x01\x01\x86");
    out.extend(b"\x8b\x04\x00\x00\x00\x01\x00\x00\x00M\x01\x00\x86");
    out.extend(b"G\x3f\xe0\x00\x00\x00\x00\x00\x00");
    out.extend(b"\x89}tR.");
    out
}

testkit::test_cases! {
fn a_stack_global_names_the_class_that_was_pushed_last() {
    let tensors = Vm::run(&protocol4_stream()).expect("协议 4 张量");
    assert_eq!(tensors.len(), 1);
    assert_eq!(
        tensors[0].shape(),
        &[1, 1],
        "LONG1 与 BININT2 各解出一个维度"
    );
    assert_eq!(
        tensors[0].dtype(),
        dtype::DType::F32,
        "storage 类名解成 F32"
    );
    assert_eq!(tensors[0].offset, 0, "BININT 偏移");
    assert_eq!(tensors[0].len, 4, "LONG4 stride 与 1×1 元素数");
}

fn a_huge_memo_idx_is_rejected_not_resized() {
    let error = Vm::run(b"K\x01r\xff\xff\xff\xff.").expect_err("memo 索引必须报错");
    assert!(error.to_string().contains("memo"), "{error}");
}
}

/// 一段 `_rebuild_tensor_v2` 的 pickling：可选的 dict 键、可调用对象、实参元组与 REDUCE。
fn rebuild(key: &str, name: Option<&str>, size: (u8, u8), stride: (u8, u8)) -> Vec<u8> {
    let mut out = Vec::new();
    if let Some(name) = name {
        out.push(b'\x8c');
        out.push(name.len() as u8);
        out.extend(name.as_bytes());
    }
    out.extend(b"ctorch._utils\n_rebuild_tensor_v2\n");
    out.push(b'(');
    out.extend(b"\x8c\x07storage");
    out.extend(b"c__main__\nFloatStorage\n");
    out.push(b'\x8c');
    out.push(key.len() as u8);
    out.extend(key.as_bytes());
    out.push(0x87);
    out.push(b'Q');
    out.push(b'K');
    out.push(0);
    for dim in [size.0, size.1] {
        out.push(b'K');
        out.push(dim);
    }
    out.push(0x86);
    for step in [stride.0, stride.1] {
        out.push(b'K');
        out.push(step);
    }
    out.push(0x86);
    out.push(0x89);
    out.push(b'}');
    out.push(b't');
    out.push(b'R');
    out
}

testkit::test_cases! {
fn a_non_contiguous_tensor_is_refused_instead_of_read_as_contiguous() {
    // 2×2 的连续 stride 是 (2, 1)；(1, 2) 是转置布局，字节区间不等于「4 个元素 × 4 字节」。
    let mut pkl = b"\x80\x02".to_vec();
    pkl.extend(rebuild("0", Some("t"), (2, 2), (1, 2)));
    pkl.push(b'.');
    let error = Vm::run(&pkl).expect_err("非连续张量必须报错");
    assert!(error.to_string().contains("连续"), "{error}");
}

fn a_contiguous_tensor_still_parses() {
    let mut pkl = b"\x80\x02".to_vec();
    pkl.extend(rebuild("0", Some("w"), (2, 2), (2, 1)));
    pkl.push(b'.');
    let tensors = Vm::run(&pkl).expect("连续布局");
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].shape(), &[2, 2]);
    assert_eq!(tensors[0].name(), "w");
}

fn a_tensor_without_a_dict_key_gets_a_distinct_positional_name() {
    let mut pkl = b"\x80\x02\x28".to_vec();
    pkl.extend(rebuild("a", None, (1, 1), (1, 1)));
    pkl.extend(rebuild("b", None, (1, 1), (1, 1)));
    pkl.push(0x86);
    pkl.push(b'.');
    let tensors = Vm::run(&pkl).expect("元组里的两个张量");
    assert_eq!(tensors.len(), 2);
    assert_ne!(tensors[0].name(), tensors[1].name());
    assert!(!tensors[0].name().is_empty());
}
}

// ---- ZIP 容器 ----

testkit::test_cases! {
fn parses_real_torch_zip() {
    let Some(raw) = real_bin_bytes() else {
        return;
    };
    let zip = Zip::parse(&raw).expect("解析");
    assert!(zip.get_suffix("data.pkl").is_some());
    assert!(zip.get_suffix("/data/0").is_some());
    assert_eq!(zip.data_suffix("/data/0").expect("数据").len(), 96);
}

fn rejects_non_zip() {
    assert!(Zip::parse(b"not a zip archive at all").is_err());
}

fn slices_with_overflowing_offsets_are_errors_not_panics() {
    assert!(slice(b"abc", usize::MAX, 4).is_err());
    assert!(slice(b"abc", 1, usize::MAX).is_err());
}

fn zip64_fields_must_fit_the_declared_size() {
    let mut bytes = vec![0x01, 0x00, 8, 0];
    bytes.extend(7u64.to_le_bytes());
    bytes.extend(0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
    assert_eq!(zip64_extra(&bytes, 0, 12, 0).ok(), Some(7));
    assert!(zip64_extra(&bytes, 0, 12, 1).is_err());
}

fn zip64_extra_handles_absurd_lengths_without_overflow() {
    let bytes = vec![0x01, 0x00, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    assert!(zip64_extra(&bytes, 0, usize::MAX, 0).is_ok());
    assert!(zip64_extra(&bytes, usize::MAX - 2, 12, 0).is_err());
}
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
