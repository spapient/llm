//! `bin` 的单元测试：Tensor/Bin 只读解析。
use super::*;
use crate::testutil::real_bin_bytes;
use dtype::DType;

#[test]
fn parses_real_torch_bin() {
    let Some(raw) = real_bin_bytes() else {
        return;
    };
    let bin = Bin::parse(&raw).expect("解析");
    assert_eq!(bin.tensors().len(), 4);

    let w = bin.tensor("layer.weight").expect("张量");
    assert_eq!(w.dtype(), DType::F32);
    assert_eq!(w.shape(), &[4, 6]);
    assert_eq!(bin.data(w).expect("字节").len(), 96);

    let embed = bin.tensor("embed").expect("f16");
    assert_eq!(embed.dtype(), DType::F16);
    assert_eq!(bin.data(embed).expect("字节").len(), 24);

    let count = bin.tensor("count").expect("i64");
    assert_eq!(count.dtype(), DType::I64);
    assert_eq!(bin.data(count).expect("字节").len(), 24);
}

#[test]
fn bytes_match_torch_storage() {
    let Some(raw) = real_bin_bytes() else {
        return;
    };
    let bin = Bin::parse(&raw).expect("解析");
    let bias = bin.tensor("layer.bias").expect("bias");
    let bytes = bin.data(bias).expect("字节");
    assert_eq!(bytes.len(), 16);
}

#[test]
fn rejects_non_bin() {
    assert!(Bin::parse(b"definitely not a torch bin").is_err());
}
