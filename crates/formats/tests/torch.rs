//! `torch` 的单元测试：torch.save 检查点（`Tensor` / `Checkpoint`）的只读解析。
use super::*;
use crate::torch_sample::real_bin_bytes;
use dtype::DType;

#[test]
fn parses_real_torch_bin() {
    let Some(raw) = real_bin_bytes() else {
        return;
    };
    let file = Checkpoint::parse(&raw).expect("解析");
    assert_eq!(file.tensors().len(), 4);

    let w = file.tensor("layer.weight").expect("张量");
    assert_eq!(w.dtype(), DType::F32);
    assert_eq!(w.shape(), &[4, 6]);
    assert_eq!(file.data(w).expect("字节").len(), 96);

    let embed = file.tensor("embed").expect("f16");
    assert_eq!(embed.dtype(), DType::F16);
    assert_eq!(file.data(embed).expect("字节").len(), 24);

    let count = file.tensor("count").expect("i64");
    assert_eq!(count.dtype(), DType::I64);
    assert_eq!(file.data(count).expect("字节").len(), 24);
}

#[test]
fn bytes_match_torch_storage() {
    let Some(raw) = real_bin_bytes() else {
        return;
    };
    let file = Checkpoint::parse(&raw).expect("解析");
    let bias = file.tensor("layer.bias").expect("bias");
    let bytes = file.data(bias).expect("字节");
    assert_eq!(bytes.len(), 16);
}

#[test]
fn rejects_a_non_archive() {
    assert!(Checkpoint::parse(b"definitely not a torch bin").is_err());
}
