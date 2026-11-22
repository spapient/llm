//! `torch` 的单元测试：torch.save 检查点（`Tensor` / `Checkpoint`）的只读解析。
use super::*;
use crate::testutil::real_bin_bytes;
use dtype::DType;

#[test]
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

#[test]
fn rejects_a_non_archive() {
    assert!(Checkpoint::parse(b"definitely not a torch bin").is_err());
}
