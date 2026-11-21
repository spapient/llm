//! 格式 → 中性段流的桥接判据。
//!
//! 桥接函数 `import::segments_from_*` 依赖 [`Segment`](crate::Segment)，只能留在本 crate，
//! 所以 torch `.bin` 与 GGUF 的导入用例跟着它走；字节夹具仍由 `formats` 那边的
//! `tests/gguf_fixture.rs` 单点生成（跨 crate `#[path]` 复用，避免两份同构夹具）。
use crate::bin::Bin;
use crate::gguf::Gguf;
use crate::import::{segments_from_bin, segments_from_gguf};
use crate::testutil::real_bin_bytes;
use diagnostic::Code;
use dtype::DType;

#[path = "../../formats/tests/gguf_fixture.rs"]
mod fixture;

#[test]
fn keeps_torch_tensors_as_neutral_segments() {
    let Some(raw) = real_bin_bytes() else {
        return;
    };
    let bin = Bin::parse(&raw).expect("解析");
    let segments = segments_from_bin(&bin).expect("导入中性段流");
    assert_eq!(segments.len(), bin.tensors().len());
    for (segment, tensor) in segments.iter().zip(bin.tensors()) {
        assert_eq!(segment.name(), tensor.name());
        assert_eq!(segment.dtype(), tensor.dtype());
        assert_eq!(segment.shape(), tensor.shape());
        assert_eq!(segment.data, bin.data(tensor).expect("字节"));
    }
}

#[test]
fn propagates_unreadable_bin() {
    let error = Bin::parse(b"definitely not a torch bin").and_then(|bin| segments_from_bin(&bin));
    assert!(error.is_err(), "读不通的 bin 必须显式失败");
}

#[test]
fn maps_scalar_gguf_tensors_to_neutral_segments() {
    let raw = fixture::plain();
    let file = Gguf::parse(&raw).unwrap();
    let segments = segments_from_gguf(&file).expect("标量张量可导入");
    assert_eq!(segments.len(), 2);

    let f32 = &segments[0];
    assert_eq!(f32.name(), "f32.weight");
    assert_eq!(f32.dtype(), DType::F32);
    assert_eq!(f32.shape(), &[2]);
    assert_eq!(f32.data.len(), 8);

    let half = &segments[1];
    assert_eq!(half.name(), "half.weight");
    assert_eq!(half.dtype(), DType::F16);
    assert_eq!(half.shape(), &[2]);
    assert_eq!(half.data.len(), 4);
}

#[test]
fn rejects_gguf_block_quantization_without_dequantizing() {
    let raw = fixture::standard();
    let file = Gguf::parse(&raw).unwrap();
    let error = segments_from_gguf(&file).expect_err("块量化必须显式失败");
    assert_eq!(error.code(), Code::Unsupported);
}
