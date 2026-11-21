//! `lifecycle` 的单元测试：mode 与 patch 两条路径。
use super::*;
use crate::testutil::{archive, f32_segment, real_bin_bytes};
use crate::bin::Bin;
use crate::gguf::Gguf;
use crate::import::{segments_from_bin, segments_from_gguf};
use crate::writer::Writer;
use diagnostic::Code;
use dtype::DType;

#[test]
fn release_strips_opt_and_marks_infer() {
    let bytes = archive(vec![
        f32_segment("w", &[1.0, 2.0]),
        f32_segment("opt.m.w", &[0.5, 0.5]),
    ]);
    let src = Archive::parse(&bytes).expect("src");

    let released = release(&src).expect("release");
    let names: Vec<&str> = released.iter().map(|s| s.name()).collect();
    assert!(names.contains(&"w"));
    assert!(!names.iter().any(|n| is_opt(n)));

    let released_bytes = archive(released);
    let released_archive = Archive::parse(&released_bytes).expect("x");
    assert_eq!(mode_of(&released_archive).expect("模式"), Mode::Infer);
}

#[test]
fn missing_marker_defaults_infer() {
    let bytes = archive(vec![f32_segment("w", &[1.0])]);
    assert_eq!(
        mode_of(&Archive::parse(&bytes).expect("a")).expect("模式"),
        Mode::Infer
    );
}

#[test]
fn train_marker_roundtrips() {
    let mut segments = vec![f32_segment("w", &[1.0])];
    segments.push(Mode::Train.to_segment().expect("标记"));
    let bytes = archive(segments);
    let parsed = Archive::parse(&bytes).expect("a");
    assert_eq!(mode_of(&parsed).expect("模式"), Mode::Train);
}

#[test]
fn an_unknown_mode_code_is_rejected_not_fallen_back_to_infer() {
    let bad = Segment::new(MODE_NAME, Wire::U8, vec![1], None, vec![7]).expect("段");
    let bytes = archive(vec![f32_segment("w", &[1.0]), bad]);
    let parsed = Archive::parse(&bytes).expect("a");
    let error = mode_of(&parsed).expect_err("未知码必须报错");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("mode"), "{error}");
}

#[test]
fn xor_roundtrip_recovers_cur() {
    let base_bytes = archive(vec![f32_segment("w", &[1.0, 2.0, 3.0])]);
    let base = Archive::parse(&base_bytes).expect("base");
    let cur_bytes = archive(vec![f32_segment("w", &[1.0, 2.0, 3.5])]);
    let cur = Archive::parse(&cur_bytes).expect("cur");

    let patch_bytes = archive(diff(&base, &cur).expect("diff"));
    let patch = Archive::parse(&patch_bytes).expect("patch");

    let joined = join(&base, &patch).expect("join");
    let expected = cur.load(&cur.segments()[0]).expect("cur 字节");
    assert_eq!(joined[0].data, &expected[..]);
}

#[test]
fn small_change_patch_is_smaller() {
    let base_bytes = archive(vec![f32_segment("w", &[1.0; 4096])]);
    let base = Archive::parse(&base_bytes).expect("base");
    let mut changed = vec![1.0f32; 4096];
    changed[10] = 1.0001;
    let cur_bytes = archive(vec![f32_segment("w", &changed)]);
    let cur = Archive::parse(&cur_bytes).expect("cur");

    let mut w = Writer::new();
    w.compress(true);
    for seg in diff(&base, &cur).expect("diff") {
        w.push(seg).expect("push");
    }
    assert!(w.finish(6).expect("patch").len() < cur_bytes.len());
}

#[test]
fn missing_base_is_full_not_garbage() {
    let base_bytes = archive(vec![f32_segment("a", &[1.0])]);
    let base = Archive::parse(&base_bytes).expect("base");
    let cur_bytes = archive(vec![f32_segment("b", &[9.0])]);
    let cur = Archive::parse(&cur_bytes).expect("cur");

    let patch = diff(&base, &cur).expect("diff");
    assert_eq!(patch[0].name(), "b");
    assert_eq!(patch[0].data, f32_segment("b", &[9.0]).data);
}

// —— 格式 → 中性段流的桥接：`import::segments_from_*` 依赖本 crate 的 `Segment`，用例只能留在这里。
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
