//! `lifecycle` 的单元测试：mode 与 patch 两条路径。
use super::*;
use crate::gguf::Gguf;
use crate::import::{segments_from_gguf, segments_from_torch};
use crate::testutil::{archive, f32_segment, real_bin_bytes};
use crate::torch::Checkpoint;
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
use crate::gguf_fixture as fixture;

#[test]
fn keeps_torch_tensors_as_neutral_segments() {
    let Some(raw) = real_bin_bytes() else {
        return;
    };
    let file = Checkpoint::parse(&raw).expect("解析");
    let segments = segments_from_torch(&file).expect("导入中性段流");
    assert_eq!(segments.len(), file.tensors().len());
    for (segment, tensor) in segments.iter().zip(file.tensors()) {
        assert_eq!(segment.name(), tensor.name());
        assert_eq!(segment.dtype(), tensor.dtype());
        assert_eq!(segment.shape(), tensor.shape());
        assert_eq!(segment.data, file.data(tensor).expect("字节"));
    }
}

#[test]
fn propagates_unreadable_torch_checkpoint() {
    let error = Checkpoint::parse(b"definitely not a torch bin")
        .and_then(|file| segments_from_torch(&file));
    assert!(error.is_err(), "读不通的检查点必须显式失败");
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
fn imports_every_block_kind_as_dequantized_f32() {
    let raw = fixture::standard();
    let file = Gguf::parse(&raw).unwrap();
    let segments = segments_from_gguf(&file).expect("块量化在导入时反量化");
    let values = |name: &str| -> Vec<f32> {
        let segment = segments.iter().find(|s| s.name() == name).expect("段");
        assert_eq!(segment.dtype(), DType::F32, "{name} 反量化后是 f32");
        segment
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    };
    assert_eq!(
        values("q8.weight"),
        (0..32).map(|i| i as f32 - 16.0).collect::<Vec<f32>>()
    );
    // 夹具的块量化是 d=1.0、min=0、scales 全 1、qs 全 0x21，数值按 GGML 规范手推：
    // Q4_K 每 64 值是 32 个 1.0 后接 32 个 2.0；Q6_K 每 128 值是 64 个 -31.0 后接 64 个 -30.0。
    for group in values("q4.weight").chunks(64) {
        assert!(group[..32].iter().all(|&v| v == 1.0), "Q4_K 低位");
        assert!(group[32..].iter().all(|&v| v == 2.0), "Q4_K 高位");
    }
    for group in values("q6.weight").chunks(128) {
        assert!(group[..64].iter().all(|&v| v == -31.0), "Q6_K 低位");
        assert!(group[64..].iter().all(|&v| v == -30.0), "Q6_K 高位");
    }
    // Q5_K 的 qh 全 0x5A：组 0/1 是 1.0/18.0，组 2/3 是 17.0/2.0。
    for (g, chunk) in values("q5.weight").chunks(64).enumerate() {
        let (low, high) = if g < 2 { (1.0, 18.0) } else { (17.0, 2.0) };
        let (a, b) = chunk.split_at(32);
        assert!(a.iter().all(|&v| v == low), "Q5_K 组 {g} 低位");
        assert!(b.iter().all(|&v| v == high), "Q5_K 组 {g} 高位");
    }
}
