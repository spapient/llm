//! `gguf` 模块的单元测试。

use super::*;

#[test]

fn half_specials() {
    assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);

    assert!(f16_to_f32(0xFC00).is_infinite() && f16_to_f32(0xFC00) < 0.0);

    assert!(f16_to_f32(0x7C01).is_nan());

    assert_eq!(f16_to_f32(0), 0.0);
}

#[test]

fn value_views_match_variant() {
    assert_eq!(Value::Int(-7).int(), Some(-7));

    assert_eq!(Value::Float(1.5).float(), Some(1.5));

    assert_eq!(Value::Bool(true).bool(), Some(true));

    let pieces = Value::Arr(vec![Value::Str("a".into()), Value::Str("b".into())]);

    assert_eq!(pieces.text_array(), Some(vec!["a", "b"]));

    let scores = Value::Arr(vec![Value::Float(0.1), Value::Float(-0.2)]);

    assert_eq!(scores.float_array(), Some(vec![0.1, -0.2]));

    // 错型与混型返回 None，不做隐式转换。

    assert!(Value::Uint(3).int().is_none());

    let mixed = Value::Arr(vec![Value::Str("a".into()), Value::Float(1.0)]);

    assert!(mixed.text_array().is_none());

    assert!(Value::Str("x".to_string()).arr().is_none());
}

#[test]

fn kind_maps_ggml_codes_and_delegates_layout() {
    const CODES: [(u32, Kind); 6] = [
        (0, Kind::F32),
        (1, Kind::F16),
        (8, Kind::Q8_0),
        (12, Kind::Q4_K),
        (13, Kind::Q5_K),
        (14, Kind::Q6_K),
    ];

    assert_eq!(SUPPORTED.len(), CODES.len(), "支持表与锁定码表必须同规模");

    for (code, kind) in CODES {
        assert_eq!(kind.code(), code);

        assert_eq!(Kind::from_code(code).unwrap(), kind);

        let hits = SUPPORTED.iter().filter(|(other, _)| *other == kind).count();

        assert_eq!(hits, 1, "{kind:?} 在 SUPPORTED 里必须恰好出现一次");
    }

    assert!(Kind::from_code(2).is_err(), "未支持的 ggml 码必须显式失败");

    assert!(Kind::from_code(15).is_err(), "未支持的 ggml 码必须显式失败");

    assert_eq!(
        Kind::F32.layout().expect("布局"),
        Layout::Scalar(DType::F32)
    );

    assert_eq!(
        Kind::F16.layout().expect("布局"),
        Layout::Scalar(DType::F16)
    );

    assert_eq!(
        Kind::Q8_0.layout().expect("布局"),
        Layout::Block(QType::Q8_0)
    );

    assert_eq!(
        Kind::Q4_K.layout().expect("布局"),
        Layout::Block(QType::Q4_K)
    );

    assert_eq!(
        Kind::Q5_K.layout().expect("布局"),
        Layout::Block(QType::Q5_K)
    );

    assert_eq!(
        Kind::Q6_K.layout().expect("布局"),
        Layout::Block(QType::Q6_K)
    );

    assert_eq!(
        Kind::Q8_0.elems().expect("布局"),
        QType::Q8_0.block_len() as usize
    );

    assert_eq!(
        Kind::Q8_0.span().expect("布局"),
        QType::Q8_0.block_bytes() as usize
    );

    assert_eq!(
        Kind::Q4_K.elems().expect("布局"),
        QType::Q4_K.block_len() as usize
    );

    assert_eq!(
        Kind::Q4_K.span().expect("布局"),
        QType::Q4_K.block_bytes() as usize
    );

    assert_eq!(
        Kind::F32.span().expect("布局"),
        DType::F32.bytes_for(1) as usize
    );

    assert_eq!(
        Kind::F16.span().expect("布局"),
        DType::F16.bytes_for(1) as usize
    );
}

#[test]

fn undecoded_qtypes_are_explicit() {
    let decoded: Vec<QType> = SUPPORTED
        .iter()
        .filter_map(|(_, layout)| match layout {
            Layout::Scalar(_) => None,

            Layout::Block(qtype) => Some(*qtype),
        })
        .collect();

    assert_eq!(
        decoded,
        vec![QType::Q8_0, QType::Q4_K, QType::Q5_K, QType::Q6_K]
    );

    assert_eq!(
        QType::ALL.len() - decoded.len(),
        21,
        "其余 QType 尚无解码器；接上时要同步这张表与反量化分支"
    );

    for qtype in &decoded {
        assert!(
            QType::ALL.contains(qtype),
            "{qtype:?} 必须在 dtype 的闭集里"
        );
    }
}

fn text(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());

    out.extend_from_slice(value.as_bytes());
}

fn word(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn wide(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn key(out: &mut Vec<u8>, name: &str, kind: u32) {
    text(out, name);

    word(out, kind);
}

fn build() -> Vec<u8> {
    let f32s: Vec<u8> = [1.0f32, -0.5, 3.25, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();

    let halves: Vec<u8> = [0x3800u16, 0xC000, 0, 0x3E00]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();

    let mut q8 = 0x3C00u16.to_le_bytes().to_vec();

    q8.extend((0..32).map(|i| (i as i8) - 16).map(|q| q as u8));

    let mut q4 = 0x3C00u16.to_le_bytes().to_vec();

    q4.extend(0u16.to_le_bytes());

    q4.extend([1u8; 12]);

    q4.extend([0x21u8; 128]);

    let mut q6 = vec![0x21u8; 128];

    q6.extend([0u8; 64]);

    q6.extend([1u8; 16]);

    q6.extend(0x3C00u16.to_le_bytes());

    let pieces = vec![
        ("f32.weight", vec![2, 2], Kind::F32, f32s),
        ("half.weight", vec![4], Kind::F16, halves),
        ("q8.weight", vec![32], Kind::Q8_0, q8),
        ("q4.weight", vec![256], Kind::Q4_K, q4),
        ("q6.weight", vec![256], Kind::Q6_K, q6),
    ];

    assemble(&pieces)
}

/// 只有标量 F32/F16 的最小夹具：供导入通路的成功路径对拍。

fn plain_build() -> Vec<u8> {
    let f32s: Vec<u8> = [1.0f32, -0.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();

    let halves: Vec<u8> = [0x3800u16, 0xC000]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();

    let pieces = vec![
        ("f32.weight", vec![2], Kind::F32, f32s),
        ("half.weight", vec![2], Kind::F16, halves),
    ];

    assemble(&pieces)
}

fn assemble(pieces: &[(&str, Vec<u64>, Kind, Vec<u8>)]) -> Vec<u8> {
    let mut head = Vec::new();

    head.extend_from_slice(&MAGIC);

    word(&mut head, VERSION);

    wide(&mut head, pieces.len() as u64);

    wide(&mut head, 4);

    key(&mut head, "general.name", 8);

    text(&mut head, "fixture");

    key(&mut head, "general.architecture", 8);

    text(&mut head, "llama");

    key(&mut head, "general.alignment", 4);

    word(&mut head, DEFAULT_ALIGN as u32);

    key(&mut head, "llama.block_count", 4);

    word(&mut head, 1);

    let mut offset = 0u64;

    let mut infos = Vec::new();

    let mut body = Vec::new();

    for (name, shape, kind, bytes) in pieces {
        offset = offset.div_ceil(DEFAULT_ALIGN as u64) * DEFAULT_ALIGN as u64;

        text(&mut infos, name);

        word(&mut infos, shape.len() as u32);

        for dim in shape {
            wide(&mut infos, *dim);
        }

        word(&mut infos, kind.code());

        wide(&mut infos, offset);

        body.extend(std::iter::repeat_n(0, offset as usize - body.len()));

        body.extend_from_slice(bytes);

        offset += bytes.len() as u64;
    }

    head.extend(infos);

    let fill = (DEFAULT_ALIGN - head.len() % DEFAULT_ALIGN) % DEFAULT_ALIGN;

    head.extend(std::iter::repeat_n(0, fill));

    head.extend(body);

    head
}

#[test]

fn reads_header_and_metadata() {
    let raw = build();

    let file = Gguf::parse(&raw).unwrap();

    assert_eq!(file.name(), Some("fixture"));

    assert_eq!(file.arch(), Some("llama"));

    assert_eq!(file.metas().len(), 4);

    assert_eq!(file.tensors().len(), 5);

    assert_eq!(file.tensor("q4.weight").unwrap().kind, Kind::Q4_K);
}

#[test]

fn decodes_plain_types() {
    let raw = build();

    let file = Gguf::parse(&raw).unwrap();

    assert_eq!(
        file.floats("f32.weight").unwrap(),
        vec![1.0, -0.5, 3.25, 0.0]
    );

    assert_eq!(
        file.floats("half.weight").unwrap(),
        vec![0.5, -2.0, 0.0, 1.5]
    );

    assert_eq!(
        file.floats("q8.weight").unwrap(),
        (0..32).map(|i| i as f32 - 16.0).collect::<Vec<_>>()
    );
}

#[test]

fn decodes_q4k_and_q6k() {
    let raw = build();

    let file = Gguf::parse(&raw).unwrap();

    let q4 = file.floats("q4.weight").unwrap();

    assert_eq!(q4.len(), 256);

    let q6 = file.floats("q6.weight").unwrap();

    assert_eq!(q6.len(), 256);
}

#[test]

fn rejects_bad_magic_and_overflow() {
    let mut raw = build();

    raw[0] = b'X';

    assert!(Gguf::parse(&raw).is_err());

    let tensor = Tensor {
        name: "huge".into(),

        kind: Kind::F32,

        shape: vec![1u64 << 62],

        offset: 0,
    };

    assert!(tensor.bytes().is_err());
}

#[test]

fn raw_rejects_an_offset_whose_length_overflows() {
    let raw = build();

    let file = Gguf::parse(&raw).unwrap();

    let offset = u64::try_from(usize::MAX - file.data).unwrap();

    let tensor = Tensor {
        name: "straddle".into(),

        kind: Kind::F32,

        shape: vec![1],

        offset,
    };

    assert!(
        file.raw(&tensor).is_err(),
        "偏移 + 长度溢出必须走错误分支，而不是靠 debug 的算术溢出 panic"
    );
}

#[test]

fn import_maps_scalar_tensors_to_neutral_segments() {
    let raw = plain_build();

    let file = Gguf::parse(&raw).unwrap();

    let segments = crate::import::segments_from_gguf(&file).expect("标量张量可导入");

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

fn import_rejects_block_quantization_without_dequantizing() {
    let raw = build();

    let file = Gguf::parse(&raw).unwrap();

    let error = crate::import::segments_from_gguf(&file).expect_err("块量化必须显式失败");

    assert_eq!(error.code(), Code::Unsupported);
}
