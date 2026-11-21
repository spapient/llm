//! `gguf` 模块的单元测试。
use super::*;
use crate::gguf::dequant::f16_to_f32;

#[path = "gguf_fixture.rs"]
mod fixture;

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

#[test]
fn parses_scalar_only_fixture() {
    let raw = fixture::plain();

    assert_eq!(Gguf::parse(&raw).unwrap().tensors().len(), 2);
}

#[test]
fn reads_header_and_metadata() {
    let raw = fixture::standard();

    let file = Gguf::parse(&raw).unwrap();

    assert_eq!(file.name(), Some("fixture"));

    assert_eq!(file.arch(), Some("llama"));

    assert_eq!(file.metas().len(), 4);

    assert_eq!(file.tensors().len(), 5);

    assert_eq!(file.tensor("q4.weight").unwrap().kind, Kind::Q4_K);
}

#[test]
fn decodes_plain_types() {
    let raw = fixture::standard();

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
    let raw = fixture::standard();

    let file = Gguf::parse(&raw).unwrap();

    let q4 = file.floats("q4.weight").unwrap();

    assert_eq!(q4.len(), 256);

    let q6 = file.floats("q6.weight").unwrap();

    assert_eq!(q6.len(), 256);
}

#[test]
fn rejects_bad_magic_and_overflow() {
    let mut raw = fixture::standard();

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
    let raw = fixture::standard();

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
fn a_partial_quantized_block_is_an_error_not_a_zero_tail() {
    let raw = [0u8; 34];
    let mut out = [1.0f32; 33];
    let error = dequantize(Kind::Q8_0, &raw, &mut out).expect_err("33 不是 32 的整数倍");
    assert!(error.to_string().contains("整除"), "{error}");
}
