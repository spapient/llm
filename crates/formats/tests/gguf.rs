//! `gguf` 模块的单元测试。
use super::*;
use crate::quant::f16_to_f32;

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
    // 线格式写死成字面量，才不是拿上游常量自证：`QType`/`DType` 改口径时这里必须一起改。
    const WIRE: [(Kind, Layout, usize, usize); 6] = [
        (Kind::F32, Layout::Scalar(DType::F32), 1, 4),
        (Kind::F16, Layout::Scalar(DType::F16), 1, 2),
        (Kind::Q8_0, Layout::Block(QType::Q8_0), 32, 34),
        (Kind::Q4_K, Layout::Block(QType::Q4_K), 256, 144),
        (Kind::Q5_K, Layout::Block(QType::Q5_K), 256, 176),
        (Kind::Q6_K, Layout::Block(QType::Q6_K), 256, 210),
    ];
    for (kind, layout, elems, span) in WIRE {
        assert_eq!(kind.layout().expect("布局"), layout, "{kind:?}");
        assert_eq!(kind.elems().expect("布局"), elems, "{kind:?} 元素数");
        assert_eq!(kind.span().expect("布局"), span, "{kind:?} 字节数");
        // 委派口径：块格式的元素数/字节数必须与 dtype 的 QType 表一致。
        match layout {
            Layout::Block(qtype) => {
                assert_eq!(span, qtype.block_bytes() as usize, "{kind:?}");
                assert_eq!(elems, qtype.block_len() as usize, "{kind:?}");
            }
            Layout::Scalar(dtype) => assert_eq!(span, dtype.bytes_for(1) as usize, "{kind:?}"),
        }
    }
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
    assert_eq!(file.tensors().len(), 6);
    assert_eq!(file.tensor("q4.weight").unwrap().kind, Kind::Q4_K);
}

#[test]
fn decodes_every_supported_kind_to_pinned_values() {
    let raw = fixture::standard();
    let file = Gguf::parse(&raw).unwrap();
    assert_eq!(file.floats("f32.weight").unwrap(), [1.0, -0.5, 3.25, 0.0]);
    assert_eq!(file.floats("half.weight").unwrap(), [0.5, -2.0, 0.0, 1.5]);
    let q8: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
    assert_eq!(file.floats("q8.weight").unwrap(), q8);
    // 夹具的块量化是 d=1.0、min=0、scales 全 1、qs 全 0x21：低半字节 1、高半字节 2。
    // 因此 Q4_K 每组 64 个值是 32 个 1.0 后接 32 个 2.0；Q6_K 的量化点 1/1/2/2 减 32 后
    // 是 64 个 -31.0 后接 64 个 -30.0。数值按 GGML 规范手推，不是拿实现自证。
    let q4 = file.floats("q4.weight").unwrap();
    let q5 = file.floats("q5.weight").unwrap();
    let q6 = file.floats("q6.weight").unwrap();
    assert_eq!((q4.len(), q5.len(), q6.len()), (256, 256, 256));
    for g in q4.chunks(64) {
        assert!(g[..32].iter().all(|&v| v == 1.0), "{:?}", &g[..4]);
        assert!(g[32..].iter().all(|&v| v == 2.0), "{:?}", &g[32..36]);
    }
    // Q5_K 的 qh 全 0x5A，位 1/3 落在组 0/1 的高位半字节、位 4/6 落在组 2/3 的低位半字节：
    // 于是组 0/1 是 32 个 1.0 后接 32 个 18.0，组 2/3 是 32 个 17.0 后接 32 个 2.0。
    for (g, group) in q5.chunks(64).enumerate() {
        let (low, high) = if g < 2 { (1.0, 18.0) } else { (17.0, 2.0) };
        assert!(group[..32].iter().all(|&v| v == low), "Q5_K 组 {g} 低位");
        assert!(group[32..].iter().all(|&v| v == high), "Q5_K 组 {g} 高位");
    }
    for p in q6.chunks(128) {
        assert!(p[..64].iter().all(|&v| v == -31.0), "{:?}", &p[..4]);
        assert!(p[64..].iter().all(|&v| v == -30.0), "{:?}", &p[64..68]);
    }
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
    assert!(error.to_string().contains("整数倍"), "{error}");
}
