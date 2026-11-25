use super::*;
use crate::model::{Buf, Values};

const TEMPLATE: &str = "typedef unsigned char u8;\n\
        void cutile_probe_kernel(const float *x, const u8 *w, const float *scales,\n\
                                 float *y, int M, int N, int gsz) {\n\
            @dot(y, x, w, k);\n\
        }\n";

fn sample() -> Sample {
    Sample {
        m: 4,
        n: 2,
        scalars: vec![("M", 4), ("N", 2), ("gsz", 2)],
        bufs: vec![
            Buf {
                name: "x",
                values: Values::F32(vec![1.0; 4]),
            },
            Buf {
                name: "w",
                values: Values::U8(vec![0x98, 0x76]),
            },
            Buf {
                name: "scales",
                values: Values::F32(vec![0.5; 2]),
            },
            Buf {
                name: "y",
                values: Values::F32(vec![0.0; 2]),
            },
        ],
    }
}

#[test]
fn signature_derives_the_argument_list_from_the_template() {
    let signature = Signature::parse("probe", TEMPLATE).expect("签名");
    assert_eq!(signature.kernel(), "cutile_probe_kernel");
    assert_eq!(signature.output(), Some("y"));
    assert_eq!(
        signature.call_args(&sample()).expect("实参"),
        "x, w, scales, y, 4, 2, 2"
    );
}

#[test]
fn an_unsupported_buf_type_is_rejected_with_the_parameter_name() {
    let broken = TEMPLATE.replace("const float *scales", "double *scales");
    let error = Signature::parse("probe", &broken).expect_err("double 不是缓冲类型");
    let text = error.to_string();
    assert!(text.contains("probe"), "{text}");
    assert!(text.contains("scales"), "{text}");
    assert!(text.contains("double"), "{text}");
    assert!(text.contains("第 2 行"), "{text}");

    // 形参从下一行开始时，行号要指向该形参，而不是上一行的逗号。
    let broken = TEMPLATE.replace("float *y,", "double *y,");
    let error = Signature::parse("probe", &broken).expect_err("double 不是缓冲类型");
    assert!(error.to_string().contains("第 3 行"), "{error}");
}

#[test]
fn the_output_must_be_the_single_writable_pointer() {
    let broken = TEMPLATE.replace("float *y", "const float *y");
    let error = Signature::parse("probe", &broken).expect_err("没有可写输出");
    assert!(error.to_string().contains("可写缓冲"), "{error}");
}

#[test]
fn sample_names_and_types_must_match_the_signature() {
    let signature = Signature::parse("probe", TEMPLATE).expect("签名");

    let mut mistyped = sample();
    mistyped.bufs[2].values = Values::U8(vec![0; 2]);
    let error = signature.check_sample(&mistyped).expect_err("类型不符");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("scales"), "{error}");
    assert!(error.to_string().contains("float"), "{error}");

    let mut stray = sample();
    stray.bufs.push(Buf {
        name: "bias",
        values: Values::F32(vec![0.0; 2]),
    });
    let error = signature.check_sample(&stray).expect_err("多出的缓冲");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("bias"), "{error}");

    let mut missing = sample();
    missing.scalars.retain(|(name, _)| *name != "gsz");
    let error = signature.check_sample(&missing).expect_err("缺标量");
    assert_eq!(error.code(), Code::Missing);
    assert!(error.to_string().contains("gsz"), "{error}");
}

/// 输出类型要能从签名读出来，而且与样本侧用同一个词：注解里的 `dtype` 因此有得比对。
#[test]
fn the_output_type_is_readable_and_names_the_same_element_type() {
    let signature = Signature::parse("probe", TEMPLATE).expect("签名");
    assert_eq!(signature.output_type(), Some(CType::F32));
    assert_eq!(CType::F32.dtype(), Some(DType::F32));
    assert_eq!(CType::U8.dtype(), Some(DType::U8));
    assert_eq!(CType::Int.dtype(), None, "标量不占缓冲，没有元素类型");

    // 可写输出换成字节缓冲时，类型查询跟着变——这正是"回读只认 float"要拦的那一类。
    let broken = TEMPLATE.replace("float *y", "u8 *y");
    let signature = Signature::parse("probe", &broken).expect("u8 是闭集内的缓冲类型");
    assert_eq!(signature.output_type(), Some(CType::U8));
    assert_eq!(signature.output(), Some("y"));
}

#[test]
fn vector_signature_preserves_dtype_and_lane_identity() {
    let signature = Signature::parse(
        "vec4",
        "void cutile_vec4_kernel(const float4 *a, float4 *y, int N) {}",
    )
    .expect("float4 签名");
    assert_eq!(signature.output_type(), Some(CType::F32x4));
    assert_eq!(CType::F32x4.dtype(), Some(DType::F32));
    assert_eq!(CType::F32x4.lanes(), 4);

    let mut sample = Sample {
        m: 4,
        n: 4,
        scalars: vec![("N", 4)],
        bufs: vec![
            Buf {
                name: "a",
                values: Values::F32x4(crate::model::Vector::from_fn(1, |idx| idx as f32)),
            },
            Buf {
                name: "y",
                values: Values::F32x4(crate::model::Vector::from_fn(1, |_| 0.0)),
            },
        ],
    };
    signature.check_sample(&sample).expect("同档样本");
    sample.bufs[0].values = Values::F32(vec![0.0; 4]);
    assert!(signature.check_sample(&sample).is_err(), "标量不能冒充向量");
}

#[test]
fn a_void_signature_has_no_output_and_is_rejected() {
    let error =
        Signature::parse("probe", "void cutile_probe_kernel(void) {}\n").expect_err("没有输出");
    assert!(error.to_string().contains("可写缓冲"), "{error}");
}
