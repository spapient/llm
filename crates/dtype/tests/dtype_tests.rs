use super::*;
use std::{format, mem::size_of};

#[test]
fn discriminant_fits_one_byte() {
    assert_eq!(size_of::<DType>(), 1, "repr(u8) 加 24 变体应正好一字节");
}

#[test]
fn every_u16_is_either_a_valid_code_or_rejected() {
    // 穷举全部 65536 个 code：合法集恰好是 `1..=ALL.len()`，且往返一致。
    let last = u16::try_from(DType::ALL.len()).expect("DType 变体数远小于 u16 上限");
    let valid = 1..=last;
    for code in u16::MIN..=u16::MAX {
        match DType::from_code(code) {
            Some(dtype) => {
                assert!(valid.contains(&code), "{code} 不应被接受");
                assert_eq!(dtype.code(), code);
                assert_eq!(DType::try_from(code), Ok(dtype));
            }
            None => assert!(!valid.contains(&code), "{code} 不应被拒绝"),
        }
    }
}

#[test]
fn identity_table_is_locked() {
    // 一行一个类型：线上 code、规范名、变体名、位宽、数值类别与浮点位域 `(exp, mantissa, bias)`。
    // 长度由 `ALL.len()` 锚定，加类型时这张表必须同步（漏写一行直接编译失败）；顺序也在这里
    // 锁住——重排会让 code 漂移。
    /// 锁定表里的浮点位域列：`(exp_bits, mantissa_bits, exponent_bias)`。
    type FloatBits = Option<(u32, u32, u32)>;

    const LOCKED: [(u16, &str, &str, u32, Class, FloatBits); DType::ALL.len()] = [
        (1, "bool", "Bool", 1, Class::Bool, None),
        (2, "u1", "U1", 1, Class::UInt, None),
        (3, "u2", "U2", 2, Class::UInt, None),
        (4, "u4", "U4", 4, Class::UInt, None),
        (5, "u8", "U8", 8, Class::UInt, None),
        (6, "u16", "U16", 16, Class::UInt, None),
        (7, "u32", "U32", 32, Class::UInt, None),
        (8, "u64", "U64", 64, Class::UInt, None),
        (9, "i1", "I1", 1, Class::SInt, None),
        (10, "i2", "I2", 2, Class::SInt, None),
        (11, "i4", "I4", 4, Class::SInt, None),
        (12, "i8", "I8", 8, Class::SInt, None),
        (13, "i16", "I16", 16, Class::SInt, None),
        (14, "i32", "I32", 32, Class::SInt, None),
        (15, "i64", "I64", 64, Class::SInt, None),
        (16, "f8e4m3", "F8E4M3", 8, Class::Float, Some((4, 3, 7))),
        (17, "f8e5m2", "F8E5M2", 8, Class::Float, Some((5, 2, 15))),
        (18, "f16", "F16", 16, Class::Float, Some((5, 10, 15))),
        (19, "bf16", "BF16", 16, Class::Float, Some((8, 7, 127))),
        (20, "f32", "F32", 32, Class::Float, Some((8, 23, 127))),
        (21, "f64", "F64", 64, Class::Float, Some((11, 52, 1023))),
        (22, "f4", "F4", 4, Class::Float, Some((2, 1, 1))),
        (
            23,
            "f8e4m3fnuz",
            "F8E4M3FNUZ",
            8,
            Class::Float,
            Some((4, 3, 8)),
        ),
        (
            24,
            "f8e5m2fnuz",
            "F8E5M2FNUZ",
            8,
            Class::Float,
            Some((5, 2, 16)),
        ),
    ];
    for (dtype, (code, name, variant, bits, class, float)) in DType::ALL.iter().zip(LOCKED) {
        let spec = dtype.spec();
        assert_eq!(
            (
                dtype.code(),
                spec.name,
                spec.variant,
                spec.bits,
                spec.class,
                spec.float.map(|layout| (
                    layout.exp_bits,
                    layout.mantissa_bits,
                    layout.exponent_bias
                )),
            ),
            (code, name, variant, bits, class, float),
            "{dtype:?} 的身份行被改动"
        );
    }
}

#[test]
fn variant_names_match_the_derived_debug_identifiers() {
    for dtype in DType::ALL {
        assert_eq!(
            dtype.variant(),
            format!("{dtype:?}"),
            "变体名与 Debug 输出必须一致，annotation 生成的 `DType::` 变体才编得过"
        );
    }
}

#[test]
fn wire_code_traits_agree_with_the_const_accessors() {
    for dtype in DType::ALL {
        assert_eq!(u16::from(*dtype), dtype.code());
        assert_eq!(DType::try_from(dtype.code()), Ok(*dtype));
    }
    assert_eq!(
        DType::try_from(0),
        Err(crate::format::UnknownCode { code: 0 })
    );
    let past_last = u16::try_from(DType::ALL.len()).expect("DType 变体数远小于 u16 上限") + 1;
    assert_eq!(
        DType::try_from(past_last),
        Err(crate::format::UnknownCode { code: past_last })
    );
}

#[test]
fn predicates_read_the_class_column() {
    let (mut bools, mut floats, mut integers, mut signed, mut unsigned) = (0, 0, 0, 0, 0);
    for dtype in DType::ALL {
        let class = dtype.spec().class;
        assert_eq!(dtype.is_bool(), class == Class::Bool, "{dtype:?}");
        assert_eq!(dtype.is_float(), class == Class::Float, "{dtype:?}");
        assert_eq!(dtype.is_signed(), class == Class::SInt, "{dtype:?}");
        assert_eq!(dtype.is_unsigned(), class == Class::UInt, "{dtype:?}");
        assert_eq!(
            dtype.is_integer(),
            matches!(class, Class::UInt | Class::SInt),
            "{dtype:?}"
        );
        assert_eq!(dtype.is_sub_byte(), dtype.bits() < 8, "{dtype:?}");
        let sign = dtype.sign_bits();
        assert_eq!(
            sign,
            match class {
                Class::Float | Class::SInt => 1,
                Class::Bool | Class::UInt => 0,
            },
            "{dtype:?}"
        );
        bools += usize::from(dtype.is_bool());
        floats += usize::from(dtype.is_float());
        integers += usize::from(dtype.is_integer());
        signed += usize::from(dtype.is_signed());
        unsigned += usize::from(dtype.is_unsigned());
    }
    // 类别普查：加类型时这里要显式改数，避免行表悄悄多出第 25 个变体。四个类别两两互斥、
    // 合起来铺满闭集（1 + 7 + 7 + 9 = 24），`sign_bits` 只对浮点与有符号整数亮 1。
    assert_eq!(
        (bools, floats, integers, signed, unsigned),
        (1, 9, 14, 7, 7)
    );
}

#[test]
fn float_layout_is_present_exactly_for_floats() {
    // 两列同真同假：`class` 说它是浮点，`float` 就必须有位域，反之亦然。这是 `Option` 冗余
    // 的安全网——任一侧漏填都在这里对不上。
    for dtype in DType::ALL {
        assert_eq!(
            dtype.float_layout().is_some(),
            dtype.is_float(),
            "{dtype:?} 的浮点位域与类别分家"
        );
        assert_eq!(dtype.exp_bits(), dtype.float_layout().map(|l| l.exp_bits));
        assert_eq!(
            dtype.mantissa_bits(),
            dtype.float_layout().map(|l| l.mantissa_bits)
        );
        assert_eq!(
            dtype.exponent_bias(),
            dtype.float_layout().map(|l| l.exponent_bias)
        );
    }
}

#[test]
fn float_bit_fields_fill_the_whole_element() {
    // 位域是可加的量：符号 1 位 + 指数 + 尾数必须正好等于位宽，既不能少算也不能重复占位。
    for dtype in DType::ALL {
        if let Some(layout) = dtype.float_layout() {
            assert_eq!(
                dtype.sign_bits() + layout.exp_bits + layout.mantissa_bits,
                dtype.bits(),
                "{dtype:?} 的浮点位域之和不等于位宽"
            );
            assert!(layout.exp_bits > 0, "{dtype:?} 必须有指数字段");
        }
    }
}

#[test]
fn bytes_for_packs_sub_byte_types_without_partial_bytes() {
    // 字面值而不是用 `bits` 重算公式：亚字节跨元素打包，整字节类型不额外补齐。
    assert_eq!((DType::U1.bytes_for(8), DType::U1.bytes_for(9)), (1, 2));
    assert_eq!((DType::U4.bytes_for(1), DType::U4.bytes_for(2)), (1, 1));
    assert_eq!(DType::U4.bytes_for(3), 2);
    assert_eq!((DType::I2.bytes_for(3), DType::I2.bytes_for(4)), (1, 1));
    assert_eq!((DType::F4.bytes_for(1), DType::F4.bytes_for(8)), (1, 4));
    assert_eq!((DType::Bool.bytes_for(8), DType::Bool.bytes_for(9)), (1, 2));
    assert_eq!(DType::U8.bytes_for(1), 1);
    assert_eq!(DType::F32.bytes_for(3), 12);
    assert_eq!(DType::F64.bytes_for(2), 16);
}
