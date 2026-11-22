use super::*;
use std::{format, mem::size_of};

#[test]
fn mx_type_discriminant_fits_one_byte() {
    assert_eq!(size_of::<MxType>(), 1, "repr(u8) 加 7 个格式应正好一字节");
}

#[test]
fn element_table_is_locked() {
    // 逐行锁死元素身份：位宽、名字、标量对应与浮点位域 `(exp, mantissa, bias)` 被误改时
    // 失败在这一行。6 位 MX 格式没有标量对应，但位域照样进表——反量化不需要 `DType` 才能解。
    /// 锁定表里的浮点位域列：`(exp_bits, mantissa_bits, exponent_bias)`。
    type FloatBits = Option<(u32, u32, u32)>;

    const EXPECTED: [(&str, &str, u32, Option<DType>, FloatBits); Element::ALL.len()] = [
        ("e2m1", "E2M1", 4, Some(DType::F4), Some((2, 1, 1))),
        ("e2m3", "E2M3", 6, None, Some((2, 3, 1))),
        ("e3m2", "E3M2", 6, None, Some((3, 2, 3))),
        ("e4m3", "E4M3", 8, Some(DType::F8E4M3), Some((4, 3, 7))),
        ("e5m2", "E5M2", 8, Some(DType::F8E5M2), Some((5, 2, 15))),
        ("int8", "Int8", 8, Some(DType::I8), None),
    ];
    for (element, (name, variant, bits, dtype, float)) in Element::ALL.iter().zip(EXPECTED) {
        let spec = element.spec();
        assert_eq!(
            (
                spec.name,
                spec.variant,
                spec.bits,
                spec.dtype,
                spec.float.map(|layout| (
                    layout.exp_bits,
                    layout.mantissa_bits,
                    layout.exponent_bias
                )),
            ),
            (name, variant, bits, dtype, float),
            "{element:?} 的身份行被改动"
        );
        assert_eq!(element.name(), variant.to_ascii_lowercase());
        assert_eq!(element.variant(), format!("{element:?}"));
        assert_eq!(Element::parse_name(name), Some(*element));
    }
}

#[test]
fn element_float_layout_matches_its_scalar_dtype() {
    // 跨轴不漂移：元素有标量对应且是浮点时，两条路必须读出同一个位域；`Int8` 两侧都为空。
    // 6 位格式没有 `DType`，只锁「有位域、能加满位宽」。
    for element in Element::ALL {
        let layout = element.float_layout();
        assert_eq!(layout, element.spec().float, "{element:?}");
        assert_eq!(
            element.exp_bits(),
            layout.map(|l| l.exp_bits),
            "{element:?}"
        );
        assert_eq!(
            element.mantissa_bits(),
            layout.map(|l| l.mantissa_bits),
            "{element:?}"
        );
        assert_eq!(
            element.exponent_bias(),
            layout.map(|l| l.exponent_bias),
            "{element:?}"
        );
        match element.dtype() {
            Some(dtype) => assert_eq!(
                layout,
                dtype.float_layout(),
                "{element:?} 的浮点位域与标量 {dtype:?} 分家"
            ),
            None => assert!(layout.is_some(), "{element:?} 是浮点，位域不能为空"),
        }
        if let Some(layout) = layout {
            assert_eq!(
                1 + layout.exp_bits + layout.mantissa_bits,
                element.bits(),
                "{element:?} 的浮点位域之和不等于位宽"
            );
        }
    }
}

#[test]
fn element_discriminant_fits_one_byte() {
    assert_eq!(size_of::<Element>(), 1, "repr(u8) 加 6 个元素应正好一字节");
}

#[test]
fn accessors_read_the_element_row() {
    // 行表锁字段，这里锁「取哪一列」：`Element::bits()`/`dtype()` 没有其它读取点。
    for element in Element::ALL {
        let spec = element.spec();
        assert_eq!(element.name(), spec.name);
        assert_eq!(element.variant(), spec.variant);
        assert_eq!(element.bits(), spec.bits);
        assert_eq!(element.dtype(), spec.dtype);
        assert_eq!(element.float_layout(), spec.float);
    }
}

#[test]
fn layout_table_is_locked() {
    // 一行一个格式：规范名、块长、块字节、元素、scale、家族。位宽由元素派生，不在这里
    // 重复一列；块字节是派生量，但派生公式的等价改写不该改变这里锁死的数值——事实先于
    // 公式。表格是给人逐行核对的，所以跳过格式化、保持一行一个格式。
    #[rustfmt::skip]
    const LOCKED: [(&str, u32, u32, Element, Scale, Family); MxType::ALL.len()] = [
        ("mxfp8_e4m3", 32, 33, Element::E4M3, Scale::E8M0, Family::Ocp),
        ("mxfp8_e5m2", 32, 33, Element::E5M2, Scale::E8M0, Family::Ocp),
        ("mxfp6_e3m2", 32, 25, Element::E3M2, Scale::E8M0, Family::Ocp),
        ("mxfp6_e2m3", 32, 25, Element::E2M3, Scale::E8M0, Family::Ocp),
        ("mxfp4_e2m1", 32, 17, Element::E2M1, Scale::E8M0, Family::Ocp),
        ("mxint8", 32, 33, Element::Int8, Scale::E8M0, Family::Ocp),
        ("nvfp4", 16, 9, Element::E2M1, Scale::E4M3, Family::Vendor),
    ];
    for (mx, (name, block_len, block_bytes, element, scale, family)) in
        MxType::ALL.iter().zip(LOCKED)
    {
        let spec = mx.spec();
        assert_eq!(
            (
                spec.name,
                spec.block_len,
                mx.block_bytes(),
                spec.element,
                spec.scale,
                spec.family,
            ),
            (name, block_len, block_bytes, element, scale, family),
            "{mx:?} 的布局行被改动"
        );
    }
    // 位宽是元素的函数：元素一旦锁死，这三个 `bits()` 就只能是 4/6/8。
    assert_eq!(
        (
            MxType::MxFp4E2M1.bits(),
            MxType::MxFp6E3M2.bits(),
            MxType::MxInt8.bits(),
        ),
        (4, 6, 8)
    );
}

#[test]
fn bytes_for_pads_to_whole_blocks() {
    assert_eq!(MxType::MxFp4E2M1.bytes_for(32), 17);
    assert_eq!(MxType::MxFp4E2M1.bytes_for(64), 34);
    assert_eq!(MxType::MxFp4E2M1.bytes_for(31), 17, "不足一块按整块计");
    assert_eq!(MxType::NvFp4.bytes_for(16), 9);
    // GGML 的 NVFP4 容器按 64 元素成块，4 个原生微块正好 36 字节。
    assert_eq!(MxType::NvFp4.bytes_for(64), 36);
}
