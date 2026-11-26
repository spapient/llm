use crate::std::string::ToString;

const FORMATS: [(WeightFormat, DType); 9] = [
    (WeightFormat::F32, DType::F32),
    (WeightFormat::F16, DType::F16),
    (WeightFormat::BF16, DType::BF16),
    (WeightFormat::I8, DType::I8),
    (WeightFormat::PACKED2, DType::U2),
    (WeightFormat::PACKED4, DType::U4),
    (WeightFormat::int2(32), DType::U2),
    (WeightFormat::int4(32), DType::U4),
    (WeightFormat::int8(32), DType::I8),
];

testkit::test_cases! {
fn every_element_type_maps_to_a_format_and_back() {
    for dtype in DType::ALL.iter().copied() {
        if let Some(format) = WeightFormat::of_dtype(dtype) {
            assert_eq!(format.dtype(), dtype, "{dtype:?} 往返");
        }
    }
    assert_eq!(
        WeightFormat::of_dtype(DType::U4),
        Some(WeightFormat::PACKED4)
    );
}

fn every_weight_format_has_an_explicit_dtype() {
    for (format, dtype) in FORMATS {
        assert_eq!(format.dtype(), dtype, "{format} 的存储类型必须显式声明");
    }
    assert_eq!(std::mem::size_of::<WeightFormat>(), 16);
}

fn int8_elements_take_one_byte_each() {
    assert_eq!(WeightFormat::INT8.dtype(), DType::I8);
    assert_eq!(WeightFormat::INT8.dtype().bits(), 8);
}

fn the_display_form_is_stable() {
    assert_eq!(WeightFormat::F32.to_string(), "f32");
    assert_eq!(WeightFormat::INT4.to_string(), "int4（分组 32）");
    assert_eq!(WeightFormat::INT8.to_string(), "int8（分组 32）");
    assert_eq!(WeightFormat::PACKED2.to_string(), "2 位打包（未声明分组）");
}

fn grouped_formats_expose_dtype_and_group() {
    assert_eq!(WeightFormat::INT2.dtype(), DType::U2);
    assert_eq!(WeightFormat::INT4.dtype(), DType::U4);
    assert_eq!(WeightFormat::INT8.dtype(), DType::I8);
    assert_eq!(WeightFormat::INT2.group(), Some(32));
    assert_eq!(WeightFormat::INT4.group(), Some(32));
    assert_eq!(WeightFormat::INT8.group(), Some(32));
    assert_eq!(WeightFormat::int4(0).group(), Some(0));
    assert_eq!(WeightFormat::PACKED4.group(), None);
    assert_eq!(WeightFormat::F32.group(), None);
}

fn quantized_bit_widths_are_explicit() {
    for (bits, format) in [
        (2, WeightFormat::int2(32)),
        (4, WeightFormat::int4(32)),
        (8, WeightFormat::int8(32)),
    ] {
        assert_eq!(WeightFormat::from_quantized(bits, 32), Some(format));
    }
    for bits in [0, 1, 3, 16, u8::MAX] {
        assert_eq!(WeightFormat::from_quantized(bits, 32), None);
    }
    assert_eq!(
        WeightFormat::from_quantized(4, 0),
        Some(WeightFormat::int4(0))
    );
}

#[cfg(feature = "serde")]
fn toml_wire_formats_are_stable_and_round_trip() {
    for (format, _) in FORMATS {
        let encoded = toml::to_string(&format).expect("权重编码可序列化");
        let decoded: WeightFormat = toml::from_str(&encoded).expect("权重编码可反序列化");
        assert_eq!(decoded, format, "配置往返：{encoded}");
    }
    assert_eq!(
        toml::to_string(&WeightFormat::INT4).expect("int4 配置"),
        "type = \"int4\"\ngroup = 32\n"
    );
    assert_eq!(
        toml::to_string(&WeightFormat::PACKED4).expect("packed4 配置"),
        "type = \"packed4\"\n"
    );
    assert!(toml::from_str::<WeightFormat>("type = \"int4\"\n").is_err());
    assert!(toml::from_str::<WeightFormat>("type = \"f32\"\ngroup = 32\n").is_err());
}
}
