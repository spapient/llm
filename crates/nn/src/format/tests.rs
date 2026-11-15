use super::*;

#[test]
fn every_element_type_maps_to_a_format_and_back() {
    for dtype in DType::ALL.iter().copied() {
        if let Some(format) = WeightFormat::of_dtype(dtype) {
            assert_eq!(format.dtype(), dtype, "{dtype:?} 往返");
        }
    }
    assert_eq!(
        WeightFormat::of_dtype(DType::U4),
        Some(WeightFormat::Packed4)
    );
}

#[test]
fn only_the_declared_int4_group_reaches_an_operator() {
    // 分组来自算子语义表：这里把它读出来而不是再断言一个 32。
    let group = op::model(crate::Linear::OP).expect("算子有语义表").group as usize;
    assert_eq!(
        WeightFormat::int4(group).operator(),
        Some(crate::Linear::OP),
        "与算子声明的分组一致时才选得到算子"
    );
    assert_eq!(WeightFormat::INT4.operator(), Some(crate::Linear::OP));
    assert_eq!(
        WeightFormat::int4(group * 2).operator(),
        None,
        "分组不同就是另一个算子，不能套用"
    );
}

#[test]
fn only_the_declared_int8_group_reaches_an_operator() {
    let group = op::model(crate::Linear::I8)
        .expect("gemv_i8 有语义表")
        .group as usize;
    assert_eq!(
        WeightFormat::int8(group).operator(),
        Some(crate::Linear::I8),
        "与算子声明的分组一致时才选得到算子"
    );
    assert_eq!(WeightFormat::INT8.operator(), Some(crate::Linear::I8));
    assert_eq!(
        WeightFormat::int8(group * 2).operator(),
        None,
        "分组不同就是另一个算子，不能套用"
    );
    assert_eq!(
        WeightFormat::int8(group).dtype(),
        DType::I8,
        "int8 逐元素存一字节"
    );
}

#[test]
fn a_format_without_an_operator_says_so_instead_of_guessing() {
    for format in [
        WeightFormat::F16,
        WeightFormat::Bf16,
        WeightFormat::I8,
        WeightFormat::Packed4,
    ] {
        assert_eq!(format.operator(), None, "{format} 还没有已声明的算子");
        assert!(
            !format.unpriced_reason().is_empty(),
            "取不到算子时必须说清缺什么"
        );
    }
}

/// f32 现在有算子，而且就是稠密那一条：格式与算子名的对应不是注释，是可执行的断言。
#[test]
fn a_dense_format_reaches_the_dense_operator() {
    assert_eq!(WeightFormat::F32.operator(), Some(crate::Linear::DENSE));
    assert_eq!(WeightFormat::F32.dtype(), DType::F32);
    assert_ne!(
        WeightFormat::F32.operator(),
        Some(crate::Linear::OP),
        "稠密与量化是两条算子，不能共用"
    );
}

/// 每个映射到的算子名都必须在 `op` 注册表里真的存在：否则 `ContractSpec::new` 会在构造时
/// 以 `missing` 拒绝，这里提前把漂移钉在测试里。
#[test]
fn every_mapped_operator_name_is_declared() {
    let formats = [WeightFormat::F32, WeightFormat::INT4, WeightFormat::INT8];
    for format in formats {
        let operator = format.operator().expect("这三种格式都有算子");
        assert!(
            op::builtin_named(operator).is_some(),
            "{format} 映射到未注册的算子 `{operator}`"
        );
    }
}

#[test]
fn the_display_form_is_stable() {
    assert_eq!(WeightFormat::F32.to_string(), "f32");
    assert_eq!(WeightFormat::INT4.to_string(), "int4（分组 32）");
    assert_eq!(WeightFormat::INT8.to_string(), "int8（分组 32）");
}

#[test]
fn grouped_formats_map_back_to_their_storage_dtype() {
    assert_eq!(WeightFormat::INT4.dtype(), DType::U4);
    assert_eq!(WeightFormat::INT8.dtype(), DType::I8);
}
