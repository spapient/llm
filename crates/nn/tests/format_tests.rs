use super::*;

/// 一档精度的三件事实：绑定的算子名、按分组拼格式的构造器、默认格式。
type Rung = (&'static str, fn(usize) -> WeightFormat, WeightFormat);

/// 「分组对得上才选得到算子」是三档共用的一条契约；分组一律从算子语义表读出。
#[test]
fn a_declared_group_is_the_only_one_that_reaches_an_operator() {
    let rungs: [Rung; 3] = [
        (crate::Linear::I2, WeightFormat::int2, WeightFormat::INT2),
        (crate::Linear::OP, WeightFormat::int4, WeightFormat::INT4),
        (crate::Linear::I8, WeightFormat::int8, WeightFormat::INT8),
    ];
    for (op_name, with_group, default) in rungs {
        let group = nn::registry::model(op_name)
            .unwrap_or_else(|| panic!("{op_name} 有语义表"))
            .group as usize;
        assert_eq!(
            crate::Linear::operator_for(with_group(group)),
            Some(op_name),
            "{op_name}：与算子声明的分组一致时才选得到算子"
        );
        assert_eq!(
            crate::Linear::operator_for(default),
            Some(op_name),
            "{op_name}：默认格式的分组必须就是内核声明的那一档"
        );
        assert_eq!(
            crate::Linear::operator_for(with_group(group * 2)),
            None,
            "{op_name}：分组不同就是另一个算子，不能套用"
        );
    }
}

#[test]
fn a_format_without_an_operator_says_so_instead_of_guessing() {
    for format in [
        WeightFormat::F16,
        WeightFormat::BF16,
        WeightFormat::I8,
        WeightFormat::PACKED4,
    ] {
        assert_eq!(
            crate::Linear::operator_for(format),
            None,
            "{format} 还没有已声明的算子"
        );
        assert!(
            !crate::Linear::unpriced_reason_for(format).is_empty(),
            "取不到算子时必须说清缺什么"
        );
    }
}

/// f32 有稠密算子，而且不能借用 int4 的算子名。
#[test]
fn a_dense_format_reaches_the_dense_operator() {
    assert_eq!(
        crate::Linear::operator_for(WeightFormat::F32),
        Some(crate::Linear::DENSE)
    );
    assert_eq!(WeightFormat::F32.dtype(), DType::F32);
    assert_ne!(
        crate::Linear::operator_for(WeightFormat::F32),
        Some(crate::Linear::OP),
        "稠密与量化是两条算子，不能共用"
    );
}

/// 每个映射到的算子名都必须在注册表里真实存在。
#[test]
fn every_mapped_operator_name_is_declared() {
    for format in [WeightFormat::F32, WeightFormat::INT4, WeightFormat::INT8] {
        let operator = crate::Linear::operator_for(format).expect("这三种格式都有算子");
        assert!(
            prof::catalogue::named(operator).is_some(),
            "{format} 映射到未注册的算子 `{operator}`"
        );
    }
}

/// int2 的分组不匹配诊断、存储类型和无分组视图必须来自同一格式。
#[test]
fn the_int2_rung_is_packed_and_says_which_group_mismatches() {
    let group = nn::registry::model(crate::Linear::I2)
        .expect("gemv_i2 有语义表")
        .group as usize;
    assert_eq!(
        crate::Linear::unpriced_reason_for(WeightFormat::int2(group * 2)),
        format!(
            "按 {group_double} 个元素一组量化（int2），而内核声明的是 {group} 个元素一组：分组不同不能共用同一条内核",
            group_double = group * 2
        ),
        "缺什么要说清是分组对不上，而不是笼统说没有算子"
    );
    assert_eq!(WeightFormat::INT2.dtype(), DType::U2);
    assert_eq!(
        WeightFormat::of_dtype(DType::U2),
        Some(WeightFormat::PACKED2)
    );
    assert_eq!(WeightFormat::PACKED2.to_string(), "2 位打包（未声明分组）");
}

/// 2 位权重能真的建出一个带下界的层。
#[test]
fn int2_weights_give_a_linear_a_real_lower_bound() {
    use crate::Contract;
    let layer = crate::Linear::with_format("q", WeightFormat::INT2, 512, 64).expect("合法");
    assert_eq!(layer.spca().ops(), [crate::Linear::I2]);
    let floor = layer.bounds().expect("合法").expect("有算子");
    assert!(floor.read() > 0 && floor.flops() > 0.0);
}
