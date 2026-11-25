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
fn every_weight_format_has_an_explicit_dtype() {
    let formats = [
        (WeightFormat::F32, DType::F32),
        (WeightFormat::F16, DType::F16),
        (WeightFormat::Bf16, DType::BF16),
        (WeightFormat::I8, DType::I8),
        (WeightFormat::Packed2, DType::U2),
        (WeightFormat::Packed4, DType::U4),
        (WeightFormat::Int2 { group: 32 }, DType::U2),
        (WeightFormat::Int4 { group: 32 }, DType::U4),
        (WeightFormat::Int8 { group: 32 }, DType::I8),
    ];
    for (format, dtype) in formats {
        assert_eq!(format.dtype(), dtype, "{format} 的存储类型必须显式声明");
    }
}

/// 一档精度的三件事实：绑定的算子名、按分组拼格式的构造器、默认格式。
type Rung = (&'static str, fn(usize) -> WeightFormat, WeightFormat);

/// 「分组对得上才选得到算子」是三档共用的一条契约，三处拷贝此前各写一遍同样的三步。
/// 收成一张表还有个额外好处：多一档精度时这张表会立刻要求补一行，而不是静默漏掉。
/// 各档自己的差异（元素类型、默认分组、诊断文字）留在各自那一条用例里。
#[test]
fn a_declared_group_is_the_only_one_that_reaches_an_operator() {
    // 分组一律从算子语义表读出来，不在用例里再断言一个 32。
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
            with_group(group).operator(),
            Some(op_name),
            "{op_name}：与算子声明的分组一致时才选得到算子"
        );
        assert_eq!(
            default.operator(),
            Some(op_name),
            "{op_name}：默认格式的分组必须就是内核声明的那一档"
        );
        assert_eq!(
            with_group(group * 2).operator(),
            None,
            "{op_name}：分组不同就是另一个算子，不能套用"
        );
    }
}

/// int8 这一档独有的是元素类型：它不打包，一个元素就是一字节。
#[test]
fn int8_elements_take_one_byte_each() {
    let group = nn::registry::model(crate::Linear::I8)
        .expect("gemv_i8 有语义表")
        .group as usize;
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

/// 每个映射到的算子名都必须在 `nn` 注册表里真的存在：否则 `ContractSpec::new` 会在构造时
/// 以 `missing` 拒绝，这里提前把漂移钉在测试里。
#[test]
fn every_mapped_operator_name_is_declared() {
    let formats = [WeightFormat::F32, WeightFormat::INT4, WeightFormat::INT8];
    for format in formats {
        let operator = format.operator().expect("这三种格式都有算子");
        assert!(
            prof::catalogue::named(operator).is_some(),
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

/// 2 位这一档此前在格式轴上不存在，而 `gemv_i2` / `gemm_i2` / `dequant_i2` 三个算子早已注册：
/// 算子表里有、层侧表达不出，正是「精度档缺一级」这种缺口的形态。这条用例锁住它被补上。
#[test]
fn the_int2_rung_is_packed_and_says_which_group_mismatches() {
    let group = nn::registry::model(crate::Linear::I2)
        .expect("gemv_i2 有语义表")
        .group as usize;
    assert_eq!(
        WeightFormat::int2(group * 2).unpriced_reason(),
        format!(
            "按 {group_double} 个元素一组量化（int2），而内核声明的是 {group} 个元素一组：分组不同不能共用同一条内核",
            group_double = group * 2
        ),
        "缺什么要说清是分组对不上，而不是笼统说没有算子"
    );
    assert_eq!(
        WeightFormat::INT2.dtype(),
        DType::U2,
        "2 位打包的元素类型就是 U2"
    );
    assert_eq!(
        WeightFormat::of_dtype(DType::U2),
        Some(WeightFormat::Packed2),
        "元素类型 → 格式这一向也要能认出 2 位打包"
    );
    assert_eq!(
        WeightFormat::Packed2.to_string(),
        "2 位打包（未声明分组）",
        "没有分组的打包要说清「未声明」，不能退化成元素类型名 u2"
    );
}

/// 2 位权重能真的建出一个带下界的层：格式轴补上这一档之后，`Linear` 立刻可计价。
#[test]
fn int2_weights_give_a_linear_a_real_lower_bound() {
    use crate::Contract;
    let layer = crate::Linear::with_format("q", WeightFormat::INT2, 512, 64).expect("合法");
    assert_eq!(layer.spca().ops(), [crate::Linear::I2]);
    let floor = layer.bounds().expect("合法").expect("有算子");
    assert!(floor.read() > 0 && floor.flops() > 0.0);
}
