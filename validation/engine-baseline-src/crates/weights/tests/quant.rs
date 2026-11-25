//! `quant` 的单元测试：方案自洽与量化条目配对两条路径。
use super::*;
use tensor::Shape;

fn round(quant: &Quant, rank: u32) -> Quant {
    let mut buf = Vec::new();
    quant.encode(&mut buf, rank).expect("编码合法");
    Quant::decode(&mut Cursor::new(&buf), rank).expect("解码合法")
}

fn header(entries: Vec<Entry>) -> Header {
    // 夹具按顺序把区间摆开：`Header::new` 要求区间不重叠，默认全 0 会让每个用例都失败在
    // 夹具而不是被测逻辑上。
    let mut offset = 0;
    let laid_out = entries
        .into_iter()
        .map(|entry| {
            let start = offset;
            offset += entry.len();
            entry.at(start)
        })
        .collect();
    Header::new(laid_out, 0).expect("测试夹具的区间合法")
}

/// 一个 4 位打包的条目：`[out, in]`，字节数是 `ceil(out × in / 2)`（位数组，按行不补齐）。
fn packed(name: &str, out: usize, in_features: usize) -> Entry {
    let entry = Entry::new(name, DType::U4, Shape::new([out, in_features]));
    assert_eq!(
        entry.len(),
        (out * in_features).div_ceil(2),
        "夹具自证打包口径"
    );
    entry
}

fn scales(name: &str, out: usize, groups: usize, dtype: DType) -> Entry {
    Entry::new(name, dtype, Shape::new([out, groups]))
}

/// 真实文件的打包条目：`U8` 一个字节两个 nibble，第二维是字节数。
fn nibbles(name: &str, out: usize, in_features: usize) -> Entry {
    Entry::new(name, DType::U8, Shape::new([out, in_features / 2]))
}

#[test]
fn per_block_and_per_tensor_schemes_round_trip() {
    let block = Quant::new(2, 1, 32, false, Wire::F16, None).expect("per-block");
    assert_eq!(round(&block, 2), block);

    let tensor = Quant::new(2, 2, 0, true, Wire::F32, Some(Wire::I8)).expect("per-tensor+zp");
    assert_eq!(round(&tensor, 2), tensor);
}

#[test]
fn inconsistent_schemes_are_rejected() {
    assert!(
        Quant::new(2, 1, 0, false, Wire::F16, None).is_err(),
        "block=0 沿轴"
    );
    assert!(
        Quant::new(2, 2, 4, false, Wire::F16, None).is_err(),
        "per-tensor 带 block"
    );
    assert!(
        Quant::new(2, 1, 32, false, Wire::I8, None).is_err(),
        "整数标度"
    );
    assert!(
        Quant::new(2, 1, 32, true, Wire::F16, None).is_err(),
        "缺零点类型"
    );
    assert!(
        Quant::new(2, 1, 32, true, Wire::F16, Some(Wire::F32)).is_err(),
        "浮点零点"
    );
}

#[test]
fn the_group_comes_from_the_scale_shape() {
    let header = header(vec![
        packed("blk.0.q.weight", 8, 128),
        scales("blk.0.q.scales", 8, 4, DType::F16),
    ]);
    let quantized = header
        .quantization("blk.0.q.weight")
        .expect("自洽")
        .expect("是量化");
    assert_eq!(quantized.group(), 32, "128 个输入 / 4 个分组");
    assert_eq!(quantized.bits(), 4);
    assert_eq!(quantized.scales().name(), "blk.0.q.scales");
    assert_eq!(header.scales_names().expect("自洽"), ["blk.0.q.scales"]);
}

#[test]
fn all_four_naming_families_are_recognised() {
    for (weight, scale) in [
        ("a.weight", "a.weight_scale"),
        ("a.weight", "a.weight_scales"),
        ("a.weight", "a.scales"),
        ("a.weight", "a.scale"),
    ] {
        let header = header(vec![packed(weight, 4, 64), scales(scale, 4, 2, DType::F32)]);
        let quantized = header
            .quantization(weight)
            .expect("自洽")
            .unwrap_or_else(|| panic!("`{scale}` 应被认出"));
        assert_eq!(quantized.group(), 32, "`{scale}`");
    }
}

/// 字节数是结构性的：它由 `shape × dtype` 推出，本层不需要再校验一遍。钉成断言，免得
/// 日后有人真去加一条"拿一个值与自己比"的检查，并误以为它在挡 int8 那种歧义。
///
/// 宽度为奇数时口径是 `ceil(out × in / 2)`，不是 `out × ceil(in / 2)`：这里把 4 位当作一整块
/// 紧贴的位数组，**按行不补齐**。按行补齐的导出格式会与这条不符——那正是另一种打包方案，
/// 形状上看不出差异，只能由上层用分组与内核声明去拒。
#[test]
fn the_packed_byte_count_is_structural_rather_than_checked_here() {
    for (out, in_features) in [(8, 128), (8, 7), (3, 1)] {
        let entry = Entry::new("a.weight", DType::U4, Shape::new([out, in_features]));
        assert_eq!(
            entry.len(),
            (out * in_features).div_ceil(2),
            "{out} × {in_features} 个 4 位元素"
        );
    }
    // 同一个张量用 8 位书写时，字节数恰好是上面那个数（`in` 为偶数时两者一致）。
    assert_eq!(nibbles("a.weight", 8, 128).len(), 8 * 64);
}

#[test]
fn an_entry_without_a_scale_partner_is_not_quantized() {
    let header = header(vec![packed("a.weight", 4, 64)]);
    assert_eq!(header.quantization("a.weight").expect("无伙伴"), None);
    assert!(header.scales_names().expect("自洽").is_empty());
    assert_eq!(
        header.quantization("absent").expect("不是条目"),
        None,
        "名字不在表里不是错误"
    );
}

#[test]
fn a_disagreeing_output_dimension_is_an_error_not_a_guess() {
    let header = header(vec![
        packed("a.weight", 8, 64),
        scales("a.scales", 4, 2, DType::F16),
    ]);
    let error = header.quantization("a.weight").expect_err("输出维不等");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("输出维 4"), "{error}");
}

#[test]
fn a_group_that_does_not_divide_the_input_is_an_error() {
    let header = header(vec![
        packed("a.weight", 8, 100),
        scales("a.scales", 8, 3, DType::F16),
    ]);
    let error = header.quantization("a.weight").expect_err("100 / 3");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("不是它的整数倍"), "{error}");
}

#[test]
fn a_scale_that_is_not_rank_two_is_an_error() {
    let header = header(vec![
        packed("a.weight", 8, 64),
        Entry::new("a.scales", DType::F16, Shape::new([8])),
    ]);
    let error = header.quantization("a.weight").expect_err("秩 1");
    assert!(error.to_string().contains("秩 2"), "{error}");
}

#[test]
fn the_real_file_shape_reports_the_same_group_as_the_unpacked_view() {
    // 同一层权重，两种书写：解包视图（U4，第二维 = 元素数）与真实文件（U8 打包，第二维 = 字节数）。
    let unpacked = header(vec![
        packed("a.weight", 8, 128),
        scales("a.scales", 8, 4, DType::F16),
    ]);
    let packed_bytes = header(vec![
        nibbles("a.weight", 8, 128),
        scales("a.scales", 8, 4, DType::F16),
    ]);
    let from_unpacked = unpacked
        .quantization("a.weight")
        .expect("自洽")
        .expect("量化");
    let from_file = packed_bytes
        .quantization("a.weight")
        .expect("自洽")
        .expect("量化");
    assert_eq!(from_unpacked.group(), 32);
    assert_eq!(from_file.group(), 32, "U8 条目的第二维是字节数");
    assert_eq!(from_file.in_features(), 128, "宽度是元素数，不是字节数");
    assert_eq!(from_file.out_features(), 8);
    assert_eq!(from_unpacked.in_features(), from_file.in_features());
}

#[test]
fn a_signed_byte_entry_is_one_element_per_byte() {
    // 逐元素 int8：第二维就是元素数，宽度不翻倍，分组按元素数推。
    let weight = Entry::new("a.weight", DType::I8, Shape::new([8, 64]));
    let header = header(vec![weight, scales("a.scales", 8, 2, DType::F16)]);
    let quantized = header
        .quantization("a.weight")
        .expect("自洽")
        .expect("逐元素 int8");
    assert_eq!(quantized.bits(), 8);
    assert_eq!(quantized.in_features(), 64);
    assert_eq!(quantized.group(), 32, "64 元素 / 2 组 = 32");
}

#[test]
fn an_unsigned_byte_entry_packs_two_nibbles_per_byte() {
    // 无符号字节打包：一个字节两个 nibble，宽度按字节数翻倍。
    let weight = Entry::new("a.weight", DType::U8, Shape::new([8, 32]));
    let header = header(vec![weight, scales("a.scales", 8, 2, DType::F16)]);
    let quantized = header
        .quantization("a.weight")
        .expect("自洽")
        .expect("打包 nibble");
    assert_eq!(quantized.bits(), 4);
    assert_eq!(quantized.in_features(), 64, "32 字节 ×2 = 64 个 nibble");
    assert_eq!(quantized.group(), 32, "64 元素 / 2 组 = 32");
}

#[test]
fn a_wide_element_type_is_not_four_bit_quantization() {
    let header = header(vec![
        Entry::new("a.weight", DType::F16, Shape::new([8, 128])),
        scales("a.scales", 8, 4, DType::F16),
    ]);
    assert_eq!(header.quantization("a.weight").expect("自洽"), None);
}

#[test]
fn an_integer_scale_is_another_scheme_rather_than_a_packed_four_bit_one() {
    let header = header(vec![
        packed("a.weight", 8, 64),
        scales("a.scales", 8, 2, DType::I8),
    ]);
    assert_eq!(header.quantization("a.weight").expect("自洽"), None);
}

#[test]
fn the_same_entry_is_never_reported_as_two_layers() {
    // `scales_names` 与 `quantization` 用同一条规则，所以标度必然能被精确排除。
    let header = header(vec![
        packed("x.weight", 4, 32),
        scales("x.weight_scale", 4, 1, DType::F16),
        packed("y.weight", 4, 32),
        scales("y.scale", 4, 1, DType::F16),
    ]);
    let mut names = header.scales_names().expect("自洽");
    names.sort_unstable();
    assert_eq!(names, ["x.weight_scale", "y.scale"]);
}

#[test]
fn affine_scheme_wire_bytes_are_stable() {
    let quant = Quant::new(2, 0, 32, true, Wire::F16, Some(Wire::I8)).expect("合法方案");
    let mut buf = Vec::new();
    quant.encode(&mut buf, 2).expect("编码");

    // scheme(0) + axis(0) + block(32) + flags(1) + scale(F16) + zero-point(I8)。
    let expected = [
        0,
        0,
        32,
        1,
        u8::try_from(Wire::F16.code()).expect("F16 码位单字节"),
        u8::try_from(Wire::I8.code()).expect("I8 码位单字节"),
    ];
    assert_eq!(buf, expected);
    assert_eq!(
        Quant::decode(&mut Cursor::new(&buf), 2).expect("解码"),
        quant
    );
}

#[test]
fn reserved_quant_flag_bits_are_rejected() {
    let mut buf = vec![0, 0, 32, 0x02];
    buf.push(u8::try_from(Wire::F16.code()).expect("F16 码位单字节"));

    let error = Quant::decode(&mut Cursor::new(&buf), 2).expect_err("预留位非零必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn quantized_entries_produce_shared_weight_format() {
    for (weight, scale, expected) in [
        (
            packed("a.weight", 8, 64),
            scales("a.scales", 8, 2, DType::F16),
            WeightFormat::int4(32),
        ),
        (
            Entry::new("b.weight", DType::I8, Shape::new([8, 64])),
            scales("b.scales", 8, 2, DType::F16),
            WeightFormat::int8(32),
        ),
        (
            nibbles("c.weight", 8, 64),
            scales("c.scales", 8, 2, DType::F16),
            WeightFormat::int4(32),
        ),
    ] {
        let name = weight.name().to_string();
        let header = header(vec![weight, scale]);
        let quantized = header
            .quantization(&name)
            .expect("文件自洽")
            .expect("量化条目");
        assert_eq!(quantized.weight_format(), Some(expected));
    }
}
