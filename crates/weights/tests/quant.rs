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

testkit::test_cases! {
fn per_block_and_per_tensor_schemes_round_trip() {
    let block = Quant::new(2, 1, 32, false, Wire::F16, None).expect("per-block");
    assert_eq!(round(&block, 2), block);

    let tensor = Quant::new(2, 2, 0, true, Wire::F32, Some(Wire::I8)).expect("per-tensor+zp");
    assert_eq!(round(&tensor, 2), tensor);
}

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

fn a_disagreeing_output_dimension_is_an_error_not_a_guess() {
    let header = header(vec![
        packed("a.weight", 8, 64),
        scales("a.scales", 4, 2, DType::F16),
    ]);
    let error = header.quantization("a.weight").expect_err("输出维不等");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("输出维 4"), "{error}");
}

fn a_group_that_does_not_divide_the_input_is_an_error() {
    let header = header(vec![
        packed("a.weight", 8, 100),
        scales("a.scales", 8, 3, DType::F16),
    ]);
    let error = header.quantization("a.weight").expect_err("100 / 3");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("不是它的整数倍"), "{error}");
}

fn a_scale_that_is_not_rank_two_is_an_error() {
    let header = header(vec![
        packed("a.weight", 8, 64),
        Entry::new("a.scales", DType::F16, Shape::new([8])),
    ]);
    let error = header.quantization("a.weight").expect_err("秩 1");
    assert!(error.to_string().contains("秩 2"), "{error}");
}

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

fn a_wide_element_type_is_not_four_bit_quantization() {
    let header = header(vec![
        Entry::new("a.weight", DType::F16, Shape::new([8, 128])),
        scales("a.scales", 8, 4, DType::F16),
    ]);
    assert_eq!(header.quantization("a.weight").expect("自洽"), None);
}

fn an_integer_scale_is_another_scheme_rather_than_a_packed_four_bit_one() {
    let header = header(vec![
        packed("a.weight", 8, 64),
        scales("a.scales", 8, 2, DType::I8),
    ]);
    assert_eq!(header.quantization("a.weight").expect("自洽"), None);
}

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

fn reserved_quant_flag_bits_are_rejected() {
    let mut buf = vec![0, 0, 32, 0x02];
    buf.push(u8::try_from(Wire::F16.code()).expect("F16 码位单字节"));

    let error = Quant::decode(&mut Cursor::new(&buf), 2).expect_err("预留位非零必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

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
}

// ---- 自适应位宽分配 ----

testkit::test_cases! {
/// 档位表必须按位宽升序覆盖 `{2,3,4,8}`：降档贪心假定 `LADDER[i]` 的位宽随下标单调递增。
fn ladder_widths_ascend_and_cover_the_supported_bits() {
    let widths: Vec<u8> = LADDER.iter().map(|scheme| scheme.bits).collect();
    assert_eq!(widths, vec![2, 3, 4, 8]);
}

/// 分组区间：末组按剩余元素收窄，不越界也不丢元素。
fn spans_cover_the_input_exactly() {
    assert_eq!(span(4, 10, 0), 0..4);

    assert_eq!(span(4, 10, 2), 8..10);

    assert_eq!(span(4, 8, 1), 4..8);
}

/// 极紧预算下所有组停在最宽档：`I8` 也有均匀量化误差，零预算因此一律未达标。
///
/// 「停在 8bit」与「达标」是两件事：预估误差是 `n·scale²/12`，`scale` 只在 `max_abs = 0` 时
/// 才为零，所以零预算下没有一档能过——把它读成「8bit 无损」会掩盖这一步其实没达标。
fn tight_budget_keeps_every_group_wide() {
    let x = [0.5f32, -0.25, 1.0, -2.0];

    let adaptive = adaptive_quantize(2, &x, 0.0).expect("合法输入");

    assert_eq!(adaptive.widths, vec![8, 8]);

    assert!(!adaptive.met);

    assert_eq!(adaptive.bpw, 8.0);

    // 预算为零 ⇒ 全部停在 `I8`，逐位等于父模块的固定 `I8` 往返（不是「近似无损」）。

    let (packed, scales) = quantize(I8, 2, &x).expect("合法输入");

    let mut expect = vec![0.0f32; x.len()];

    dequantize(I8, &packed, &scales, 2, &mut expect).expect("合法输入");

    assert_eq!(adaptive.recon, expect);
}

/// 极宽预算下所有组降到最窄档：贪心没有「只降一档就停」的口子。
fn loose_budget_reaches_the_narrowest_rung() {
    let x = [0.5f32, -0.25, 1.0, -2.0];

    let adaptive = adaptive_quantize(2, &x, 1e9).expect("合法输入");

    assert_eq!(adaptive.widths, vec![2, 2]);

    assert!(adaptive.met);

    assert_eq!(adaptive.bpw, 2.0);
}

/// 预算不可能达标时 `met = false`，位宽图仍给出（调用方据此决定拒压，而不是拿到空结果）。
fn unmet_budget_still_reports_the_map() {
    let x = [0.5f32, -0.25, 1.0, -2.0];

    let adaptive = adaptive_quantize(2, &x, 0.0).expect("合法输入");

    let tight = adaptive_quantize(2, &x, 1e-12).expect("合法输入");

    assert!(!tight.met);

    assert_eq!(tight.widths, adaptive.widths);
}

/// 预算越宽，选出的位宽只会更低：单调性是贪心可用的前提。
fn wider_budgets_never_pick_wider_widths() {
    let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();

    let mut previous = u8::MAX;

    for budget in [0.0f64, 0.5, 2.0, 8.0, 32.0] {
        let adaptive = adaptive_quantize(8, &x, budget).expect("合法输入");

        let widest = adaptive.widths.iter().copied().max().expect("非空");

        assert!(widest <= previous, "预算 {budget} 反而选了更宽的档");

        previous = widest;
    }
}

/// 三项入口契约各自独立失败：分组为零、非有限输入、负预算。
fn invalid_inputs_are_rejected_before_any_allocation() {
    assert!(adaptive_quantize(0, &[1.0f32], 1.0).is_err());

    assert!(adaptive_quantize(2, &[1.0f32, f32::NAN], 1.0).is_err());

    assert!(adaptive_quantize(2, &[1.0f32, 2.0], -0.5).is_err());

    assert!(LADDER.iter().all(|s| s.max_code > 0.0));
}

/// 补偿形态在无谱隙时拒压，而不是把满秩矩阵当成低秩。
fn compensation_refuses_without_a_spectral_gap() {
    let w = [1.0f32, 0.0, 0.0, 1.0];

    assert!(
        compensate_adaptive(2, &w, 2, 2, 20.0)
            .expect("合法形状")
            .is_none()
    );
}

/// 目标 SNR 非有限在入口被拒：`+inf` 会静默退化成零误差预算。
fn non_finite_target_snr_is_rejected() {
    let w = [1.0f32, 0.0, 0.0, 1.0];

    assert!(compensate_adaptive(2, &w, 2, 2, f64::INFINITY).is_err());

    assert!(compensate_adaptive(2, &w, 2, 2, f64::NAN).is_err());
}
}

/// 确定性合成矩阵：`r` 条 rank-1 项加正弦噪声（固定公式，无随机源）。
///
/// E5 明确「不得使用合成矩阵代替模型级结果」，所以这条只作**机制探针**：它量的是「位宽图与
/// 平均码位是否随预算单调、重构 SNR 是否达到同一预算」，不出 E5 的判决。
fn synthetic(m: usize, n: usize, rank: usize, noise: f32) -> Vec<f32> {
    let mut w = vec![0.0f32; m * n];

    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;

            for k in 0..rank {
                let a = ((row * (k + 1)) as f32 * 0.017).sin();

                let b = ((col * (k + 1)) as f32 * 0.013).cos();

                acc += a * b;
            }

            let jitter = ((row * 31 + col * 17) as f32 * 0.001).sin() * noise;

            w[row * n + col] = acc + jitter;
        }
    }

    w
}

/// 实测探针：一层合成权重上的秩、位宽图、平均码位与**权重空间** SNR。
///
/// 命令：`cargo test -p llm-weights --release --lib -- --ignored e5_width_map --nocapture`。
///
/// 报的 SNR 是权重空间口径——E5 禁止拿它替代输出空间 SNR，这里只用来确认「预算越紧、位宽
/// 越宽、SNR 越高」这条单调关系在实现里真的成立。
#[test]
#[ignore = "实测探针：打印一层合成权重的位宽图与权重空间 SNR，不参与默认门禁"]
fn e5_width_map() {
    let (m, n, group) = (512usize, 512usize, 32usize);

    let w = synthetic(m, n, 8, 0.01);

    let energy: f64 = w.iter().map(|&v| f64::from(v) * f64::from(v)).sum();

    println!("E5 探针：{m}×{n}（rank-8 + 噪声），group={group}，‖W‖²_F={energy:.1}");

    for target in [30.0f64, 40.0, 50.0, 60.0] {
        let Some(card) = compensate_adaptive(group, &w, m, n, target).expect("合法输入") else {
            println!("  target {target} dB：无谱隙，拒压");

            continue;
        };

        let residual: f64 = card
            .recon
            .iter()
            .zip(&w)
            .map(|(&a, &b)| {
                let d = f64::from(a) - f64::from(b);

                d * d
            })
            .sum();

        let snr = 10.0 * (energy / residual.max(f64::MIN_POSITIVE)).log10();

        let mut histogram = [0usize; 4];

        for width in &card.widths {
            let slot = match width {
                2 => 0,

                3 => 1,

                4 => 2,

                _ => 3,
            };

            histogram[slot] += 1;
        }

        println!(
            "  target {target} dB：rank={} gap={:.3e} sigma1={:.1} bpw={:.3} met={} 位宽图 2/3/4/8 = {}/{}/{}/{}，权重空间 SNR={snr:.2} dB",
            card.rank,
            card.gap,
            card.sigma1,
            card.bpw,
            card.met,
            histogram[0],
            histogram[1],
            histogram[2],
            histogram[3]
        );

        assert!(
            card.met,
            "目标 {target} dB 未达标：位宽图仍给出，但预算没满足"
        );
    }
}
