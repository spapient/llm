//! `stack` 的用例。
//!
//! 生产文件只留执行与判据，用例集中于此；N3 行数门限不再受用例增删影响。

use super::*;
use crate::LmHead;
use tensor::{DType, Shape};

fn header(entries: Vec<Entry>) -> Header {
    Header::new(entries, 0).expect("测试夹具的区间合法")
}

fn entry(name: &str, dims: [usize; 2], offset: usize) -> Entry {
    Entry::new(name, DType::F32, Shape::new(dims)).at(offset)
}

fn packed(name: &str, dims: [usize; 2], offset: usize) -> Entry {
    Entry::new(name, DType::U4, Shape::new(dims)).at(offset)
}

#[test]
fn a_two_dimensional_entry_becomes_a_projection() {
    let model = Stack::from_header(&header(vec![entry("blk.0.attn_q.weight", [64, 32], 0)]))
        .expect("形状合法");
    let layers = model.layers();
    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0].name(), "blk.0.attn_q", "去掉 `.weight` 后缀");
    assert!(model.uncovered().is_empty());
}

#[test]
fn the_entry_dtype_decides_the_format_not_a_global_default() {
    let model = Stack::from_header(&header(vec![entry("blk.0.attn_q.weight", [64, 32], 0)]))
        .expect("形状合法");
    // 文件里写的是 f32：它由稠密算子计价，而不是按 int4 报一个 8 倍偏低的数。
    let floor = model.floor();
    assert!(floor.unpriced().is_empty(), "f32 现在有算子");
    let layer = &model.layers()[0];
    assert_eq!(
        layer.spca().ops(),
        [Linear::DENSE],
        "f32 权重选的是稠密那条算子"
    );
    assert_eq!(
        floor.total(),
        layer.bounds().expect("合法").expect("有算子"),
        "栈的总下界就是各层的和"
    );
    assert_ne!(
        floor.total(),
        Bounds::of_op(Linear::OP, 32, 64).expect("int4 语义"),
        "按 int4 计价会把 f32 权重报成八分之一"
    );
}

/// 一份全是 f32 的权重文件（真实 safetensors 最常见的形态）现在能全额计价。
#[test]
fn a_dense_model_is_priced_end_to_end() {
    let model = Stack::from_header(&header(vec![
        entry("blk.0.attn_q.weight", [64, 4096], 0),
        entry("blk.0.attn_k.weight", [64, 4096], 64 * 4096 * 4),
    ]))
    .expect("形状合法");
    let floor = model.floor();
    assert_eq!(model.layers().len(), 2);
    assert!(floor.unpriced().is_empty(), "{:?}", floor.unpriced());
    // 每层：权重 64×4096 个 f32 + 输入 4096 个 f32；两层相加。
    let per_layer = (64 * 4096 + 4096) as u64 * 4;
    assert_eq!(floor.total().read(), 2 * per_layer);
    assert_eq!(floor.total().flops(), 2.0 * 2.0 * 64.0 * 4096.0);
}

#[test]
fn a_packed_entry_without_a_group_is_reported_not_guessed() {
    let model = Stack::from_header(&header(vec![packed("blk.0.attn_q.weight", [64, 32], 0)]))
        .expect("形状合法");
    let floor = model.floor();
    assert_eq!(floor.total(), Bounds::ZERO);
    assert!(
        floor.unpriced()[0].reason().contains("分组"),
        "分组缺失必须写在原因里：{}",
        floor.unpriced()[0]
    );
}

#[test]
fn entries_that_are_not_matrices_are_reported_not_dropped() {
    let model = Stack::from_header(&header(vec![
        Entry::new("blk.0.norm.weight", DType::F32, Shape::new([64])).at(0),
        Entry::new("blk.0.attn_q.weight", DType::F16, Shape::new([64, 64])).at(256),
    ]))
    .expect("形状合法");
    assert_eq!(model.layers().len(), 1);
    assert_eq!(model.uncovered(), ["blk.0.norm.weight"]);
    // 未覆盖与无法计价是两件事：前者没进层清单，后者进了但推不出下界。
    let floor = model.floor();
    assert_eq!(floor.unpriced().len(), 1, "只有那条 16 位投影无法计价");
    assert!(
        floor.unpriced()[0].reason().contains("16 位"),
        "{}",
        floor.unpriced()[0]
    );
}

#[test]
fn the_model_floor_is_the_sum_of_its_layers() {
    let model = Stack::of(vec![
        Box::new(Linear::new("a", 32, 64).expect("合法")),
        Box::new(Linear::new("b", 64, 128).expect("合法")),
    ])
    .expect("名字唯一");
    let sum = model
        .layers()
        .iter()
        .map(|layer| layer.bounds().expect("合法").expect("有算子"))
        .fold(Bounds::ZERO, Bounds::plus);
    assert_eq!(model.floor().total(), sum);
    assert!(sum.read() > 0 && sum.write() > 0 && sum.flops() > 0.0);
}

#[test]
fn an_annotated_layer_joins_the_same_sum() {
    let model = Stack::of(vec![
        Box::new(LmHead),
        Box::new(Linear::new("blk.0.attn_q", 4096, 4096).expect("合法")),
    ])
    .expect("名字唯一");
    let floor = model.floor();
    assert!(floor.unpriced().is_empty(), "两侧都有形状");
    let head = LmHead.bounds().expect("合法").expect("有算子");
    let projection = Linear::new("blk.0.attn_q", 4096, 4096)
        .expect("合法")
        .bounds()
        .expect("合法")
        .expect("有算子");
    assert_eq!(floor.total(), head.plus(projection), "注解层没有第二条公式");
}

#[test]
fn a_layer_without_a_legal_kernel_only_takes_itself_out() {
    // in = 8 不是量化分组的整数倍：这一层没有合法内核，但其余层照常相加。
    let model = Stack::of(vec![
        Box::new(Linear::new("ok", 64, 4).expect("合法")),
        Box::new(Linear::new("odd", 8, 4).expect("层可构造")),
    ])
    .expect("名字唯一");
    let floor = model.floor();
    assert_eq!(floor.unpriced().len(), 1);
    assert_eq!(floor.unpriced()[0].name(), "odd");
    assert!(
        floor.unpriced()[0].reason().contains("整数倍"),
        "{}",
        floor.unpriced()[0]
    );
    assert_eq!(
        floor.total(),
        Bounds::of_op("gemv_i4", 64, 4).expect("合法"),
        "一层的失败没有清零整张模型"
    );
}

#[test]
fn duplicate_names_after_stripping_the_suffix_are_rejected() {
    let error = Stack::from_header(&header(vec![
        entry("a.weight", [64, 32], 0),
        entry("a", [32, 64], 64 * 32 * 4),
    ]))
    .expect_err("去掉后缀后重名");
    assert!(error.to_string().contains("层名 `a` 重复"), "{error}");
}

#[test]
fn a_zero_width_projection_is_rejected() {
    let error =
        Stack::from_header(&header(vec![entry("empty.weight", [0, 32], 0)])).expect_err("宽度为零");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("empty"), "{error}");
}

/// 真实量化 checkout 的形状：4 位打包权重 + 每 `group` 个元素一个 f16 标度。
fn quantized(name: &str, out: usize, in_features: usize, groups: usize) -> Vec<Entry> {
    vec![
        packed(&format!("{name}.weight"), [out, in_features], 0),
        Entry::new(
            format!("{name}.scales"),
            DType::F16,
            Shape::new([out, groups]),
        )
        .at(1 << 20),
    ]
}

#[test]
fn a_packed_weight_and_its_scales_become_one_priced_layer() {
    // 128 个输入 / 4 个分组 = 每 32 个元素一个标度，正是内核声明的那个分组。
    let model =
        Stack::from_header(&header(quantized("blk.0.attn_q", 64, 128, 4))).expect("文件自洽");
    assert_eq!(model.layers().len(), 1, "标度不是层");
    assert!(model.uncovered().is_empty(), "标度也不该进未覆盖清单");

    let floor = model.floor();
    assert!(floor.unpriced().is_empty(), "{:?}", floor.unpriced());
    // 分组来自标度的形状，不是默认值：下界必须与算子语义逐个数字一致。
    assert_eq!(
        floor.total(),
        Bounds::of_op(Linear::OP, 128, 64).expect("同一份语义")
    );
    assert_eq!(model.layers()[0].spca().ops(), [Linear::OP]);
}

#[test]
fn a_group_the_kernel_does_not_declare_is_named_in_the_reason() {
    // 512 / 4 = 每 128 个元素一个标度：真实权重里很常见，但不是本内核声明的那一档。
    let model =
        Stack::from_header(&header(quantized("blk.0.attn_q", 64, 512, 4))).expect("文件自洽");
    let floor = model.floor();
    assert_eq!(floor.total(), Bounds::ZERO, "分组不同就不给数字");
    assert_eq!(floor.unpriced().len(), 1);
    let reason = floor.unpriced()[0].reason();
    assert!(reason.contains("128"), "要说出文件里的分组：{reason}");
    assert!(reason.contains("32"), "也要说出内核声明的分组：{reason}");
}

#[test]
fn a_packed_weight_without_scales_says_what_is_missing() {
    let model =
        Stack::from_header(&header(vec![packed("w.weight", [8, 32], 0)])).expect("文件合法");
    let floor = model.floor();
    assert_eq!(floor.unpriced().len(), 1);
    assert!(
        floor.unpriced()[0].reason().contains("标度条目"),
        "{}",
        floor.unpriced()[0]
    );
}

#[test]
fn a_scale_that_contradicts_its_weight_stops_the_whole_stack() {
    // 输出维对不上：这不是"另一种量化方案"，是文件自相矛盾，算下去一定是错数。
    let entries = vec![
        packed("blk.0.attn_q.weight", [64, 128], 0),
        Entry::new("blk.0.attn_q.scales", DType::F16, Shape::new([32, 4])).at(1 << 20),
    ];
    let error = Stack::from_header(&header(entries)).expect_err("输出维不等");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("输出维 32"), "{error}");
}

#[test]
fn a_declared_model_rejects_duplicate_names_too() {
    let one = Linear::new("blk.0", 32, 32).expect("合法");
    let two = Linear::new("blk.0", 32, 32).expect("合法");
    let error = Stack::new(vec![one, two]).expect_err("重名");
    assert!(error.to_string().contains("blk.0"), "{error}");
}

#[test]
fn a_quantized_entry_and_its_unpacked_view_declare_the_same_floor() {
    // 同一层权重的两种书写：解包视图（U4，第二维是元素数）与真实文件（U8 打包，第二维是字节数）。
    let packed_bytes = Entry::new("blk.0.q.weight", DType::U8, Shape::new([64, 64])).at(0);
    let scales = Entry::new("blk.0.q.scales", DType::F16, Shape::new([64, 4])).at(4096);
    let plain = Entry::new("blk.0.q.weight", DType::U4, Shape::new([64, 128])).at(0);
    let plain_scales = Entry::new("blk.0.q.scales", DType::F16, Shape::new([64, 4])).at(4096);

    let from_file = Stack::from_header(&header(vec![packed_bytes, scales])).expect("形状合法");
    let from_view = Stack::from_header(&header(vec![plain, plain_scales])).expect("形状合法");

    assert_eq!(from_file.layers().len(), 1, "标度不是层");
    assert_eq!(from_view.layers().len(), 1);
    assert!(
        from_file.uncovered().is_empty(),
        "标度被当成未覆盖条目：{:?}",
        from_file.uncovered()
    );
    let a = from_file.floor().total();
    let b = from_view.floor().total();
    assert!(a.bytes() > 0, "量化层应当可计价");
    assert_eq!(a, b, "8 位条目的第二维是字节数，宽度必须由标度推出");
}

#[test]
fn a_scale_with_no_matching_group_is_named_not_silently_priced() {
    // 打包 nibble：group = 128 个元素一组，而内核声明 32：不是同一个东西，必须点名。
    let weight = Entry::new("blk.0.q.weight", DType::U8, Shape::new([64, 64])).at(0);
    let scales = Entry::new("blk.0.q.scales", DType::F16, Shape::new([64, 1])).at(4096);
    let model = Stack::from_header(&header(vec![weight, scales])).expect("形状合法");

    let floor = model.floor();
    assert_eq!(floor.total(), Bounds::ZERO);
    assert_eq!(floor.unpriced().len(), 1);
    assert!(
        floor.unpriced()[0].reason().contains("128"),
        "理由要点名真实分组：{}",
        floor.unpriced()[0]
    );
    assert!(
        floor.unpriced()[0].reason().contains("32"),
        "理由要点名内核要求的分组：{}",
        floor.unpriced()[0]
    );
}

/// 一份真实 LSTM 检查点：2 层、双向、隐宽 5、输入 16。
///
/// 第二层的输入宽是 `隐宽·2 = 10` 而不是 16，故意与首层不同：若识别实现把「第一层吃输入宽」
/// 抄成「每层都吃输入宽」，这条用例就会在重新生成那一步对不上。
fn lstm_entries() -> Vec<Entry> {
    let (input, hidden, gates, layers, dirs) = (16usize, 5usize, 4usize, 2usize, 2usize);
    let mut entries = Vec::new();
    let mut at = 0usize;
    for level in 0..layers {
        let width = if level == 0 { input } else { hidden * dirs };
        for dir in 0..dirs {
            let rev = if dir == 0 { "" } else { "_reverse" };
            let rows = gates * hidden;
            for (suffix, shape) in [
                ("weight_ih", Shape::new([rows, width])),
                ("weight_hh", Shape::new([rows, hidden])),
                ("bias_ih", Shape::new([rows])),
                ("bias_hh", Shape::new([rows])),
            ] {
                let name = format!("blk.lstm.{suffix}_l{level}{rev}");
                let offset = at;
                at += shape.element_bytes(DType::F32);
                entries.push(Entry::new(name, DType::F32, shape).at(offset));
            }
        }
    }
    entries
}

/// 四件权重被认成**一次**循环层，而不是八次投影。
#[test]
fn a_recurrent_family_is_recognised_from_its_own_shapes() {
    let model = Stack::from_header(&header(lstm_entries())).expect("形状合法");
    assert_eq!(model.layers().len(), 1, "四件 × 层 × 方向只算一层");
    let layer = model.layers()[0].recurrent().expect("认出的是循环层");
    assert_eq!(layer.cell(), Cell::Lstm, "门数 4 反推出四门单元");
    assert_eq!(
        (layer.input(), layer.hidden(), layer.layers(), layer.dirs()),
        (16, 5, 2, 2),
        "输入宽看 `weight_ih`、隐宽看 `weight_hh` 的第二维、规模看层号最大值"
    );
    assert!(layer.bias(), "有 `bias_ih` 就是带偏置的单元");
    assert!(model.uncovered().is_empty(), "四件全部归入这一层");
    assert!(
        model.floor().total().flops() > 0.0,
        "认出来就要能计价，否则只是换了个地方放名字"
    );
}

/// 缺一件就一件都不认：秩为 2 的 `weight_ih_l0` 不会被悄悄当成一个投影。
#[test]
fn a_partial_recurrent_family_is_reported_not_priced_as_a_projection() {
    let model = Stack::from_header(&header(vec![
        Entry::new("blk.lstm.weight_ih_l0", DType::F32, Shape::new([20, 16])).at(0),
    ]))
    .expect("形状合法");
    assert!(model.layers().is_empty(), "没有 `weight_hh` 就定不出门数与隐宽");
    assert_eq!(
        model.uncovered(),
        ["blk.lstm.weight_ih_l0"],
        "认不出就如实报出：把它当投影会让层名与真实张量对不上"
    );
}
