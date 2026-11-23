//! `src/recurrent.rs` 的全部用例：三种单元共用一份声明，所以每条断言都按单元表遍历，
//! 而不是给 RNN / GRU / LSTM 各写一遍。

use super::*;

use crate::Contract;
use diagnostic::Code;

/// 门数就是参数宽度的唯一来源：两张门矩阵的行数都是 `门数·隐宽`。
#[test]
fn the_gate_table_drives_parameter_bytes() {
    let bytes = |cell| {
        Recurrent::new("r", cell, 8, 4, 1, 1, true)
            .expect("合法")
            .param_bytes()
    };
    let rnn = bytes(Cell::Rnn);
    // 一门：输入侧 4×8、隐侧 4×4、两条偏置各 4 个元素。
    assert_eq!(rnn, (4 * 8 + 4 * 4 + 4 + 4) * 4, "RNN 四件张量");
    assert_eq!(bytes(Cell::Gru), rnn * 3, "GRU 三门恰好三倍");
    assert_eq!(bytes(Cell::Lstm), rnn * 4, "LSTM 四门恰好四倍");
}

/// 偏置是**两条**向量（输入侧与隐侧各一条），与输入宽无关；关掉它就少这两条。
///
/// 只算一条是本层最容易被写漏的地方：PyTorch / Burn 的 checkpoint 里 `bias_ih` 与 `bias_hh`
/// 是两个张量，融合写法会把参数总量少报一半偏置。
#[test]
fn the_bias_costs_two_vectors_per_direction() {
    let bytes = |bias| {
        Recurrent::new("r", Cell::Lstm, 8, 4, 1, 1, bias)
            .expect("合法")
            .param_bytes()
    };
    assert_eq!(
        bytes(true) - bytes(false),
        2 * (4 * 4) * 4,
        "四门 × 隐宽 4 × 两条偏置 × 4 字节"
    );
}

/// 每层每方向四件张量，名字与真实 checkpoint 逐字相同（反向加 `_reverse`）。
#[test]
fn every_direction_owns_the_four_checkpoint_tensors() {
    let layer = Recurrent::new("blk.gru", Cell::Gru, 16, 8, 2, 2, true).expect("合法");
    let entries = layer.weight_entries();
    assert_eq!(entries.len(), 2 * 2 * 4, "两层双向 × 四件");
    let names: Vec<&str> = entries.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        &names[..4],
        [
            "blk.gru.weight_ih_l0",
            "blk.gru.weight_hh_l0",
            "blk.gru.bias_ih_l0",
            "blk.gru.bias_hh_l0",
        ],
        "首层首方向四件"
    );
    assert!(
        names.contains(&"blk.gru.weight_ih_l1_reverse"),
        "反向层的名字带 _reverse，与 PyTorch 一致"
    );
    let unique: std::collections::BTreeSet<&str> = names.iter().copied().collect();
    assert_eq!(unique.len(), names.len(), "条目名必须互不重复");
}

/// 形状按门堆叠：矩阵 `[门数·隐宽, 该侧宽]`、偏置 `[门数·隐宽]`；隐侧第二维永远是隐宽。
#[test]
fn the_entry_shapes_stack_the_gates() {
    let layer = Recurrent::new("blk.lstm", Cell::Lstm, 16, 8, 2, 2, true).expect("合法");
    let shape = |name: &str| {
        layer
            .weight_entries()
            .into_iter()
            .find(|(n, _)| n == name)
            .map(|(_, shape)| shape.dims().to_vec())
            .expect("条目存在")
    };
    assert_eq!(shape("blk.lstm.weight_ih_l0"), [32, 16], "首层吃输入宽 16");
    assert_eq!(
        shape("blk.lstm.weight_ih_l1"),
        [32, 16],
        "其后层吃拼接结果 `隐宽·2`"
    );
    assert_eq!(shape("blk.lstm.weight_hh_l1"), [32, 8], "隐侧永远是隐宽");
    assert_eq!(shape("blk.lstm.bias_ih_l1"), [32], "偏置一维");
    assert_eq!(
        shape("blk.lstm.bias_hh_l0_reverse"),
        [32],
        "反向偏置同样一维"
    );
}

/// 参数总量与声明清单互相钉住：谁改了口径，另一侧立刻不动。
#[test]
fn the_parameter_total_equals_the_declared_tensors() {
    for cell in Cell::ALL {
        for bias in [false, true] {
            let layer = Recurrent::new("r", cell, 6, 5, 2, 2, bias).expect("合法");
            let declared: usize = layer
                .weight_entries()
                .iter()
                .map(|(_, shape)| shape.numel())
                .sum();
            assert_eq!(
                layer.param_bytes(),
                declared as u64 * 4,
                "{} 的字节数必须等于其声明张量之和（bias = {bias}）",
                cell.name()
            );
        }
    }
}

/// 第一层吃输入宽，其后的层吃上一层的拼接结果（双向是 `隐宽·2`）。
#[test]
fn the_first_layer_wires_the_input_width_and_the_rest_wire_the_stack() {
    let layer = Recurrent::new("r", Cell::Rnn, 16, 8, 3, 2, true).expect("合法");
    let projections: Vec<(usize, usize)> = layer.projections().collect();
    assert_eq!(
        projections,
        [(16, 8), (16, 8), (16, 8), (16, 8), (16, 8), (16, 8)],
        "单向两层的输入宽都是 16"
    );

    let deep = Recurrent::new("r", Cell::Rnn, 16, 8, 2, 2, true).expect("合法");
    let widths: Vec<usize> = deep.projections().map(|(inp, _)| inp).collect();
    assert_eq!(
        widths,
        [16, 16, 16, 16],
        "第二层吃 `隐宽·2`，与输入宽 16 恰好同值"
    );
}

/// 双向把两个方向拼起来：单层时输出宽、参数与下界都精确翻倍。
///
/// 只取单层是有意的：两层以上时第二层的输入宽本身随方向数变（单向 `隐宽`、双向 `隐宽·2`），
/// 那时比值会大于 2，用「精确两倍」去钉多层就是把一条错的口径固定下来。
#[test]
fn going_bidirectional_doubles_parameters_and_floors() {
    let uni = Recurrent::new("r", Cell::Lstm, 16, 8, 1, 1, true).expect("合法");
    let bi = Recurrent::new("r", Cell::Lstm, 16, 8, 1, 2, true).expect("合法");

    assert_eq!(uni.output().dims(), [8], "单向输出一条隐宽");
    assert_eq!(bi.output().dims(), [16], "双向输出拼成两倍");
    assert_eq!(bi.param_bytes(), uni.param_bytes() * 2, "参数精确翻倍");

    let floor = |layer: &Recurrent| layer.bounds().expect("合法").expect("有算子");
    assert_eq!(floor(&bi).read(), floor(&uni).read() * 2, "读侧精确翻倍");
    assert_eq!(floor(&bi).write(), floor(&uni).write() * 2, "写侧精确翻倍");
    assert_eq!(
        floor(&bi).flops(),
        floor(&uni).flops() * 2.0,
        "算术量精确翻倍"
    );
}

/// 下界随隐宽、层数、输入宽单调增；单元越"重"越贵。这条用例盯的是「下界真的进了门表」。
#[test]
fn floors_grow_with_every_declared_dimension() {
    let floor = |cell, input, hidden, layers, dirs| {
        Recurrent::new("r", cell, input, hidden, layers, dirs, true)
            .expect("合法")
            .bounds()
            .expect("合法")
            .expect("有算子")
    };
    let base = floor(Cell::Rnn, 16, 8, 1, 1);
    assert!(floor(Cell::Rnn, 32, 8, 1, 1).read() > base.read(), "输入宽");
    assert!(floor(Cell::Rnn, 16, 16, 1, 1).read() > base.read(), "隐宽");
    assert!(floor(Cell::Rnn, 16, 8, 2, 1).read() > base.read(), "层数");
    assert!(
        floor(Cell::Gru, 16, 8, 1, 1).read() > base.read(),
        "GRU 三门比 RNN 一门贵"
    );
    assert!(
        floor(Cell::Lstm, 16, 8, 1, 1).read() > floor(Cell::Gru, 16, 8, 1, 1).read(),
        "LSTM 四门比 GRU 三门贵"
    );
    assert!(base.flops() > 0.0, "循环层有算术，不是纯搬运");
}

/// 激活次数也进下界：GRU 与 LSTM 的门数差一门，激活表也跟着差一次 `σ` 与一次 `tanh`。
#[test]
fn the_activation_table_is_the_cell_structure() {
    assert_eq!(Cell::Rnn.gates(), 1);
    assert_eq!(Cell::Gru.gates(), 3);
    assert_eq!(Cell::Lstm.gates(), 4);
    assert_eq!(Cell::Rnn.activations(), [("tanh_f32", 1)]);
    assert_eq!(
        Cell::Gru.activations(),
        [("sigmoid_f32", 2), ("tanh_f32", 1)]
    );
    assert_eq!(
        Cell::Lstm.activations(),
        [("sigmoid_f32", 3), ("tanh_f32", 2)]
    );
}

/// 声明的 `ops` 是去重后的门投影 + 激活：重复的激活名不能出现两次。
#[test]
fn the_operator_list_is_deduplicated() {
    let ops = |cell| {
        Recurrent::new("r", cell, 16, 8, 1, 1, true)
            .expect("合法")
            .spca()
            .ops()
            .to_vec()
    };
    assert_eq!(ops(Cell::Rnn), [Recurrent::GEMV, "tanh_f32"]);
    let lstm = ops(Cell::Lstm);
    assert_eq!(lstm, [Recurrent::GEMV, "sigmoid_f32", "tanh_f32"]);
    let unique: std::collections::BTreeSet<&String> = lstm.iter().collect();
    assert_eq!(unique.len(), lstm.len(), "算子名不得重复");
}

/// 零尺寸与非法方向数在构造期就拒绝，不留能构造出空轴的路径。
#[test]
fn zero_sizes_and_impossible_directions_are_rejected() {
    let of = |input, hidden, layers, dirs| {
        Recurrent::new("r", Cell::Lstm, input, hidden, layers, dirs, true).unwrap_err()
    };
    assert_eq!(of(0, 8, 1, 1).code(), Code::Invalid);
    assert_eq!(of(16, 0, 1, 1).code(), Code::Invalid);
    assert_eq!(of(16, 8, 0, 1).code(), Code::Invalid);
    assert_eq!(of(16, 8, 1, 0).code(), Code::Invalid);
    assert_eq!(of(16, 8, 1, 3).code(), Code::Invalid, "三向不存在");
    assert!(of(16, 8, 1, 3).to_string().contains("dirs = 3"));
}

/// 门数是单元的唯一标识：`of_gates` 是 `gates` 的逆，且三种单元的门数不撞车。
#[test]
fn the_gate_count_identifies_the_cell() {
    for cell in Cell::ALL {
        assert_eq!(Cell::of_gates(cell.gates()), Some(cell));
    }
    let gates: std::collections::BTreeSet<usize> =
        Cell::ALL.iter().map(|cell| cell.gates()).collect();
    assert_eq!(
        gates.len(),
        Cell::ALL.len(),
        "门数重复会让从检查点反推单元变成猜"
    );
    assert_eq!(Cell::of_gates(2), None, "两门不是任何一种单元");
}

/// 布局默认 seq-first（PyTorch 口径）；显式 batch-first 只改布局，不进权重清单、不改参数量与下界。
#[test]
fn batch_first_defaults_to_seq_first_and_does_not_touch_weights() {
    let def = Recurrent::new("r", Cell::Lstm, 16, 8, 2, 2, true).expect("合法");
    assert!(!def.batch_first(), "默认 seq-first");

    let bf =
        Recurrent::with_layout("r", Cell::Lstm, 16, 8, 2, 2, true, true).expect("合法");
    assert!(bf.batch_first(), "with_layout 显式 batch-first");

    assert_eq!(bf.weight_entries(), def.weight_entries(), "布局不进权重清单");
    assert_eq!(bf.param_bytes(), def.param_bytes(), "布局不影响参数量");
    assert_eq!(
        bf.bounds().expect("合法"),
        def.bounds().expect("合法"),
        "布局不影响下界"
    );
}

/// 末态隐态形状按 `[layers·dirs, batch, hidden]` 堆叠；LSTM 另带同形单元态，RNN/GRU 没有。
#[test]
fn state_dims_stack_layers_and_directions() {
    let uni = Recurrent::new("r", Cell::Rnn, 16, 8, 3, 1, true).expect("合法");
    assert_eq!(uni.state_dims(5), [3, 5, 8], "单向三层：[layers, batch, hidden]");
    assert!(!uni.has_cell_state(), "RNN 无单元态");

    let bi = Recurrent::new("r", Cell::Gru, 16, 8, 2, 2, true).expect("合法");
    assert_eq!(bi.state_dims(5), [4, 5, 8], "双向两层：[layers·dirs, batch, hidden]");
    assert!(!bi.has_cell_state(), "GRU 无单元态");

    let lstm = Recurrent::new("r", Cell::Lstm, 16, 8, 1, 1, true).expect("合法");
    assert!(lstm.has_cell_state(), "LSTM 带单元态，与隐态同形");
}

/// 全序列输出按布局交换 batch 与 seq 两轴，末轴恒为 `hidden·dirs`。
#[test]
fn sequence_output_dims_follow_the_layout() {
    let sf = Recurrent::new("r", Cell::Lstm, 16, 8, 1, 2, true).expect("合法");
    assert_eq!(
        sf.sequence_output_dims(3, 7),
        [7, 3, 16],
        "seq-first：[seq, batch, hidden·2]"
    );

    let bf =
        Recurrent::with_layout("r", Cell::Lstm, 16, 8, 1, 2, true, true).expect("合法");
    assert_eq!(
        bf.sequence_output_dims(3, 7),
        [3, 7, 16],
        "batch-first：[batch, seq, hidden·2]"
    );
}
