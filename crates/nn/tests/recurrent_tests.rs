//! `src/recurrent.rs` 的全部用例：三种单元共用一份声明，所以每条断言都按单元表遍历，
//! 而不是给 RNN / GRU / LSTM 各写一遍。

use super::*;

use crate::Contract;
use diagnostic::Code;

/// 门数就是参数宽度的唯一来源：门矩阵是 `门数·隐宽` 行。
#[test]
fn the_gate_table_drives_parameter_bytes() {
    let bytes = |cell| {
        Recurrent::new("r", cell, 8, 4, 1, 1, true)
            .expect("合法")
            .param_bytes()
    };
    let rnn = bytes(Cell::Rnn);
    assert_eq!(rnn, (4 * 8 + 4) * 4, "RNN：一门 (8 → 4) 加偏置");
    assert_eq!(bytes(Cell::Gru), rnn * 3, "GRU 三门恰好三倍");
    assert_eq!(bytes(Cell::Lstm), rnn * 4, "LSTM 四门恰好四倍");
}

/// 偏置是 `门数·隐宽` 个元素，与输入宽无关；关掉它就少这一块。
#[test]
fn the_bias_costs_one_element_per_output() {
    let with = Recurrent::new("r", Cell::Lstm, 8, 4, 1, 1, true)
        .expect("合法")
        .param_bytes();
    let without = Recurrent::new("r", Cell::Lstm, 8, 4, 1, 1, false)
        .expect("合法")
        .param_bytes();
    assert_eq!(with - without, (4 * 4) * 4, "四门 × 隐宽 4 × 4 字节");
}

/// 每层每方向一个权重整块：条目数正好 `层数 × 方向数`，且互不重复。
#[test]
fn every_layer_and_direction_owns_one_weight_entry() {
    let layer = Recurrent::new("r", Cell::Gru, 16, 8, 3, 2, true).expect("合法");
    let entries = layer.weight_entries("rnn");
    assert_eq!(entries.len(), 3 * 2, "三层双向六块");
    let unique: std::collections::BTreeSet<&String> = entries.iter().collect();
    assert_eq!(unique.len(), entries.len(), "条目名必须互不重复");
    assert!(entries.iter().all(|entry| entry.starts_with("rnn.gru")), "条目带单元名");
    assert!(entries.iter().all(|entry| entry.ends_with(WEIGHT_SUFFIX)));
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
    assert_eq!(widths, [16, 16, 16, 16], "第二层吃 `隐宽·2`，与输入宽 16 恰好同值");
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
    assert_eq!(floor(&bi).flops(), floor(&uni).flops() * 2.0, "算术量精确翻倍");
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
