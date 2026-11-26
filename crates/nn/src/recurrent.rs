//! 循环层：RNN / GRU / LSTM 三种单元共用**一份**声明。
//!
//! 三者的差别只有一张门表（1 / 3 / 4 门）与激活算子（`tanh` 或 `σ` + `tanh`），其余全部相同：
//! 每一层、每一个方向都是一次 `(输入宽 + 隐宽) → 门数·隐宽` 的 GEMV，加逐元素激活与门乘。
//! 因此这里是一个类型参数化在 [`Cell`] 上，而不是三个类型加一个 `BiLstm`——双向只是
//! `dirs = 2`，不需要第二个类型、也不需要两套权重命名。
//!
//! 权重按真实 checkpoint 的形状声明，而不是融合成一块：`weight_ih` / `weight_hh` /
//! `bias_ih` / `bias_hh` 四件、名字与 PyTorch 的 `lstm.weight_ih_l0` 逐字相同（反向加
//! `_reverse`），每件都带形状。这样 [`Recurrent::weight_entries`] 给出的清单能直接对上一份
//! 真实权重，而 [`Recurrent::param_bytes`] 由同一份清单求和推出——「参数总量」与「加载清单」
//! 因此不可能各说一套。
//!
//! 声明是**逐时间步**的：输出形状给 `[hidden·dirs]`，时间步由调用方展开；下界因此也是
//! 单步的下界，乘上步数即整段循环的代价。这样层不必凭空编一个序列长度来让它听起来更贵。
//!
//! 序列布局（batch 与 seq 谁在前）与初态/末态形状是运行时几何：[`Recurrent::batch_first`]
//! 记布局，[`Recurrent::sequence_output_dims`] / [`Recurrent::state_dims`] 按 batch、seq
//! 推出全序列输出与隐态（LSTM 另有同形单元态）的维度；这些几何不进权重清单，也不参与计价。

use diagnostic::{Code, Error, Result};
use tensor::Shape;

use crate::{Bounds, BoundsResult, Contract, ContractSpec, LayerGeometry, spec_accessors};

/// 循环单元的门结构。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cell {
    /// 单门：`h' = tanh(W·[x, h] + b)`。
    Rnn,
    /// 三门：更新门 / 重置门 / 候选态。
    Gru,
    /// 四门：遗忘 / 输入 / 输出 / 候选态。
    Lstm,
}

impl Cell {
    /// 三种单元。遍历用它，而不是在测试里各写一遍。
    pub const ALL: [Self; 3] = [Self::Rnn, Self::Gru, Self::Lstm];

    /// 门数：RNN 1、GRU 3、LSTM 4。
    pub const fn gates(self) -> usize {
        match self {
            Self::Rnn => 1,
            Self::Gru => 3,
            Self::Lstm => 4,
        }
    }

    /// 由门数反推单元类型：[`Cell::gates`] 的逆。
    ///
    /// 检查点里没有「这是 GRU 还是 LSTM」的标记，只有 `weight_hh` 的行列比。用同一张门表反推，
    /// 而不是再写一份 `4 => Lstm` 的映射：表改了两侧一起改。
    pub fn of_gates(gates: usize) -> Option<Self> {
        Self::ALL.into_iter().find(|cell| cell.gates() == gates)
    }

    /// 单元名：诊断文案与权重条目都用它，不再有第二份拼写。
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rnn => "rnn",
            Self::Gru => "gru",
            Self::Lstm => "lstm",
        }
    }

    /// 本单元每次前进要做的激活，`(算子名, 次数)`：门控单元一律 `σ` 做门、`tanh` 做候选态。
    ///
    /// 次数不是装饰：LSTM 三个门各一次 `σ`、候选态与单元态各一次 `tanh`，GRU 两个门各一次
    /// `σ`、候选态一次 `tanh`——下界要按真实次数收，否则四种单元报同一个数就失去了区分度。
    pub const fn activations(self) -> &'static [(&'static str, usize)] {
        match self {
            Self::Rnn => &[("tanh_f32", 1)],
            Self::Gru => &[("sigmoid_f32", 2), ("tanh_f32", 1)],
            Self::Lstm => &[("sigmoid_f32", 3), ("tanh_f32", 2)],
        }
    }
}

/// 循环层方向内的一件权重：名字与形状都对齐真实 checkpoint。
///
/// 融合成一块（比如 `weight_ihh`）会让真实权重对不上——输入侧与隐侧是两张矩阵，第二维不同
/// （输入侧吃 `输入宽`，隐侧永远吃 `隐宽`），偏置也是两条而不是一条。因此按四件声明。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecurrentTensor {
    /// 输入侧门矩阵 `[门数·隐宽, 输入宽]`。
    WeightIh,
    /// 隐侧门矩阵 `[门数·隐宽, 隐宽]`。
    WeightHh,
    /// 输入侧门偏置 `[门数·隐宽]`。
    BiasIh,
    /// 隐侧门偏置 `[门数·隐宽]`。
    BiasHh,
}

impl RecurrentTensor {
    /// 四件权重；`bias = false` 时只取前两件。
    pub const ALL: [Self; 4] = [Self::WeightIh, Self::WeightHh, Self::BiasIh, Self::BiasHh];

    /// 后缀 → 件：闭集只有四件，反查走这一处，拆名与装配共用同一张表。
    pub fn from_suffix(suffix: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|item| item.suffix() == suffix)
    }

    /// 名字后缀（不含层号与方向）：与 PyTorch 的 `weight_ih` / `bias_hh` 同名。
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::WeightIh => "weight_ih",
            Self::WeightHh => "weight_hh",
            Self::BiasIh => "bias_ih",
            Self::BiasHh => "bias_hh",
        }
    }

    /// 是不是偏置：偏置是一维的，且受 `bias` 开关控制。
    pub const fn is_bias(self) -> bool {
        matches!(self, Self::BiasIh | Self::BiasHh)
    }

    /// 形状：门堆叠在第一维。`width` 只对输入侧矩阵有意义。
    fn shape(self, gates: usize, hidden: usize, width: usize) -> Shape {
        let rows = gates * hidden;
        match self {
            Self::WeightIh => Shape::new([rows, width]),
            Self::WeightHh => Shape::new([rows, hidden]),
            Self::BiasIh | Self::BiasHh => Shape::new([rows]),
        }
    }
}

/// 循环层：`layers` 层、每层 `dirs` 个方向，每层每方向一次门投影。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Recurrent {
    spec: ContractSpec,
    input: usize,
    hidden: usize,
    layers: usize,
    dirs: usize,
    cell: Cell,
    bias: bool,
    batch_first: bool,
}

impl Recurrent {
    /// 门投影的算子名：一层一方向就是一次 `(输入宽 + 隐宽) → 门数·隐宽` 的 GEMV。
    pub const GEMV: &str = "gemv_f32";

    /// 由层名、单元、输入宽、隐宽、层数、方向数与偏置开关构造；序列布局取 **seq-first**
    /// （`batch_first = false`，PyTorch 口径）。需要 batch-first 布局时走 [`Self::with_layout`]。
    ///
    /// 单向输出 `[hidden]`、双向输出 `[hidden·2]`：双向把两个方向的结果拼接，故输出宽是
    /// `hidden · dirs`；第一层吃 `input`，其后的层吃上一层的拼接结果。
    ///
    /// # Errors
    ///
    /// 任一尺寸为零、或 `dirs` 不在 `{1, 2}`（[`Code::Invalid`]）；算子未在 `registry` 注册或
    /// 没有语义表（[`ContractSpec`] 的上报）。
    pub fn new(
        name: impl Into<String>,
        cell: Cell,
        input: usize,
        hidden: usize,
        layers: usize,
        dirs: usize,
        bias: bool,
    ) -> Result<Self> {
        Self::with_layout(name, cell, input, hidden, layers, dirs, bias, false)
    }

    /// 同 [`Self::new`]，但显式声明输入/输出的序列布局。
    ///
    /// `batch_first = true`：输入与全序列输出按 `[batch, seq, hidden·dirs]`；`false`：按
    /// `[seq, batch, hidden·dirs]`。默认值取 `false`（PyTorch 口径）；Burn 的 `batch_first`
    /// 默认 `true`，两侧默认值不同，跨框架对拍时必须把布局显式对齐。
    ///
    /// 布局是运行时几何，不进权重清单：从检查点反推时一律取 seq-first，不会因此认不出权重。
    #[allow(clippy::too_many_arguments)]
    pub fn with_layout(
        name: impl Into<String>,
        cell: Cell,
        input: usize,
        hidden: usize,
        layers: usize,
        dirs: usize,
        bias: bool,
        batch_first: bool,
    ) -> Result<Self> {
        crate::nonzero_all(
            "nn::Recurrent::with_layout",
            "循环层 ",
            &[
                ("input", input),
                ("hidden", hidden),
                ("layers", layers),
                ("dirs", dirs),
            ],
        )?;
        Self::two_directions(dirs)?;
        let spec = ContractSpec::new(name, Shape::new([hidden * dirs]), Self::cell_ops(cell))?;
        Ok(Self {
            spec,
            input,
            hidden,
            layers,
            dirs,
            cell,
            bias,
            batch_first,
        })
    }

    /// 循环层只有单向与双向：`dirs` 不在 `{1, 2}` 里就在构造期失败。
    fn two_directions(dirs: usize) -> Result<()> {
        if dirs > 2 {
            return Err(Error::new(
                Code::Invalid,
                format!("循环层只支持单向或双向，dirs = {dirs} 既不是 1 也不是 2"),
            )
            .with("nn::Recurrent::new"));
        }
        Ok(())
    }

    /// 本层要装配的算子：`gemv` 加上 cell 的激活。
    /// 去重靠 `contains` 而不是 `dedup`：后者只吃掉相邻重复，表的顺序一变就会漏。
    fn cell_ops(cell: Cell) -> Vec<&'static str> {
        let mut ops = vec![Self::GEMV];
        for (op, _) in cell.activations() {
            if !ops.contains(op) {
                ops.push(op);
            }
        }
        ops
    }

    crate::accessors! {
        /// 输入宽度。
        input -> usize = input,
        /// 隐宽（每个方向的）。
        hidden -> usize = hidden,
        /// 层数。
        layers -> usize = layers,
        /// 方向数：1 或 2。
        dirs -> usize = dirs,
        /// 单元类型。
        cell -> Cell = cell,
        /// 是否带门偏置。
        bias -> bool = bias,
        /// 输入/输出是否 batch-first：`true` 为 `[batch, seq, …]`，`false` 为 `[seq, batch, …]`。
        batch_first -> bool = batch_first,
    }

    /// 初态/末态隐状态的形状：`[layers·dirs, batch, hidden]`。
    ///
    /// 层数与方向沿第一维堆叠，与 PyTorch `h_n` 的 `[layers·dirs, batch, hidden]` 口径一致；
    /// 单向退化为 `[layers, batch, hidden]`。LSTM 的单元态 `c` 与隐态同形（见 [`Self::has_cell_state`]）。
    pub fn state_dims(&self, batch: usize) -> Vec<usize> {
        vec![self.layers * self.dirs, batch, self.hidden]
    }

    /// 全序列输出形状：按 [`Self::batch_first`] 交换 batch 与 seq 两轴，末轴是 `hidden·dirs`。
    pub fn sequence_output_dims(&self, batch: usize, seq: usize) -> Vec<usize> {
        let out = self.hidden * self.dirs;
        if self.batch_first {
            vec![batch, seq, out]
        } else {
            vec![seq, batch, out]
        }
    }

    /// 本单元是否带单元态：LSTM 另有一条与隐态同形的 `c`，RNN/GRU 只有隐态 `h`。
    pub fn has_cell_state(&self) -> bool {
        self.cell == Cell::Lstm
    }

    /// 唯一的一份「每层每方向：`(层, 方向, 输入宽, 门堆叠行数)`」。
    ///
    /// 投影、张量清单、参数量三者都从它展开，所以「第一层吃 `input`、其后吃拼接结果」这条口径
    /// 只有一处拼写；想改宽度规则不可能只改到三个调用方里的一处。
    fn tiling(&self) -> impl Iterator<Item = (usize, usize, usize, usize)> + '_ {
        let out = self.cell.gates() * self.hidden;
        (0..self.layers).flat_map(move |level| {
            let width = if level == 0 {
                self.input
            } else {
                self.hidden * self.dirs
            };
            (0..self.dirs).map(move |dir| (level, dir, width, out))
        })
    }

    /// 每层每方向的一次门投影 `(输入宽度, 输出宽度)`：`bounds` 按它逐次计价。
    pub fn projections(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.tiling().map(|(_, _, width, out)| (width, out))
    }

    /// 张量清单：每层每方向两张门矩阵、两条门偏置（`bias = false` 时只有矩阵）。
    fn tensors(&self) -> impl Iterator<Item = (usize, usize, RecurrentTensor, Shape)> + '_ {
        let (gates, hidden, bias) = (self.cell.gates(), self.hidden, self.bias);
        self.tiling().flat_map(move |(level, dir, width, _)| {
            RecurrentTensor::ALL
                .into_iter()
                .filter(move |t| bias || !t.is_bias())
                .map(move |t| (level, dir, t, t.shape(gates, hidden, width)))
        })
    }

    /// 一件权重的名字：`{层名}.{后缀}_l{层}{_reverse}`——与 PyTorch 的 `lstm.weight_ih_l0` /
    /// `lstm.weight_hh_l0_reverse` 逐字相同。
    ///
    /// 前缀就是本层自己的名字，不再另插一个单元段：真实 checkpoint 里的模块名是调用方定的
    /// （`self.rnn = nn.GRU(..)` 存出来是 `rnn.weight_ih_l0`），多插一段 `gru.` 反而对不上。
    pub fn weight_entry(&self, level: usize, dir: usize, tensor: RecurrentTensor) -> String {
        let rev = if dir == 0 { "" } else { "_reverse" };
        format!("{}.{}_l{level}{rev}", self.spec.name(), tensor.suffix())
    }

    /// 全部权重条目与形状，按 `(层, 方向, 件)` 升序——装配侧按它逐条取张量并核对形状。
    ///
    /// 这份清单是双向的：[`crate::Stack::from_header`] 从检查点认出循环层时，按它把名字与形状
    /// 逐条重新生成一遍，全对上才认。
    pub fn weight_entries(&self) -> Vec<(String, Shape)> {
        self.tensors()
            .map(|(level, dir, tensor, shape)| (self.weight_entry(level, dir, tensor), shape))
            .collect()
    }

    /// 参数字节数：由声明的张量形状求和，按 4 字节/元素。
    ///
    /// 与 [`Recurrent::weight_entries`] 同源，因此它不会漏算 `bias_hh`（融合写法最容易漏的就是它），
    /// 也不会与加载清单对不上。
    pub fn param_bytes(&self) -> u64 {
        self.tensors()
            .map(|(_, _, _, shape)| shape.numel() as u64)
            .sum::<u64>()
            * 4
    }
}

impl Contract for Recurrent {
    spec_accessors!();

    fn recurrent(&self) -> Option<&Recurrent> {
        Some(self)
    }

    fn bounds(&self) -> BoundsResult {
        let projections = self
            .projections()
            .try_fold(Bounds::ZERO, |total, (inp, out)| {
                Bounds::of_geometry(Self::GEMV, LayerGeometry::projection(inp, out))
                    .map(|cost| total.plus(cost))
            })?;
        Ok(Some(projections.plus(self.activation_bound()?)))
    }
}

impl Recurrent {
    /// 门激活的总代价：一元层按 `(len, 1)` 计价，与门投影同一口径，不另编一套
    /// 「隐宽当投影宽」的说法；出现次数是 `count × 方向数 × 层数`。
    fn activation_bound(&self) -> Result<Bounds> {
        self.cell
            .activations()
            .iter()
            .try_fold(Bounds::ZERO, |total, (op, count)| {
                let step = Bounds::of_geometry(op, LayerGeometry::axis(self.hidden))?;
                Ok((0..count * self.dirs * self.layers).fold(total, |sum, _| sum.plus(step)))
            })
    }
}

#[cfg(test)]
mod recurrent_tests {
    //! `src/recurrent.rs` 的全部用例：三种单元共用一份声明，所以每条断言都按单元表遍历，
    //! 而不是给 RNN / GRU / LSTM 各写一遍。

    use super::*;

    use crate::Contract;
    use diagnostic::Code;

    testkit::test_cases! {
    /// 门数就是参数宽度的唯一来源：两张门矩阵的行数都是 `门数·隐宽`。
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
    fn batch_first_defaults_to_seq_first_and_does_not_touch_weights() {
        let def = Recurrent::new("r", Cell::Lstm, 16, 8, 2, 2, true).expect("合法");
        assert!(!def.batch_first(), "默认 seq-first");

        let bf = Recurrent::with_layout("r", Cell::Lstm, 16, 8, 2, 2, true, true).expect("合法");
        assert!(bf.batch_first(), "with_layout 显式 batch-first");

        assert_eq!(
            bf.weight_entries(),
            def.weight_entries(),
            "布局不进权重清单"
        );
        assert_eq!(bf.param_bytes(), def.param_bytes(), "布局不影响参数量");
        assert_eq!(
            bf.bounds().expect("合法"),
            def.bounds().expect("合法"),
            "布局不影响下界"
        );
    }

    /// 末态隐态形状按 `[layers·dirs, batch, hidden]` 堆叠；LSTM 另带同形单元态，RNN/GRU 没有。
    fn state_dims_stack_layers_and_directions() {
        let uni = Recurrent::new("r", Cell::Rnn, 16, 8, 3, 1, true).expect("合法");
        assert_eq!(
            uni.state_dims(5),
            [3, 5, 8],
            "单向三层：[layers, batch, hidden]"
        );
        assert!(!uni.has_cell_state(), "RNN 无单元态");

        let bi = Recurrent::new("r", Cell::Gru, 16, 8, 2, 2, true).expect("合法");
        assert_eq!(
            bi.state_dims(5),
            [4, 5, 8],
            "双向两层：[layers·dirs, batch, hidden]"
        );
        assert!(!bi.has_cell_state(), "GRU 无单元态");

        let lstm = Recurrent::new("r", Cell::Lstm, 16, 8, 1, 1, true).expect("合法");
        assert!(lstm.has_cell_state(), "LSTM 带单元态，与隐态同形");
    }

    /// 全序列输出按布局交换 batch 与 seq 两轴，末轴恒为 `hidden·dirs`。
    fn sequence_output_dims_follow_the_layout() {
        let sf = Recurrent::new("r", Cell::Lstm, 16, 8, 1, 2, true).expect("合法");
        assert_eq!(
            sf.sequence_output_dims(3, 7),
            [7, 3, 16],
            "seq-first：[seq, batch, hidden·2]"
        );

        let bf = Recurrent::with_layout("r", Cell::Lstm, 16, 8, 1, 2, true, true).expect("合法");
        assert_eq!(
            bf.sequence_output_dims(3, 7),
            [3, 7, 16],
            "batch-first：[batch, seq, hidden·2]"
        );
    }
    }
}
