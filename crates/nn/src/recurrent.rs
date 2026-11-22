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

use diagnostic::{Code, Error, Result};
use tensor::Shape;

use crate::{Bounds, BoundsResult, Contract, ContractSpec, spec_accessors};

/// 循环单元的门结构。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recurrent {
    spec: ContractSpec,
    input: usize,
    hidden: usize,
    layers: usize,
    dirs: usize,
    cell: Cell,
    bias: bool,
}

impl Recurrent {
    /// 门投影的算子名：一层一方向就是一次 `(输入宽 + 隐宽) → 门数·隐宽` 的 GEMV。
    pub const GEMV: &str = "gemv_f32";

    /// 由层名、单元、输入宽、隐宽、层数、方向数与偏置开关构造。
    ///
    /// 单向输出 `[hidden]`、双向输出 `[hidden·2]`：双向把两个方向的结果拼接，故输出宽是
    /// `hidden · dirs`；第一层吃 `input`，其后的层吃上一层的拼接结果。
    ///
    /// # Errors
    ///
    /// 任一尺寸为零、或 `dirs` 不在 `{1, 2}`（[`Code::Invalid`]）；算子未在 `op` 注册或
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
        crate::nonzero_all(
            "nn::Recurrent::new",
            "循环层 ",
            &[
                ("input", input),
                ("hidden", hidden),
                ("layers", layers),
                ("dirs", dirs),
            ],
        )?;
        if dirs > 2 {
            return Err(Error::new(
                Code::Invalid,
                format!("循环层只支持单向或双向，dirs = {dirs} 既不是 1 也不是 2"),
            )
            .with("nn::Recurrent::new"));
        }
        // 去重靠 `contains` 而不是 `dedup`：后者只吃掉相邻重复，表的顺序一变就会漏。
        let mut ops: Vec<&str> = vec![Self::GEMV];
        for (op, _) in cell.activations() {
            if !ops.contains(op) {
                ops.push(op);
            }
        }
        let spec = ContractSpec::new(name, Shape::new([hidden * dirs]), ops)?;
        Ok(Self {
            spec,
            input,
            hidden,
            layers,
            dirs,
            cell,
            bias,
        })
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
        let mut total = Bounds::ZERO;
        for (inp, out) in self.projections() {
            total = total.plus(Bounds::of_op(Self::GEMV, inp, out)?);
        }
        // 门激活与门投影同一口径：一元层按 `(len, 1)` 计价，故这里也用 `of_ctx`，
        // 不另编一套「隐宽当投影宽」的说法。
        for (op, count) in self.cell.activations() {
            let step = Bounds::of_ctx(op, self.hidden, 1)?;
            for _ in 0..(*count * self.dirs * self.layers) {
                total = total.plus(step);
            }
        }
        Ok(Some(total))
    }
}

#[cfg(test)]
#[path = "../tests/recurrent_tests.rs"]
mod recurrent_tests;
