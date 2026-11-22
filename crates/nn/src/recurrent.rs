//! 循环层：RNN / GRU / LSTM 三种单元共用**一份**声明。
//!
//! 三者的差别只有一张门表（1 / 3 / 4 门）与激活算子（`tanh` 或 `σ` + `tanh`），其余全部相同：
//! 每一层、每一个方向都是一次 `(输入宽 + 隐宽) → 门数·隐宽` 的 GEMV，加逐元素激活与门乘。
//! 因此这里是一个类型参数化在 [`Cell`] 上，而不是三个类型加一个 `BiLstm`——双向只是
//! `dirs = 2`，不需要第二个类型、也不需要两套权重命名。
//!
//! 声明是**逐时间步**的：输出形状给 `[hidden·dirs]`，时间步由调用方展开；下界因此也是
//! 单步的下界，乘上步数即整段循环的代价。这样层不必凭空编一个序列长度来让它听起来更贵。

use diagnostic::{Code, Error, Result};
use tensor::Shape;

use crate::{Bounds, BoundsResult, Contract, ContractSpec, WEIGHT_SUFFIX, spec_accessors};

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
            &[("input", input), ("hidden", hidden), ("layers", layers), ("dirs", dirs)],
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

    /// 每层每方向的 `(输入宽度, 输出宽度)`：第一层吃 `input`，其余层吃上一层的拼接结果。
    pub fn projections(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let out = self.cell.gates() * self.hidden;
        (0..self.layers).flat_map(move |level| {
            let width = if level == 0 {
                self.input
            } else {
                self.hidden * self.dirs
            };
            (0..self.dirs).map(move |_| (width, out))
        })
    }

    /// 权重条目名：`{块}.{单元}{层}.{方向}{WEIGHT_SUFFIX}`。门矩阵按层按方向各驻留一整块。
    pub fn weight_entry(&self, block: &str, level: usize, dir: usize) -> String {
        format!(
            "{block}.{}{level}.{dir}{WEIGHT_SUFFIX}",
            self.cell.name()
        )
    }

    /// 全部权重条目名，按 `(层, 方向)` 升序——装配侧按它逐条取权重整块。
    pub fn weight_entries(&self, block: &str) -> Vec<String> {
        (0..self.layers)
            .flat_map(|level| (0..self.dirs).map(move |dir| (level, dir)))
            .map(|(level, dir)| self.weight_entry(block, level, dir))
            .collect()
    }

    /// 参数字节数：门矩阵 `out × in` 加可选的门偏置 `out`，按 4 字节/元素。
    pub fn param_bytes(&self) -> u64 {
        let bias = u64::from(self.bias) * self.cell.gates() as u64 * self.hidden as u64;
        self.projections()
            .map(|(inp, out)| out as u64 * inp as u64 + bias)
            .sum::<u64>()
            * 4
    }
}

impl Contract for Recurrent {
    spec_accessors!();

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
