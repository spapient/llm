//! 配置优先的模型装配：外部数据只描述层，具体形状、执行与计价仍由层实现负责。
//!
//! [`ModelConfig`] 是可序列化的完整层栈，[`LayerConfig`] 使用 `type = "..."` 区分层类型。
//! 所有配置都拒绝未知字段；装配直接消费字符串与集合，不为类型擦除额外克隆配置。
//!
//! # TOML 样例
//!
//! ```toml
//! [[layers]]
//! type = "linear"
//! name = "blk.0.q"
//! in_features = 128
//! out_features = 64
//!
//! [layers.format]
//! type = "int4"
//! group = 32
//!
//! [[layers]]
//! type = "axis"
//! name = "blk.0.norm"
//! op = "rmsnorm_f32"
//! len = 128
//! role = "norm"
//! ```

use diagnostic::Result;
use serde::{Deserialize, Serialize};

use crate::recurrent::{Cell, Recurrent};
use crate::{
    Attention, AttentionAlgo, AttentionBlock, Contract, DecodeAttention, Embedding, Gauge,
    LatentAttention, Linear, MatMul, MemoryBlock, OpLayer, ParamRole, Precision, Stack,
    WeightFormat,
};

/// 一份完整的模型层栈配置。
///
/// 字段顺序就是执行与层名下界归因顺序。未知顶层字段会被拒绝，避免拼写错误被静默忽略。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// 按装配顺序排列的层。
    pub layers: Vec<LayerConfig>,
}

impl ModelConfig {
    /// 由层配置列表构造模型配置。
    pub fn new(layers: Vec<LayerConfig>) -> Self {
        Self { layers }
    }

    /// 消费配置并构造层栈，不克隆层名或配置集合。
    ///
    /// # Errors
    ///
    /// 任一层构造失败或层名重复时返回对应诊断，并包含 `layers[N]` 配置路径。
    pub fn build(self) -> Result<Stack> {
        Stack::from_configs(self.layers)
    }
}

/// 用一张配置表生成层枚举、稳定短名、层名访问与构造分派。
///
/// 每个变体只声明一次字段和构造器；新增层时不可能忘记同步 `kind`、`name` 或 `build`。
/// 宏仅在编译期展开，生成的仍是普通穷尽 `match`，没有运行时注册表或动态分派开销。
macro_rules! layer_config_table {
    ($(
        $variant:ident => {
            wire = $wire:literal,
            doc = $doc:literal,
            fields = { $(
                $(#[$field_attr:meta])* $field:ident : $field_ty:ty => $field_doc:literal
            ),* $(,)? },
            build = $constructor:expr $(,)?
        }
    ),* $(,)?) => {
        /// 可由模型描述逐条生成的层配置。
        ///
        /// 构造产物是原有具体类型的 trait object，因此所有 `Contract` 类型化视图保持有效。
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(tag = "type", deny_unknown_fields)]
        pub enum LayerConfig {
            $(
                #[doc = $doc]
                #[serde(rename = $wire)]
                $variant {
                    $(
                        $(#[$field_attr])*
                        #[doc = $field_doc]
                        $field: $field_ty
                    ),*
                }
            ),*
        }

        impl LayerConfig {
            /// 稳定的配置类型短名，与序列化字段 `type` 一致。
            pub const fn kind(&self) -> &'static str {
                match self {
                    $(Self::$variant { .. } => $wire),*
                }
            }

            /// 配置声明的层名；构造前即可用于诊断与重名检查。
            pub fn name(&self) -> &str {
                match self {
                    $(Self::$variant { name, .. } => name),*
                }
            }

            /// 用原有构造器物化这一层；所有结构错误都在返回 trait object 前报出。
            ///
            /// # Errors
            ///
            /// 错误原样来自对应层构造器：未注册算子、零宽形状、非法分组或非法几何均不会被改写。
            pub fn build(self) -> Result<Box<dyn Contract>> {
                match self {
                    $(
                        Self::$variant { $($field),* } => {
                            ($constructor)($($field),*).map(boxed)
                        }
                    ),*
                }
            }
        }
    };
}

layer_config_table! {
    Axis => {
        wire = "axis",
        doc = "任意已注册的单轴算子；`op` 可直接来自配置文件。",
        fields = {
            name: String => "层名。",
            op: String => "注册算子名。",
            len: usize => "轴长。",
            #[serde(default)] role: ParamRole => "参数角色；缺省为普通可训练参数。"
        },
        build = |name, op, len, role| OpLayer::new(name, op, len, role),
    },
    Linear => {
        wire = "linear",
        doc = "线性投影。",
        fields = {
            name: String => "层名。",
            in_features: usize => "输入宽度。",
            out_features: usize => "输出宽度。",
            format: WeightFormat => "权重编码；量化格式必须显式声明分组。"
        },
        build = |name, in_features, out_features, format| {
            Linear::with_format(name, format, in_features, out_features)
        },
    },
    MatMul => {
        wire = "matmul",
        doc = "矩阵乘。",
        fields = {
            name: String => "层名。",
            rows: usize => "输出行数。",
            cols: usize => "输出列数。",
            precision: Precision => "元素精度。",
            gauge: Gauge => "收缩轴档位。"
        },
        build = |name, rows, cols, precision, gauge| {
            MatMul::with_precision(name, rows, cols, precision, gauge)
        },
    },
    Embedding => {
        wire = "embedding",
        doc = "嵌入查表。",
        fields = {
            name: String => "层名。",
            vocab: usize => "词表大小。",
            dim: usize => "嵌入宽度。",
            tokens: usize => "token 数。"
        },
        build = |name, vocab, dim, tokens| Embedding::new(name, vocab, dim, tokens),
    },
    Attention => {
        wire = "attention",
        doc = "已切分 Q/K/V 的注意力。",
        fields = {
            name: String => "层名。",
            seq: usize => "序列长度。",
            head_dim: usize => "头宽。",
            algo: AttentionAlgo => "实现算法。"
        },
        build = |name, seq, head_dim, algo| Attention::new(name, seq, head_dim, algo),
    },
    DecodeAttention => {
        wire = "decode_attention",
        doc = "解码期注意力。",
        fields = {
            name: String => "层名。",
            q_rows: usize => "query 行数。",
            head_dim: usize => "头宽。",
            algo: AttentionAlgo => "实现算法。"
        },
        build = |name, q_rows, head_dim, algo| DecodeAttention::new(name, q_rows, head_dim, algo),
    },
    LatentAttention => {
        wire = "latent_attention",
        doc = "潜空间注意力。",
        fields = {
            name: String => "层名。",
            seq: usize => "序列长度。",
            latent: usize => "压缩宽度。",
            heads: usize => "头数。"
        },
        build = |name, seq, latent, heads| LatentAttention::new(name, seq, latent, heads),
    },
    AttentionBlock => {
        wire = "attention_block",
        doc = "Q/K/V、注意力与输出投影组成的组合块。",
        fields = {
            name: String => "块名。",
            hidden: usize => "隐藏宽度。",
            kv_width: usize => "Q/K/V 投影宽度。",
            head_dim: usize => "头宽。",
            seq: usize => "序列长度。",
            algo: AttentionAlgo => "实现算法。",
            format: WeightFormat => "投影权重编码。"
        },
        build = |name, hidden, kv_width, head_dim, seq, algo, format| {
            AttentionBlock::new(name, hidden, kv_width, head_dim, seq, algo, format)
        },
    },
    Recurrent => {
        wire = "recurrent",
        doc = "RNN / GRU / LSTM 循环层。",
        fields = {
            name: String => "层名。",
            cell: Cell => "单元类型。",
            input: usize => "输入宽度。",
            hidden: usize => "隐宽。",
            #[serde(default = "one")] layers: usize => "层数；缺省为 1。",
            #[serde(default = "one")] dirs: usize => "方向数（1 或 2）；缺省为单向。",
            #[serde(default = "yes")] bias: bool => "是否带偏置；缺省为 `true`。",
            #[serde(default)] batch_first: bool => "是否采用 batch-first 布局；缺省为 seq-first。"
        },
        build = |name, cell, input, hidden, layers, dirs, bias, batch_first| {
            Recurrent::with_layout(name, cell, input, hidden, layers, dirs, bias, batch_first)
        },
    },
    Memory => {
        wire = "memory",
        doc = "条件记忆层。",
        fields = {
            name: String => "层名。",
            width: usize => "隐藏宽度。",
            order: usize => "N-gram 阶。",
            seed: u64 => "哈希种子。",
            buckets: usize => "表槽数。",
            gate: f32 => "门控强度。"
        },
        build = |name, width, order, seed, buckets, gate| {
            MemoryBlock::new(name, width, order, seed, buckets, gate)
        },
    },
}

/// 把具体层收口到栈的类型擦除边界；只在配置出口写一次。
fn boxed<L: Contract + 'static>(layer: L) -> Box<dyn Contract> {
    Box::new(layer)
}

const fn one() -> usize {
    1
}

const fn yes() -> bool {
    true
}
