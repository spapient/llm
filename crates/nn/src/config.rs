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

/// 可由模型描述逐条生成的层配置。
///
/// 这里没有第二套层实现：[`LayerConfig::build`] 的产物就是原有具体类型的 trait object，所以
/// `Contract::linear` / `embedding` / `attention_block` / `recurrent` 等类型化视图全部保留。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum LayerConfig {
    /// 任意已注册的单轴算子；`op` 可直接来自配置文件。
    Axis {
        /// 层名。
        name: String,
        /// 注册算子名。
        op: String,
        /// 轴长。
        len: usize,
        /// 参数角色；缺省为普通可训练参数。
        #[serde(default)]
        role: ParamRole,
    },
    /// 线性投影。
    Linear {
        /// 层名。
        name: String,
        /// 输入宽度。
        in_features: usize,
        /// 输出宽度。
        out_features: usize,
        /// 权重编码；量化格式必须显式声明分组。
        format: WeightFormat,
    },
    /// 矩阵乘。
    #[serde(rename = "matmul")]
    MatMul {
        /// 层名。
        name: String,
        /// 输出行数。
        rows: usize,
        /// 输出列数。
        cols: usize,
        /// 元素精度。
        precision: Precision,
        /// 收缩轴档位。
        gauge: Gauge,
    },
    /// 嵌入查表。
    Embedding {
        /// 层名。
        name: String,
        /// 词表大小。
        vocab: usize,
        /// 嵌入宽度。
        dim: usize,
        /// token 数。
        tokens: usize,
    },
    /// 已切分 Q/K/V 的注意力。
    Attention {
        /// 层名。
        name: String,
        /// 序列长度。
        seq: usize,
        /// 头宽。
        head_dim: usize,
        /// 实现算法。
        algo: AttentionAlgo,
    },
    /// 解码期注意力。
    DecodeAttention {
        /// 层名。
        name: String,
        /// query 行数。
        q_rows: usize,
        /// 头宽。
        head_dim: usize,
        /// 实现算法。
        algo: AttentionAlgo,
    },
    /// 潜空间注意力。
    LatentAttention {
        /// 层名。
        name: String,
        /// 序列长度。
        seq: usize,
        /// 压缩宽度。
        latent: usize,
        /// 头数。
        heads: usize,
    },
    /// Q/K/V、注意力与输出投影组成的组合块。
    AttentionBlock {
        /// 块名。
        name: String,
        /// 隐藏宽度。
        hidden: usize,
        /// Q/K/V 投影宽度。
        kv_width: usize,
        /// 头宽。
        head_dim: usize,
        /// 序列长度。
        seq: usize,
        /// 实现算法。
        algo: AttentionAlgo,
        /// 投影权重编码。
        format: WeightFormat,
    },
    /// RNN / GRU / LSTM 循环层。
    Recurrent {
        /// 层名。
        name: String,
        /// 单元类型。
        cell: Cell,
        /// 输入宽度。
        input: usize,
        /// 隐宽。
        hidden: usize,
        /// 层数；缺省为 1。
        #[serde(default = "one")]
        layers: usize,
        /// 方向数（1 或 2）；缺省为单向。
        #[serde(default = "one")]
        dirs: usize,
        /// 是否带偏置；缺省为 `true`。
        #[serde(default = "yes")]
        bias: bool,
        /// 是否采用 batch-first 布局；缺省为 seq-first。
        #[serde(default)]
        batch_first: bool,
    },
    /// 条件记忆层。
    Memory {
        /// 层名。
        name: String,
        /// 隐藏宽度。
        width: usize,
        /// N-gram 阶。
        order: usize,
        /// 哈希种子。
        seed: u64,
        /// 表槽数。
        buckets: usize,
        /// 门控强度。
        gate: f32,
    },
}

impl LayerConfig {
    /// 稳定的配置类型短名，与序列化字段 `type` 一致。
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Axis { .. } => "axis",
            Self::Linear { .. } => "linear",
            Self::MatMul { .. } => "matmul",
            Self::Embedding { .. } => "embedding",
            rest => Self::kind_attention(rest),
        }
    }

    /// 注意力族短名：四种 Q/K/V 变体同属一条装配族。
    const fn kind_attention(this: &Self) -> &'static str {
        match this {
            Self::Attention { .. } => "attention",
            Self::DecodeAttention { .. } => "decode_attention",
            Self::LatentAttention { .. } => "latent_attention",
            Self::AttentionBlock { .. } => "attention_block",
            rest => Self::kind_recurrent(rest),
        }
    }

    /// 循环与记忆族短名：收尾两臂穷尽枚举剩余变体。
    const fn kind_recurrent(this: &Self) -> &'static str {
        match this {
            Self::Recurrent { .. } => "recurrent",
            Self::Memory { .. } => "memory",
        }
    }

    /// 配置声明的层名；构造前即可用于诊断与重名检查。
    pub fn name(&self) -> &str {
        match self {
            Self::Axis { name, .. }
            | Self::Linear { name, .. }
            | Self::MatMul { name, .. }
            | Self::Embedding { name, .. }
            | Self::Attention { name, .. }
            | Self::DecodeAttention { name, .. }
            | Self::LatentAttention { name, .. }
            | Self::AttentionBlock { name, .. }
            | Self::Recurrent { name, .. }
            | Self::Memory { name, .. } => name,
        }
    }

    /// 用原有构造器物化这一层；所有结构错误都在返回 trait object 前报出。
    ///
    /// # Errors
    ///
    /// 错误原样来自对应层构造器：未注册算子、零宽形状、非法分组或非法几何均不会被配置层改写。
    pub fn build(self) -> Result<Box<dyn Contract>> {
        match self {
            Self::Axis {
                name,
                op,
                len,
                role,
            } => OpLayer::new(name, op, len, role).map(boxed),
            Self::Linear {
                name,
                in_features,
                out_features,
                format,
            } => Linear::with_format(name, format, in_features, out_features).map(boxed),
            Self::MatMul {
                name,
                rows,
                cols,
                precision,
                gauge,
            } => MatMul::with_precision(name, rows, cols, precision, gauge).map(boxed),
            Self::Embedding {
                name,
                vocab,
                dim,
                tokens,
            } => Embedding::new(name, vocab, dim, tokens).map(boxed),
            rest => Self::build_attention(rest),
        }
    }

    /// 注意力族构造：Q/K/V、解码期、潜空间与组合块四条通路。
    fn build_attention(config: LayerConfig) -> Result<Box<dyn Contract>> {
        match config {
            Self::Attention {
                name,
                seq,
                head_dim,
                algo,
            } => Attention::new(name, seq, head_dim, algo).map(boxed),
            Self::DecodeAttention {
                name,
                q_rows,
                head_dim,
                algo,
            } => DecodeAttention::new(name, q_rows, head_dim, algo).map(boxed),
            Self::LatentAttention {
                name,
                seq,
                latent,
                heads,
            } => LatentAttention::new(name, seq, latent, heads).map(boxed),
            Self::AttentionBlock {
                name,
                hidden,
                kv_width,
                head_dim,
                seq,
                algo,
                format,
            } => {
                AttentionBlock::new(name, hidden, kv_width, head_dim, seq, algo, format)
                    .map(boxed)
            }
            rest => Self::build_recurrent(rest),
        }
    }

    /// 循环与记忆族构造：收尾两臂穷尽枚举剩余变体。
    fn build_recurrent(config: LayerConfig) -> Result<Box<dyn Contract>> {
        match config {
            Self::Recurrent {
                name,
                cell,
                input,
                hidden,
                layers,
                dirs,
                bias,
                batch_first,
            } => Recurrent::with_layout(name, cell, input, hidden, layers, dirs, bias, batch_first)
                .map(boxed),
            Self::Memory {
                name,
                width,
                order,
                seed,
                buckets,
                gate,
            } => MemoryBlock::new(name, width, order, seed, buckets, gate).map(boxed),
        }
    }
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
