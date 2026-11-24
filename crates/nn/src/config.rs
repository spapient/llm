//! 运行时层配置：一份数据选择并构造既有层，不另设形状、算子或计价实现。
//!
//! 配置覆盖 transformer、循环与条件记忆的主干层。每个变体只保存构造参数；[`LayerConfig::build`]
//! 直接调用对应类型的公开构造器，因此零宽、分组、注册名和重复层名的判据与手写装配完全一致。
//! 空间层和下游自定义层仍可通过 [`crate::Stack::of`] 与配置层混装。

use diagnostic::Result;

use crate::{
    Attention, AttentionAlgo, AttentionBlock, Cell, Contract, DecodeAttention, Embedding, Gauge,
    LatentAttention, Linear, MatMul, MemoryBlock, OpLayer, ParamRole, Precision, Recurrent,
    WeightFormat,
};

/// 可由外部模型描述逐条生成的层配置。
///
/// 这里没有第二套 `Layer` 实现：`build` 的产物就是原有具体类型的 trait object，所以
/// `Contract::linear` / `embedding` / `attention_block` / `recurrent` 等类型化视图全部保留。
#[derive(Debug, Clone, PartialEq)]
pub enum LayerConfig {
    /// 任意已注册的单轴算子；`op` 是拥有所有权的字符串，可直接来自配置文件。
    Axis {
        /// 层名。
        name: String,
        /// 注册算子名。
        op: String,
        /// 轴长。
        len: usize,
        /// 参数角色。
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
        /// 权重编码。
        format: WeightFormat,
    },
    /// 矩阵乘。
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
        /// 层数。
        layers: usize,
        /// 方向数（1 或 2）。
        dirs: usize,
        /// 是否带偏置。
        bias: bool,
        /// 是否采用 batch-first 布局。
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
            } => AttentionBlock::new(name, hidden, kv_width, head_dim, seq, algo, format).map(boxed),
            Self::Recurrent {
                name,
                cell,
                input,
                hidden,
                layers,
                dirs,
                bias,
                batch_first,
            } => Recurrent::with_layout(
                name,
                cell,
                input,
                hidden,
                layers,
                dirs,
                bias,
                batch_first,
            )
            .map(boxed),
            Self::Memory {
                name,
                width,
                order,
                seed,
                buckets,
                gate,
            } => MemoryBlock::new(&name, width, order, seed, buckets, gate).map(boxed),
        }
    }
}

/// 把具体层收口到栈的类型擦除边界；只在配置出口写一次。
fn boxed<L: Contract + 'static>(layer: L) -> Box<dyn Contract> {
    Box::new(layer)
}
