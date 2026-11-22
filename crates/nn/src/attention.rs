//! 注意力层：`O = softmax(QKᵀ/√d)·V`。

use diagnostic::{Code, Error, Result};
use tensor::Shape;

use crate::{
    Bounds, BoundsResult, Contract, ContractSpec, Linear, WEIGHT_SUFFIX, WeightFormat,
    spec_accessors,
};

/// 注意力实现算法。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionAlgo {
    /// 朴素路径：落地完整 `s×s` 分数矩阵，作为数值参考实现。
    Naive,
    /// Flash 路径：online softmax 分块扫描 K/V，不落地完整分数矩阵。
    Flash,
}

impl AttentionAlgo {
    /// 算法对应的已注册算子名。
    pub fn operator(self) -> &'static str {
        match self {
            Self::Naive => Attention::NAIVE,
            Self::Flash => Attention::FLASH,
        }
    }
}

/// 注意力层。
#[crate::spec(name = $name, out = [$seq, $head_dim], ops = [$op])]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attention {
    seq: usize,
    head_dim: usize,
    algo: AttentionAlgo,
}

impl Attention {
    /// 朴素注意力算子名。
    pub const NAIVE: &str = "attention_f32";
    /// Flash 注意力算子名。
    pub const FLASH: &str = "flash_attention_f32";

    /// 由层名、序列长、头宽与算法构造。
    ///
    /// # Errors
    ///
    /// `seq` / `head_dim` 任一为零（[`Code::Invalid`]）；算子未在 `op` 注册或没有语义表
    /// （[`crate::ContractSpec::new`] 上报）。
    pub fn new(
        name: impl Into<String>,
        seq: usize,
        head_dim: usize,
        algo: AttentionAlgo,
    ) -> Result<Self> {
        crate::nonzero_all(
            "nn::Attention::new",
            "注意力层 ",
            &[("seq", seq), ("head_dim", head_dim)],
        )?;
        Ok(Self {
            spec: __layer_spec_Attention!(name, seq, head_dim, algo.operator())?,
            seq,
            head_dim,
            algo,
        })
    }

    crate::accessors! {
        /// 序列长（轴 `s`）。
        seq -> usize = seq,
        /// 头宽（轴 `d`）。
        head_dim -> usize = head_dim,
        /// 选用的算法。
        algo -> AttentionAlgo = algo,
    }
}

impl Contract for Attention {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        // 注意力声明了输入轴 `s` 与输出轴 `d`，按投影角色翻译宽度；代价仍来自算子语义表。
        Bounds::of_op(self.algo.operator(), self.seq, self.head_dim).map(Some)
    }
}

// 解码期注意力：`q_rows` 行 query 各扫一遍 KV 历史，历史长度不进形状。
//
// 与 [`Attention`] 的差别在于谁承担 KV 长度：prefill 的 `seq` 既是行数又是因果长度，
// 而解码期每步只来少量新行（`q_rows` 常为 1），历史长度 `h` 是运行期状态、不是层的形状。
// 因此本层的 `bounds` 走算子语义表里固定的声明档 `HIST = 4`，是**松下界**（下界允许偏松，
// 但只要装配侧的 KV 缓存长于声明档，真实代价就更高，方向不会反）。

/// 解码期注意力层：输出 `[q_rows, head_dim]`，历史长度由算子语义表的声明档给出。
#[crate::spec(name = $name, out = [$q_rows, $head_dim], ops = [$op])]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeAttention {
    q_rows: usize,
    head_dim: usize,
    algo: AttentionAlgo,
}

impl DecodeAttention {
    /// 朴素 decode 算子名。
    pub const NAIVE: &str = "decode_attention_f32";
    /// Flash decode 算子名。
    pub const FLASH: &str = "decode_flash_attention_f32";

    /// 由层名、query 行数、头宽与算法构造。
    ///
    /// # Errors
    ///
    /// `q_rows` / `head_dim` 任一为零（[`Code::Invalid`]）；算子未注册或没有语义表
    /// （[`ContractSpec::new`] 上报）。
    pub fn new(
        name: impl Into<String>,
        q_rows: usize,
        head_dim: usize,
        algo: AttentionAlgo,
    ) -> Result<Self> {
        crate::nonzero_all(
            "nn::DecodeAttention::new",
            "解码注意力层 ",
            &[("q_rows", q_rows), ("head_dim", head_dim)],
        )?;
        Ok(Self {
            spec: __layer_spec_DecodeAttention!(name, q_rows, head_dim, Self::operator(algo))?,
            q_rows,
            head_dim,
            algo,
        })
    }

    /// 算法对应的 decode 算子名。
    fn operator(algo: AttentionAlgo) -> &'static str {
        match algo {
            AttentionAlgo::Naive => Self::NAIVE,
            AttentionAlgo::Flash => Self::FLASH,
        }
    }

    crate::accessors! {
        /// 本次解码的 query 行数（轴 `q`），通常为 1。
        q_rows -> usize = q_rows,
        /// 头宽（轴 `d`）。
        head_dim -> usize = head_dim,
        /// 选用的算法。
        algo -> AttentionAlgo = algo,
    }
}

impl Contract for DecodeAttention {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        // 历史轴 `h` 由算子的 `project_shape` 固定为声明档 `HIST`，层给不出真实 KV 长度
        // ⇒ 这是松下界；`q_rows` / `head_dim` 是层真正知道的两轴。
        Bounds::of_ctx(Self::operator(self.algo), self.q_rows, self.head_dim).map(Some)
    }
}

// 注意力组合块：Q/K/V 三个投影 + 注意力 + 输出投影，内聚成一个 [`Contract`] 层。
//
// 单独的 [`Attention`](crate::Attention) 只消费已经切好的 `q/k/v` 三路；真实模型里这三路
// 由同一个隐藏向量经三个投影得到，再过注意力、最后经输出投影回到隐藏宽度。组合块把这一串
// 收在一个类型里：层名前缀下挂 `.q` / `.k` / `.v` / `.attn` / `.o` 五个子层，权重条目按
// 子层名 + [`WEIGHT_SUFFIX`](crate::WEIGHT_SUFFIX) 对齐。
//
// 块不内聚残差旁路与层归一化：旁路引用的是块外的主干隐藏向量，归装配侧接线；多头展平与
// 头内切片仍由 [`Attention`](crate::Attention) 的既有几何口径承担，本块只保证 `kv_width` 是
// `head_dim` 的整数倍。

/// 一个注意力块：`hidden → [q, k, v] → attention → o → hidden`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionBlock {
    spec: ContractSpec,
    hidden: usize,
    kv_width: usize,
    head_dim: usize,
    q: Linear,
    k: Linear,
    v: Linear,
    attn: Attention,
    out: Linear,
}

impl AttentionBlock {
    /// 由块名、隐藏宽度、Q/K/V 宽度、头宽、序列长、算法与投影权重编码构造。
    ///
    /// # Errors
    ///
    /// - `hidden` / `kv_width` / `head_dim` / `seq` 任一为零（[`Code::Invalid`]）；
    /// - `kv_width` 不是 `head_dim` 的整数倍（头数不是整数）；
    /// - 任一层引用的算子未在 `op` 注册或没有语义表（由子层 [`ContractSpec`] 上报）。
    pub fn new(
        name: impl Into<String>,
        hidden: usize,
        kv_width: usize,
        head_dim: usize,
        seq: usize,
        algo: AttentionAlgo,
        format: WeightFormat,
    ) -> Result<Self> {
        let name = name.into();
        Self::validate(hidden, kv_width, head_dim, seq)?;
        let (q, k, v, out) = Self::projections(&name, format, hidden, kv_width)?;
        let attn = Attention::new(format!("{name}.attn"), seq, head_dim, algo)?;
        let spec = Self::spec(&name, hidden, algo, format)?;

        Ok(Self {
            spec,
            hidden,
            kv_width,
            head_dim,
            q,
            k,
            v,
            attn,
            out,
        })
    }

    /// 几何口径校验：四个宽度非零，且 `kv_width` 整除 `head_dim`。
    fn validate(hidden: usize, kv_width: usize, head_dim: usize, seq: usize) -> Result<()> {
        crate::nonzero_all(
            "nn::AttentionBlock::new",
            "注意力块 ",
            &[
                ("hidden", hidden),
                ("kv_width", kv_width),
                ("head_dim", head_dim),
                ("seq", seq),
            ],
        )?;
        if !kv_width.is_multiple_of(head_dim) {
            return Err(Error::new(
                Code::Invalid,
                format!("Q/K/V 宽度 {kv_width} 不是头宽 {head_dim} 的整数倍：头数不是整数"),
            )
            .with("nn::AttentionBlock::new"));
        }
        Ok(())
    }

    /// 四个投影：Q/K/V 为 `hidden → kv_width`，输出投影为 `kv_width → hidden`。
    fn projections(
        name: &str,
        format: WeightFormat,
        hidden: usize,
        kv_width: usize,
    ) -> Result<(Linear, Linear, Linear, Linear)> {
        let q = Linear::with_format(format!("{name}.q"), format, hidden, kv_width)?;
        let k = Linear::with_format(format!("{name}.k"), format, hidden, kv_width)?;
        let v = Linear::with_format(format!("{name}.v"), format, hidden, kv_width)?;
        let out = Linear::with_format(format!("{name}.o"), format, kv_width, hidden)?;
        Ok((q, k, v, out))
    }

    /// 块级规格：输出回到隐藏宽度，算子列去重后的投影算子 + 注意力算子。
    fn spec(
        name: &str,
        hidden: usize,
        algo: AttentionAlgo,
        format: WeightFormat,
    ) -> Result<ContractSpec> {
        let ops = format.operator().into_iter().chain([algo.operator()]);
        ContractSpec::new(name, Shape::new([hidden]), ops)
    }

    crate::accessors! {
        /// 隐藏宽度（块的输入与输出宽度）。
        hidden -> usize = hidden,
        /// Q/K/V 投影宽度（= 头数 × 头宽）。
        kv_width -> usize = kv_width,
        /// 头宽。
        head_dim -> usize = head_dim,
    }

    /// 头数：`kv_width / head_dim`；构造校验已保证整除，这里是纯派生，不再二次取余。
    pub fn heads(&self) -> usize {
        self.kv_width / self.head_dim
    }

    /// 序列长（注意力层几何）。
    pub fn seq(&self) -> usize {
        self.attn.seq()
    }

    /// 选用的注意力算法。
    pub fn algo(&self) -> AttentionAlgo {
        self.attn.algo()
    }

    /// Q 投影。
    pub fn q(&self) -> &Linear {
        &self.q
    }

    /// K 投影。
    pub fn k(&self) -> &Linear {
        &self.k
    }

    /// V 投影。
    pub fn v(&self) -> &Linear {
        &self.v
    }

    /// 注意力层。
    pub fn attention(&self) -> &Attention {
        &self.attn
    }

    /// 输出投影。
    pub fn out(&self) -> &Linear {
        &self.out
    }

    /// 子层权重条目名：`{块名}.{子层}{WEIGHT_SUFFIX}`；装配侧按名取回驻留权重整块。
    pub fn weight_entry(level: &str, block: &str) -> String {
        format!("{block}.{level}{WEIGHT_SUFFIX}")
    }

    /// 四个投影的下界之和；任一无可计价形状则整块上报 `None`。
    fn floor_of_projection(q: &Linear, k: &Linear, v: &Linear, out: &Linear) -> BoundsResult {
        let mut total = Bounds::ZERO;
        for projection in [q, k, v, out] {
            match projection.bounds()? {
                Some(floor) => total = total.plus(floor),
                None => return Ok(None),
            }
        }
        Ok(Some(total))
    }
}

impl Contract for AttentionBlock {
    spec_accessors!();

    fn attention_block(&self) -> Option<&AttentionBlock> {
        Some(self)
    }

    fn bounds(&self) -> BoundsResult {
        let projection = Self::floor_of_projection(&self.q, &self.k, &self.v, &self.out)?;
        let attention = self.attn.bounds()?;
        match (projection, attention) {
            (Some(p), Some(a)) => Ok(Some(p.plus(a))),
            _ => Ok(None),
        }
    }
}

/// 潜空间注意力层：KV 按 `d_c` 压缩存量，解耦 RoPE 键单头叠加。
///
/// 与 [`Attention`] 的区别只在 KV 的口径：本层不认识 `h · d_h` 的稠密宽度，只认识压缩宽度 `d_c`。
/// 形状仍是 `[seq, d_c]`，头数进代价（行数摊成 `seq · h`），不进形状——这与算子
/// `latent_attention_f32` 的几何完全一致：算子按压平行算，输出宽就是压缩宽度。
///
/// 上投影 `W^{UK}` / `W^{UV}` 不在本层：吸收式把它们折进查询与输出投影，稠密式则整块展开，
/// 两种折法都在装配侧，层只管压缩存量与读法。
#[crate::spec(name = $name, out = [$seq, $latent], ops = [$op])]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatentAttention {
    seq: usize,
    latent: usize,
    heads: usize,
}

impl LatentAttention {
    /// 算子名。
    pub const OPERATOR: &str = "latent_attention_f32";

    /// 由层名、查询行数、压缩宽度与头数构造。
    ///
    /// # Errors
    ///
    /// `seq` / `latent` / `heads` 任一为零（[`Code::Invalid`]）；算子未注册或没有语义表
    /// （[`ContractSpec::new`] 上报）。
    pub fn new(name: impl Into<String>, seq: usize, latent: usize, heads: usize) -> Result<Self> {
        crate::nonzero_all(
            "nn::LatentAttention::new",
            "潜空间注意力层 ",
            &[("seq", seq), ("latent", latent), ("heads", heads)],
        )?;
        Ok(Self {
            spec: __layer_spec_LatentAttention!(name, seq, latent, Self::OPERATOR)?,
            seq,
            latent,
            heads,
        })
    }

    crate::accessors! {
        /// 序列长度（轴 `s`）；摊平后喂给算子的行数是 `seq · heads`。
        seq -> usize = seq,
        /// 压缩宽度（轴 `c`），也是单头的键值宽度。
        latent -> usize = latent,
        /// 头数：不进形状，但行数与代价都按它摊开。
        heads -> usize = heads,
    }
}

impl Contract for LatentAttention {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        // 行数是摊平后的 `seq · h`；宽度只认压缩宽度 `d_c`，解耦 RoPE 那段属于装配侧的几何，
        // 由运行期行宽给出，层不假装知道它——否则下界会把上投影的活算进来，重复计价。
        Bounds::of_op(Self::OPERATOR, self.seq * self.heads, self.latent).map(Some)
    }
}

#[cfg(test)]
#[path = "../tests/attention_tests.rs"]
mod attention_tests;

#[cfg(test)]
#[path = "../tests/block_tests.rs"]
mod block_tests;
