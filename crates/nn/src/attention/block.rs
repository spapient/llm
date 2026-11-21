//! 注意力组合块：Q/K/V 三个投影 + 注意力 + 输出投影，内聚成一个 [`Contract`] 层。
//!
//! 单独的 [`Attention`](crate::Attention) 只消费已经切好的 `q/k/v` 三路；真实模型里这三路
//! 由同一个隐藏向量经三个投影得到，再过注意力、最后经输出投影回到隐藏宽度。组合块把这一串
//! 收在一个类型里：层名前缀下挂 `.q` / `.k` / `.v` / `.attn` / `.o` 五个子层，权重条目按
//! 子层名 + [`WEIGHT_SUFFIX`](crate::WEIGHT_SUFFIX) 对齐。
//!
//! 块不内聚残差旁路与层归一化：旁路引用的是块外的主干隐藏向量，归装配侧接线；多头展平与
//! 头内切片仍由 [`Attention`](crate::Attention) 的既有几何口径承担，本块只保证 `kv_width` 是
//! `head_dim` 的整数倍。

use diagnostic::{Code, Error, Result};
use tensor::Shape;

use crate::{
    Attention, AttentionAlgo, Bounds, BoundsResult, Contract, ContractSpec, Linear, WEIGHT_SUFFIX,
    WeightFormat, spec_accessors,
};

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
        for (label, value) in [
            ("hidden", hidden),
            ("kv_width", kv_width),
            ("head_dim", head_dim),
            ("seq", seq),
        ] {
            if value == 0 {
                return Err(Error::new(
                    Code::Invalid,
                    format!("注意力块 {label} 必须非零（{label} = {value}）"),
                )
                .with("nn::AttentionBlock::new"));
            }
        }
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
        let mut ops: Vec<String> = Vec::new();
        if let Some(projection) = format.operator() {
            ops.push(projection.to_string());
        }
        ops.push(algo.operator().to_string());
        ContractSpec::new(name, Shape::new([hidden]), ops)
    }

    /// 隐藏宽度（块的输入与输出宽度）。
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Q/K/V 投影宽度（= 头数 × 头宽）。
    pub fn kv_width(&self) -> usize {
        self.kv_width
    }

    /// 头宽。
    pub fn head_dim(&self) -> usize {
        self.head_dim
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

#[cfg(test)]
#[path = "../../tests/block_tests.rs"]
mod block_tests;
