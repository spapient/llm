//! 注意力层：`O = softmax(QKᵀ/√d)·V`。
//!
//! 序列长 `seq`（轴 `s`）与头宽 `head_dim`（轴 `d`）由构造给出；算法选朴素或 Flash，分别绑定
//! `attention_f32` 与 `flash_attention_f32`。两条算子共用同一副骨架与轴表，只有参考实现与中间
//! 布局不同，下界都按投影口径交给算子语义表推出。

use diagnostic::{Code, Error, Result};
use tensor::Shape;

use crate::{Bounds, BoundsResult, Contract, ContractSpec};

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
    fn op(self) -> &'static str {
        match self {
            Self::Naive => Attention::NAIVE,
            Self::Flash => Attention::FLASH,
        }
    }
}

/// 注意力层。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attention {
    spec: ContractSpec,
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
    /// （[`ContractSpec::new`] 上报）。
    pub fn new(
        name: impl Into<String>,
        seq: usize,
        head_dim: usize,
        algo: AttentionAlgo,
    ) -> Result<Self> {
        for (label, value) in [("seq", seq), ("head_dim", head_dim)] {
            if value == 0 {
                return Err(Error::new(
                    Code::Invalid,
                    format!("注意力层 {label} 必须非零（{label} = {value}）"),
                )
                .with("nn::Attention::new"));
            }
        }
        let spec = ContractSpec::new(
            name,
            Shape::new([seq, head_dim]),
            vec![algo.op().to_string()],
        )?;
        Ok(Self {
            spec,
            seq,
            head_dim,
            algo,
        })
    }

    /// 序列长（轴 `s`）。
    pub fn seq(&self) -> usize {
        self.seq
    }

    /// 头宽（轴 `d`）。
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// 选用的算法。
    pub fn algo(&self) -> AttentionAlgo {
        self.algo
    }
}

impl Contract for Attention {
    fn name(&self) -> &str {
        self.spec.name()
    }

    fn output(&self) -> &Shape {
        self.spec.output()
    }

    fn spca(&self) -> &ContractSpec {
        &self.spec
    }

    fn bounds(&self) -> BoundsResult {
        // 注意力声明了输入轴 `s` 与输出轴 `d`，按投影角色翻译宽度；代价仍来自算子语义表。
        Bounds::of_op(self.algo.op(), self.seq, self.head_dim).map(Some)
    }
}

#[cfg(test)]
mod tests;
