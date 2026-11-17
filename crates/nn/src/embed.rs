//! Embedding 查表层：`y[i, :] = table[idx[i], :]`（行向量 gather）。
//!
//! 层记录词表大小 `vocab`、行宽 `dim` 与一次查表的 token 数 `tokens`，用于对齐权重表
//! `[vocab, dim]`；下界跟随 `embed_f32` 的语义表（按被访问的 token 计价），层不自行把权重表
//! 整张读一遍——常驻表的流量口径属于算子代价模型，这里不写第二份公式。

use diagnostic::{Code, Error, Result};
use tensor::Shape;

use crate::{Bounds, BoundsResult, Contract, ContractSpec, ParamRole};

/// 嵌入查表层。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Embedding {
    spec: ContractSpec,
    vocab: usize,
    dim: usize,
    tokens: usize,
}

impl Embedding {
    /// 绑定的算子名。
    pub const OP: &str = "embed_f32";

    /// 由层名、词表大小、行宽与 token 数构造。
    ///
    /// # Errors
    ///
    /// `vocab` / `dim` / `tokens` 任一为零（[`Code::Invalid`]）；算子未在 `op` 注册或没有
    /// 语义表（[`ContractSpec::new`] 上报）。
    pub fn new(name: impl Into<String>, vocab: usize, dim: usize, tokens: usize) -> Result<Self> {
        for (label, value) in [("vocab", vocab), ("dim", dim), ("tokens", tokens)] {
            if value == 0 {
                return Err(Error::new(
                    Code::Invalid,
                    format!("Embedding 的 {label} 必须非零（{label} = {value}）"),
                )
                .with("nn::Embedding::new"));
            }
        }
        let spec = ContractSpec::new_with_role(
            name,
            Shape::new([tokens, dim]),
            vec![Self::OP.to_string()],
            ParamRole::Embedding,
        )?;
        Ok(Self {
            spec,
            vocab,
            dim,
            tokens,
        })
    }

    /// 词表大小（权重表行数）。
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// 行宽（每个 token 向量的维度）。
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 一次查表的 token 数。
    pub fn tokens(&self) -> usize {
        self.tokens
    }
}

impl Contract for Embedding {
    fn name(&self) -> &str {
        self.spec.name()
    }

    fn output(&self) -> &Shape {
        self.spec.output()
    }

    fn spca(&self) -> &ContractSpec {
        &self.spec
    }

    fn embedding(&self) -> Option<&Embedding> {
        Some(self)
    }

    fn bounds(&self) -> BoundsResult {
        Bounds::of_context(Self::OP, self.tokens, 1).map(Some)
    }
}

#[cfg(test)]
#[path = "../tests/embed_tests.rs"]
mod tests;
