//! Embedding 查表层：`y[i, :] = table[idx[i], :]`（行向量 gather）。

use diagnostic::{Code, Error, Result};

use crate::{Bounds, BoundsResult, Contract, ParamRole, spec_accessors};

/// 嵌入查表层。
#[crate::spec(name = $name, out = [$tokens, $dim], ops = [$op], role = $role)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Embedding {
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
    /// 语义表（[`crate::ContractSpec::new`] 上报）。
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
        Ok(Self {
            spec: __layer_spec_Embedding!(name, tokens, dim, Self::OP, ParamRole::Embedding)?,
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
    spec_accessors!();

    fn embedding(&self) -> Option<&Embedding> {
        Some(self)
    }

    fn bounds(&self) -> BoundsResult {
        Bounds::of_context(Self::OP, self.tokens, 1).map(Some)
    }
}

#[cfg(test)]
#[path = "../tests/embed_tests.rs"]
mod embed_tests;
