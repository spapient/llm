//! Embedding 查表层：`y[i, :] = table[idx[i], :]`（行向量 gather）。
//!
//! 层记录词表大小 `vocab`、行宽 `dim` 与一次查表的 token 数 `tokens`，用于对齐权重表
//! `[vocab, dim]`；下界跟随 `embed_f32` 的语义表（按被访问的 token 计价），层不自行把权重表
//! 整张读一遍——常驻表的流量口径属于算子代价模型，这里不写第二份公式。

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
mod tests {
    use super::*;

    #[test]
    fn an_embedding_declares_its_table_and_output_shape() {
        let layer = Embedding::new("tok_emb", 32_000, 128, 8).expect("合法");
        assert_eq!(layer.name(), "tok_emb");
        assert_eq!(layer.vocab(), 32_000);
        assert_eq!(layer.dim(), 128);
        assert_eq!(layer.tokens(), 8);
        assert_eq!(layer.output().dims(), [8, 128]);
        assert_eq!(layer.spca().ops(), [Embedding::OP]);
        assert_eq!(layer.spca().role(), ParamRole::Embedding);
        assert!(layer.spca().role().wd_exempt());
    }

    #[test]
    fn an_embedding_floor_follows_the_gather_semantics() {
        let layer = Embedding::new("tok_emb", 32_000, 128, 8).expect("合法");
        let floor = layer.bounds().expect("合法").expect("有算子");
        assert_eq!(
            floor,
            Bounds::of_context(Embedding::OP, 8, 1).expect("同一份语义")
        );
    }

    #[test]
    fn a_zero_dimension_embedding_is_rejected() {
        assert_eq!(
            Embedding::new("x", 0, 128, 8).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(
            Embedding::new("x", 32, 0, 8).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(
            Embedding::new("x", 32, 128, 0).unwrap_err().code(),
            Code::Invalid
        );
    }
}
