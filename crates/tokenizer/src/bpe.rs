//! BPE 分词实现：在中性 [`Vocab`] 上按合并秩迭代归并相邻片段。
//!
//! 合并语义对齐公开教育版 BPE：初始为逐字节片段，每轮扫描相邻片段对，取秩最小
//! （合并表中最靠前）的一对归并，重复到表内无可用对；最终片段按词表反查编号。
//!
//! **边界（D2 待办）**：预切分（空白/标点/Unicode 脚本的词边界切分）尚未接入，
//! 当前把整段 UTF-8 字节当作一个序列处理；字节级可见字符映射（ByteLevel charset）
//! 同样随预切分层一起落地。合并算法本身不依赖正则。

use crate::{SpecialTokens, Token, Tokenizer, Vocab, VocabFamily};
use diagnostic::{Code, Error, Result};
use std::collections::HashMap;

/// 基于 BPE 词表的分词器；构造后词表与秩表不可变。
#[derive(Debug, Clone, PartialEq)]
pub struct Bpe {
    vocab: Vocab,
    /// 左片段 → （右片段 → 合并秩）；秩越小越先合并。
    ranks: HashMap<Vec<u8>, HashMap<Vec<u8>, usize>>,
}

impl Bpe {
    /// 由已闭合的 BPE 词表构造；族不是 BPE 报 [`Code::Unsupported`]。
    pub fn new(vocab: Vocab) -> Result<Self> {
        if vocab.family() != VocabFamily::Bpe {
            return Err(Error::new(
                Code::Unsupported,
                format!("BPE 实现不支持词表族 {:?}", vocab.family()),
            )
            .with("tokenizer::Bpe::new"));
        }
        let mut ranks: HashMap<Vec<u8>, HashMap<Vec<u8>, usize>> = HashMap::new();
        for (rank, merge) in vocab.merges().iter().enumerate() {
            ranks
                .entry(merge.left.clone())
                .or_default()
                .insert(merge.right.clone(), rank);
        }
        Ok(Self { vocab, ranks })
    }

    /// 中性词表只读视图。
    #[must_use]
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// 取 `(parts[i], parts[i+1])` 对应的合并秩；表内无此对返回 [`None`]。
    fn pair_rank(&self, index: usize, parts: &[Vec<u8>]) -> Option<usize> {
        let right_ranks = self.ranks.get(&parts[index])?;
        right_ranks.get(&parts[index + 1]).copied()
    }

    /// 扫描一轮相邻片段，返回本轮应归并的最左最小秩下标；无可用对返回 [`None`]。
    fn best_pair(&self, parts: &[Vec<u8>]) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for i in 0..parts.len().saturating_sub(1) {
            let Some(rank) = self.pair_rank(i, parts) else {
                continue;
            };
            if best.is_none_or(|(_, min_rank)| rank < min_rank) {
                best = Some((i, rank));
            }
        }
        best.map(|(index, _)| index)
    }

    /// 把 `idx` 与 `idx+1` 两个片段归并为一个片段。
    fn merge_at(parts: &mut Vec<Vec<u8>>, idx: usize) {
        let merged = [parts[idx].clone(), parts[idx + 1].clone()].concat();
        parts.splice(idx..idx + 2, [merged]);
    }
}

impl Tokenizer for Bpe {
    fn vocab_size(&self) -> usize {
        self.vocab.vocab_size()
    }

    fn encode(&self, text: &str) -> Result<Vec<Token>> {
        let mut parts: Vec<Vec<u8>> = text.bytes().map(|byte| vec![byte]).collect();
        while let Some(idx) = self.best_pair(&parts) {
            Self::merge_at(&mut parts, idx);
        }
        parts
            .iter()
            .map(|piece| {
                self.vocab.lookup(piece).ok_or_else(|| {
                    Error::new(Code::Invalid, "编码后的片段不在词表内")
                        .with("tokenizer::Bpe::encode")
                })
            })
            .collect()
    }

    fn decode(&self, tokens: &[Token]) -> Result<String> {
        let mut bytes = Vec::with_capacity(tokens.len());
        for token in tokens {
            let piece = self.vocab.piece(token.id()).ok_or_else(|| {
                Error::new(
                    Code::Invalid,
                    format!(
                        "token {} 超出 BPE 词表（0..{}）",
                        token.id(),
                        self.vocab_size()
                    ),
                )
                .with("tokenizer::Bpe::decode")
            })?;
            bytes.extend_from_slice(piece);
        }
        String::from_utf8(bytes).map_err(|error| {
            Error::new(Code::Invalid, format!("token 序列不是合法 UTF-8：{error}"))
                .with("tokenizer::Bpe::decode")
        })
    }

    fn fingerprint(&self) -> crate::Fingerprint {
        *self.vocab.fingerprint()
    }

    fn special_tokens(&self) -> &SpecialTokens {
        self.vocab.special()
    }

    fn lookup(&self, piece: &[u8]) -> Result<Option<Token>> {
        Ok(self.vocab.lookup(piece))
    }

    fn piece(&self, token: Token) -> Result<Option<Vec<u8>>> {
        Ok(self.vocab.piece(token.id()).map(<[u8]>::to_vec))
    }
}
