//! Unigram 分词实现：在中性 [`Vocab`] 上按逐片分数做前向 Viterbi 最优切分。
//!
//! 算法语义对齐公开 Unigram 参照：`best[0] = 0`；对每个终点位置，枚举所有以该点
//! 结束的词片，取 `best[start] + score(piece)` 最大者；前向表建好后沿回指指针回溯。
//! 分数累积用 `f64`（多个 f32 对数分数求和在长输入上的舍入误差不应翻转近平局切分）。
//! 严格 `>` 比优，同分时保留最先扫描到的（最左）片段。
//!
//! **边界（D2 待办）**：预切分（空白/标点/脚本边界）、归一化与 SentencePiece `▁`
//! 空白映射尚未接入，当前把整段 UTF-8 字节当作一个序列做片匹配。未覆盖路径回退到
//! 已登记的 unknown 编号；未配置 `unk` 时报 [`Code::Invalid`]。

use crate::{SpecialTokens, Token, Tokenizer, Vocab, VocabFamily};
use diagnostic::{Code, Error, Result};

/// 基于 Unigram 词表的分词器；构造后词表与分数不可变。
#[derive(Debug, Clone, PartialEq)]
pub struct Unigram {
    vocab: Vocab,
}

impl Unigram {
    /// 由已闭合的 Unigram 词表构造；族不是 Unigram 报 [`Code::Unsupported`]。
    pub fn new(vocab: Vocab) -> Result<Self> {
        if vocab.family() != VocabFamily::Unigram {
            return Err(Error::new(
                Code::Unsupported,
                format!("Unigram 实现不支持词表族 {:?}", vocab.family()),
            )
            .with("tokenizer::Unigram::new"));
        }
        Ok(Self { vocab })
    }

    /// 中性词表只读视图。
    #[must_use]
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// 前向 Viterbi：`best[end]` 是切到字节 `end` 的最优累计分数，`back[end]` 记录回指起点与编号。
    fn forward(&self, bytes: &[u8]) -> (Vec<f64>, Vec<Option<(usize, u32)>>) {
        let n = bytes.len();
        let mut best = vec![f64::NEG_INFINITY; n + 1];
        let mut back = vec![None; n + 1];
        best[0] = 0.0;

        for end in 1..=n {
            self.relax_end(end, bytes, &mut best, &mut back);
        }
        (best, back)
    }

    /// 松弛到终点 `end`：枚举所有以该点结束的词片，更新最优累计分数与回指指针。
    fn relax_end(
        &self,
        end: usize,
        bytes: &[u8],
        best: &mut [f64],
        back: &mut [Option<(usize, u32)>],
    ) {
        for start in 0..end {
            if !best[start].is_finite() {
                continue;
            }
            let Some((id, score)) = self.piece_score(&bytes[start..end]) else {
                continue;
            };
            let candidate = best[start] + score;
            if candidate > best[end] {
                best[end] = candidate;
                back[end] = Some((start, id));
            }
        }
    }

    /// 词片对应的编号与分数；词表未命中返回 [`None`]。
    fn piece_score(&self, piece: &[u8]) -> Option<(u32, f64)> {
        let id = self.vocab.lookup(piece)?;
        let score = self.vocab.scores()[id.id() as usize] as f64;
        Some((id.id(), score))
    }

    /// 回指回溯：从终点沿回指指针收编号，再反转成左到右顺序。
    fn backtrack(back: &[Option<(usize, u32)>], end: usize) -> Vec<Token> {
        let mut ids = Vec::new();
        let mut pos = end;
        while pos > 0 {
            let Some((start, id)) = back[pos] else {
                break;
            };
            ids.push(Token::new(id));
            pos = start;
        }
        ids.reverse();
        ids
    }

    /// 无完整覆盖路径时：逐字节回退，词表内单字节直取，缺失字节用 unknown 编号。
    fn unknown_fallback(&self, bytes: &[u8]) -> Result<Vec<Token>> {
        let unk = self.vocab.special().unk().ok_or_else(|| {
            Error::new(
                Code::Invalid,
                "片段在词表内无覆盖路径，且未配置 unknown 编号",
            )
            .with("tokenizer::Unigram::encode")
        })?;
        let mut out = Vec::with_capacity(bytes.len());
        for byte in bytes {
            match self.vocab.lookup(&[*byte]) {
                Some(id) => out.push(id),
                None => out.push(unk),
            }
        }
        Ok(out)
    }
}

impl Tokenizer for Unigram {
    fn vocab_size(&self) -> usize {
        self.vocab.vocab_size()
    }

    fn encode(&self, text: &str) -> Result<Vec<Token>> {
        let bytes = text.as_bytes();
        let (best, back) = self.forward(bytes);
        if best[bytes.len()].is_finite() {
            return Ok(Self::backtrack(&back, bytes.len()));
        }
        self.unknown_fallback(bytes)
    }

    fn decode(&self, tokens: &[Token]) -> Result<String> {
        let mut bytes = Vec::with_capacity(tokens.len());
        for token in tokens {
            let piece = self.vocab.piece(token.id()).ok_or_else(|| {
                Error::new(
                    Code::Invalid,
                    format!(
                        "token {} 超出 Unigram 词表（0..{}）",
                        token.id(),
                        self.vocab_size()
                    ),
                )
                .with("tokenizer::Unigram::decode")
            })?;
            bytes.extend_from_slice(piece);
        }
        String::from_utf8(bytes).map_err(|error| {
            Error::new(Code::Invalid, format!("token 序列不是合法 UTF-8：{error}"))
                .with("tokenizer::Unigram::decode")
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
