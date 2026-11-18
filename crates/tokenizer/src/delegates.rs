//! 词表委托方法的共享宏：`Bpe` 与 `Unigram` 在 [`Vocab`] 上的只读面完全相同，
//! 差异只在错误文案里的词表名与调用点标记。这里用一张宏表收敛，新分词算法
//! （WordPiece / SentencePiece）落地时只传两个字符串，不再重写这五个方法。

/// 生成 `Tokenizer` 里五个只读委托：`decode`、`fingerprint`、`special_tokens`、
/// `lookup`、`piece`。调用点在 `impl Tokenizer for X` 内，且该结构体必须持有
/// `vocab: crate::Vocab` 字段。
///
/// 参数：`$label` 是错误文案里的词表名（如 `"BPE"`），`$marker` 是错误调用点标记
/// （如 `"tokenizer::Bpe::decode"`）。
macro_rules! vocab_tokenizer_delegates {
    ($label:literal, $marker:literal) => {
        fn decode(&self, tokens: &[crate::Token]) -> diagnostic::Result<String> {
            let mut bytes = Vec::with_capacity(tokens.len());
            for token in tokens {
                let piece = self.vocab.piece(token.id()).ok_or_else(|| {
                    diagnostic::Error::new(
                        diagnostic::Code::Invalid,
                        format!(
                            "token {} 超出 {} 词表（0..{}）",
                            token.id(),
                            $label,
                            self.vocab_size()
                        ),
                    )
                    .with($marker)
                })?;
                bytes.extend_from_slice(piece);
            }
            String::from_utf8(bytes).map_err(|error| {
                diagnostic::Error::new(
                    diagnostic::Code::Invalid,
                    format!("token 序列不是合法 UTF-8：{error}"),
                )
                .with($marker)
            })
        }

        fn fingerprint(&self) -> crate::Fingerprint {
            *self.vocab.fingerprint()
        }

        fn special_tokens(&self) -> &crate::SpecialTokens {
            self.vocab.special()
        }

        fn lookup(&self, piece: &[u8]) -> diagnostic::Result<Option<crate::Token>> {
            Ok(self.vocab.lookup(piece))
        }

        fn piece(&self, token: crate::Token) -> diagnostic::Result<Option<Vec<u8>>> {
            Ok(self.vocab.piece(token.id()).map(<[u8]>::to_vec))
        }
    };
}
