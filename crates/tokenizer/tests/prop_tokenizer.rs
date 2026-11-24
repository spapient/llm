//! 字节级分词与流式解码的性质：编码→解码恒等、分片不变性、越界词表必拒。
//!
//! 字节级实现是参考实现，无损是它的定义；流式解码器把"字符被切在两块之间"
//! 当成常态，分片方式不该改变拼回来的文本——这两条都是等号，不掺容差。

use llm_tokenizer::{Bytes, DecoderStream, Token, Tokenizer};
use proptest::prelude::*;
use testkit::prop;

proptest! {
    #![proptest_config(prop::config())]

    /// 编码→解码是恒等，且 token 数恰是 UTF-8 字节数：多字节字符不能被折算，
    /// 也不能在解码时丢字节。
    #[test]
    fn every_text_round_trips_through_bytes(text in any::<String>()) {
        let tokens = Bytes.encode(&text).expect("字节级编码无失败理由");
        prop_assert_eq!(tokens.len(), text.len());
        let decoded = Bytes.decode(&tokens).expect("自己编出的序列必须可解");
        prop_assert_eq!(decoded, text);
    }

    /// 分片不变性：合法 UTF-8 在任意字节位置切开（含切在多字节字符中间），
    /// 逐块 push 的输出拼上 finish 必须等于原文。这条失败意味着解码器在
    /// 分片边界上丢字符或误报非法。
    #[test]
    fn chunking_never_changes_the_decoded_text(
        text in any::<String>(),
        cuts in proptest::collection::vec(any::<usize>(), 0..=8),
    ) {
        let bytes = text.as_bytes();
        let mut cuts: Vec<usize> = cuts.iter().map(|c| c % (bytes.len() + 1)).collect();
        cuts.sort_unstable();
        cuts.dedup();
        let mut stream = DecoderStream::new();
        let mut out = String::new();
        let mut prev = 0;
        for cut in cuts.iter().chain(std::iter::once(&bytes.len())) {
            out.push_str(&stream.push(&bytes[prev..*cut]).expect("合法 UTF-8 的任意分片都不该报错"));
            prev = *cut;
        }
        out.push_str(&stream.finish().expect("完整序列收尾必成功"));
        prop_assert_eq!(out, text);
    }

    /// 越界 token 必拒：字节级词表只有 256，超出就是坏序列，不能悄悄折算。
    #[test]
    fn out_of_vocab_tokens_are_always_rejected(id in 256u32..) {
        prop_assert!(Bytes.decode(&[Token::new(id)]).is_err(), "token {id} 被接受");
    }
}