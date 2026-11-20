//! 合成词表夹具：单元测试、段编解码测试与跨 crate 的集成测试共用同一份构造。
//!
//! 只为测试存在（`#[doc(hidden)]`），不承诺 API 稳定：片列表与分数就是指纹的输入，
//! 改一处等于换一份词表，下游的对拍与段字节都要据此重算。

use crate::{Merge, SpecialTokens, VocabFamily, VocabSource};

/// 合成 BPE 词表：`a`/`b`/`ab` 三片，一条 `a+b` 合并，特殊片 `bos=0`、`eos=2`。
pub fn bpe_source() -> VocabSource {
    VocabSource {
        family: VocabFamily::Bpe,
        pieces: vec![b"a".to_vec(), b"b".to_vec(), b"ab".to_vec()],
        special: SpecialTokens::EMPTY.with_bos(0).with_eos(2),
        scores: Vec::new(),
        merges: vec![Merge {
            left: b"a".to_vec(),
            right: b"b".to_vec(),
        }],
    }
}

/// 合成 Unigram 词表：单字母 -1.0，`ab` -0.1（更优），`abc` -2.0，`<unk>` 0.0（编号 5）。
///
/// `register_unk` 为假时 `<unk>` 只占片位、不登记成特殊片，用于无未知片回退的词表。
pub fn unigram_source(register_unk: bool) -> VocabSource {
    let special = SpecialTokens::empty().with_unk(5);
    VocabSource {
        family: VocabFamily::Unigram,
        pieces: vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"c".to_vec(),
            b"ab".to_vec(),
            b"abc".to_vec(),
            b"<unk>".to_vec(),
        ],
        scores: vec![-1.0, -1.0, -1.0, -0.1, -2.0, 0.0],
        special: if register_unk {
            special
        } else {
            SpecialTokens::EMPTY
        },
        ..VocabSource::default()
    }
}
