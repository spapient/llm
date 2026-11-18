//! 中性词表值对象：格式无关的词片表、特殊句柄与族附属数据。
//!
//! 词表只持有可反查的纯数据；算法归并/切分规则由具体实现消费，构造期闭合
//! 双向查表、编号范围与族附属表三类不变量。适配器在装配点把包内字段映射成
//! [`VocabSource`]，本结构不出现任何格式键名或容器字节。

use crate::{Fingerprint, SpecialTokens, Token};
use diagnostic::{Code, Error, Result};
use std::collections::{HashMap, HashSet};

/// 词表算法族。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabFamily {
    /// BPE：按合并对迭代归并相邻片段。
    Bpe,
    /// Unigram：按片段分数做最优切分。
    Unigram,
    /// WordPiece：前缀标记 + 词内子词贪心。
    WordPiece,
    /// 整词词表：不做子词切分。
    WordLevel,
    /// 字节级基线：一字节一片。
    Byte,
}

impl VocabFamily {
    /// 族名（实现标识，进词表指纹）。
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            VocabFamily::Bpe => "bpe",
            VocabFamily::Unigram => "unigram",
            VocabFamily::WordPiece => "wordpiece",
            VocabFamily::WordLevel => "wordlevel",
            VocabFamily::Byte => "byte",
        }
    }

    /// 稳定 wire 序号（中性字节布局用，区别于族名字符串）。
    pub(crate) const fn id(self) -> u8 {
        match self {
            VocabFamily::Bpe => 0,
            VocabFamily::Unigram => 1,
            VocabFamily::WordPiece => 2,
            VocabFamily::WordLevel => 3,
            VocabFamily::Byte => 4,
        }
    }

    /// 由 wire 序号取族；未知序号返回 `None`。
    pub(crate) const fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(VocabFamily::Bpe),
            1 => Some(VocabFamily::Unigram),
            2 => Some(VocabFamily::WordPiece),
            3 => Some(VocabFamily::WordLevel),
            4 => Some(VocabFamily::Byte),
            _ => None,
        }
    }

    /// 本族是否必须携带合并表。
    pub(crate) const fn requires_merges(self) -> bool {
        matches!(self, VocabFamily::Bpe)
    }

    /// 本族是否必须携带逐片分数。
    pub(crate) const fn requires_scores(self) -> bool {
        matches!(self, VocabFamily::Unigram)
    }
}

/// BPE 合并对；秩由在 [`VocabSource::merges`] 中的下标决定，靠前的合并优先。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Merge {
    /// 左侧片段字节。
    pub left: Vec<u8>,
    /// 右侧片段字节。
    pub right: Vec<u8>,
}

/// 装配点提供给 [`Vocab`] 的纯数据；不含格式键名、容器字节或解析状态。
#[derive(Debug, Clone, PartialEq)]
pub struct VocabSource {
    /// 算法族。
    pub family: VocabFamily,
    /// 词片表：下标即编号。
    pub pieces: Vec<Vec<u8>>,
    /// 特殊 token 句柄。
    pub special: SpecialTokens,
    /// 逐片分数（仅 Unigram）。
    pub scores: Vec<f32>,
    /// 合并对表（仅 BPE）。
    pub merges: Vec<Merge>,
}

impl Default for VocabSource {
    fn default() -> Self {
        Self {
            family: VocabFamily::Byte,
            pieces: Vec::new(),
            special: SpecialTokens::EMPTY,
            scores: Vec::new(),
            merges: Vec::new(),
        }
    }
}

/// 一份可反查的中性词表；构造后不可变。
#[derive(Debug, Clone, PartialEq)]
pub struct Vocab {
    family: VocabFamily,
    pieces: Vec<Vec<u8>>,
    piece_map: HashMap<Vec<u8>, u32>,
    special: SpecialTokens,
    scores: Vec<f32>,
    merges: Vec<Merge>,
    fingerprint: Fingerprint,
}

impl Vocab {
    /// 构造词表并闭合不变量；违例分别归入 [`Code::Invalid`] / [`Code::Missing`]。
    pub fn try_new(source: VocabSource) -> Result<Self> {
        let len = source.pieces.len();
        let piece_map = build_map(&source.pieces)?;
        check_special(&source.special, len)?;
        check_attachments(
            source.family,
            &source.scores,
            &source.merges,
            &piece_map,
            len,
        )?;
        let mut fingerprint = Fingerprint::from_pieces(source.family.as_str(), len, &source.pieces);
        fingerprint = fold_merges(fingerprint, &source.merges);
        fingerprint = fold_scores(fingerprint, &source.scores);
        Ok(Self {
            family: source.family,
            pieces: source.pieces,
            piece_map,
            special: source.special,
            scores: source.scores,
            merges: source.merges,
            fingerprint,
        })
    }

    /// 算法族。
    #[must_use]
    pub fn family(&self) -> VocabFamily {
        self.family
    }

    /// 词表大小（含特殊 token）。
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// 词片表（编号 → 字节）。
    #[must_use]
    pub fn pieces(&self) -> &[Vec<u8>] {
        &self.pieces
    }

    /// 特殊 token 句柄。
    #[must_use]
    pub fn special(&self) -> &SpecialTokens {
        &self.special
    }

    /// 逐片分数（仅 Unigram 非空）。
    #[must_use]
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    /// 合并对表（仅 BPE 非空）。
    #[must_use]
    pub fn merges(&self) -> &[Merge] {
        &self.merges
    }

    /// 词表身份指纹。
    #[must_use]
    pub fn fingerprint(&self) -> &Fingerprint {
        &self.fingerprint
    }

    /// 把闭合词表折成对应算法的分词器；族无实现时报 [`Code::Unsupported`]。
    ///
    /// 这是"词表 → 算法"的唯一闭集扇出：上层只认 [`crate::Tokenizer`]，不关心具体实现名。
    pub fn into_tokenizer(self) -> Result<Box<dyn crate::Tokenizer>> {
        match self.family {
            VocabFamily::Bpe => Ok(Box::new(crate::Bpe::new(self)?)),
            VocabFamily::Unigram => Ok(Box::new(crate::Unigram::new(self)?)),
            VocabFamily::Byte => Ok(Box::new(crate::Bytes)),
            other => Err(Error::new(
                Code::Unsupported,
                format!("词表族 {other:?} 尚无分词器实现"),
            )
            .with("tokenizer::Vocab::into_tokenizer")),
        }
    }

    /// 编号 → 词片字节；越界返回 [`None`]。
    #[must_use]
    pub fn piece(&self, id: u32) -> Option<&[u8]> {
        self.pieces
            .get(usize::try_from(id).ok()?)
            .map(Vec::as_slice)
    }

    /// 词片字节 → 编号；未命中返回 [`None`]。
    #[must_use]
    pub fn lookup(&self, piece: &[u8]) -> Option<Token> {
        self.piece_map.get(piece).map(|&id| Token::new(id))
    }

    /// 编号是否命中已登记的特殊 token。
    #[must_use]
    pub fn is_special(&self, id: u32) -> bool {
        self.special.is_special(id)
    }
}

/// 把合并对按表内顺序（即秩序）折叠进指纹；两侧先长度前缀再字节，避免边界串扰。
fn fold_merges(mut fingerprint: Fingerprint, merges: &[Merge]) -> Fingerprint {
    for merge in merges {
        fingerprint = fingerprint.fold_extra(&merge.left.len().to_le_bytes());
        fingerprint = fingerprint.fold_extra(&merge.left);
        fingerprint = fingerprint.fold_extra(&merge.right.len().to_le_bytes());
        fingerprint = fingerprint.fold_extra(&merge.right);
    }
    fingerprint
}

/// 把逐片分数按下标顺序折叠进指纹。
fn fold_scores(mut fingerprint: Fingerprint, scores: &[f32]) -> Fingerprint {
    for score in scores {
        fingerprint = fingerprint.fold_extra(&score.to_le_bytes());
    }
    fingerprint
}

/// 建反向查表；重复词片或编号溢出报 [`Code::Invalid`]。
fn build_map(pieces: &[Vec<u8>]) -> Result<HashMap<Vec<u8>, u32>> {
    let mut map = HashMap::with_capacity(pieces.len());
    for (index, piece) in pieces.iter().enumerate() {
        let id = u32::try_from(index).map_err(|_| {
            Error::new(Code::Invalid, "词表编号超出 u32 范围").with("tokenizer::Vocab::try_new")
        })?;
        if map.insert(piece.clone(), id).is_some() {
            return Err(Error::new(Code::Invalid, format!("词片重复：编号 {index}"))
                .with("tokenizer::Vocab::try_new"));
        }
    }
    Ok(map)
}

/// 特殊编号必须落在词表内；越界报 [`Code::Invalid`]。
fn check_special(special: &SpecialTokens, len: usize) -> Result<()> {
    let slots = [
        special.bos(),
        special.eos(),
        special.unk(),
        special.pad(),
        special.sep(),
        special.cls(),
        special.mask(),
    ];
    for token in slots.into_iter().flatten() {
        if usize::try_from(token.id()).map_or(true, |id| id >= len) {
            return Err(Error::new(
                Code::Invalid,
                format!("特殊编号 {} 超出词表（0..{len}）", token.id()),
            )
            .with("tokenizer::Vocab::try_new"));
        }
    }
    Ok(())
}

/// 族与附属表一致：缺表报 [`Code::Missing`]，错配报 [`Code::Invalid`]。
fn check_attachments(
    family: VocabFamily,
    scores: &[f32],
    merges: &[Merge],
    piece_map: &HashMap<Vec<u8>, u32>,
    len: usize,
) -> Result<()> {
    check_merges(family, merges, piece_map)?;
    check_scores(family, scores, len)?;
    Ok(())
}

/// 合并表只属于 BPE：缺表报 [`Code::Missing`]，错挂报 [`Code::Invalid`]；
/// 两侧必须命中词表，且同一对不得重复登记（后者只留后登记秩会静默改写秩序）。
fn check_merges(
    family: VocabFamily,
    merges: &[Merge],
    piece_map: &HashMap<Vec<u8>, u32>,
) -> Result<()> {
    if family.requires_merges() && merges.is_empty() {
        return Err(
            Error::new(Code::Missing, "BPE 词表缺少合并表").with("tokenizer::Vocab::try_new")
        );
    }
    if !family.requires_merges() && !merges.is_empty() {
        return Err(
            Error::new(Code::Invalid, "非 BPE 词表携带了合并表").with("tokenizer::Vocab::try_new")
        );
    }
    check_merge_sides(merges, piece_map)
}

/// 逐条解析合并双侧编号并拒绝重复对；按秩下标定位报错位置。
fn check_merge_sides(merges: &[Merge], piece_map: &HashMap<Vec<u8>, u32>) -> Result<()> {
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for (rank, merge) in merges.iter().enumerate() {
        let pair = resolve_side_pair(rank, merge, piece_map)?;
        if !seen.insert(pair) {
            return Err(Error::new(
                Code::Invalid,
                format!("合并表第 {rank} 条与此前登记的合并对重复"),
            )
            .with("tokenizer::Vocab::try_new"));
        }
    }
    Ok(())
}

/// 取合并双侧的词表编号；任一侧未命中报 [`Code::Invalid`]。
fn resolve_side_pair(
    rank: usize,
    merge: &Merge,
    piece_map: &HashMap<Vec<u8>, u32>,
) -> Result<(u32, u32)> {
    let left = piece_map.get(&merge.left).ok_or_else(|| {
        Error::new(Code::Invalid, format!("合并表第 {rank} 条左片不在词表内"))
            .with("tokenizer::Vocab::try_new")
    })?;
    let right = piece_map.get(&merge.right).ok_or_else(|| {
        Error::new(Code::Invalid, format!("合并表第 {rank} 条右片不在词表内"))
            .with("tokenizer::Vocab::try_new")
    })?;
    Ok((*left, *right))
}

/// 分数表只属于 Unigram，且个数与词片齐平；错挂或不齐报 [`Code::Invalid`]。
fn check_scores(family: VocabFamily, scores: &[f32], len: usize) -> Result<()> {
    if family.requires_scores() && scores.len() != len {
        return Err(Error::new(
            Code::Invalid,
            format!("Unigram 分数个数 {} 与词片数 {len} 不齐", scores.len()),
        )
        .with("tokenizer::Vocab::try_new"));
    }
    if !family.requires_scores() && !scores.is_empty() {
        return Err(Error::new(Code::Invalid, "非 Unigram 词表携带了分数表")
            .with("tokenizer::Vocab::try_new"));
    }
    Ok(())
}
