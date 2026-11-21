//! 中性词表值对象：格式无关的词片表、特殊句柄与族附属数据。

use crate::{SpecialTokens, Token};
use diagnostic::{Code, Error, Result};
use std::collections::{HashMap, HashSet};

pub mod codec;
pub use fingerprint::Fingerprint;

/// 词表算法族。判别值即稳定 wire 序号，勿重排、勿改号。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VocabFamily {
    /// BPE：按合并对迭代归并相邻片段。
    Bpe = 0,
    /// Unigram：按片段分数做最优切分。
    Unigram = 1,
    /// WordPiece：前缀标记 + 词内子词贪心。
    WordPiece = 2,
    /// 整词词表：不做子词切分。
    WordLevel = 3,
    /// 字节级基线：一字节一片。
    Byte = 4,
}

/// 族按 wire 序号的稳定表：`from_id` 的唯一事实源。
const FAMILY_BY_ID: [VocabFamily; 5] = [
    VocabFamily::Bpe,
    VocabFamily::Unigram,
    VocabFamily::WordPiece,
    VocabFamily::WordLevel,
    VocabFamily::Byte,
];

/// 族名按 wire 序号的稳定表：`as_str` 的唯一事实源。
const FAMILY_STR: [&str; 5] = ["bpe", "unigram", "wordpiece", "wordlevel", "byte"];

impl VocabFamily {
    /// 族名（实现标识，进词表指纹）。
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        FAMILY_STR[self as usize]
    }

    /// 稳定 wire 序号（中性字节布局用，区别于族名字符串）。
    pub(crate) const fn id(self) -> u8 {
        self as u8
    }

    /// 由 wire 序号取族；未知序号返回 `None`。
    pub(crate) const fn from_id(id: u8) -> Option<Self> {
        if (id as usize) < FAMILY_BY_ID.len() {
            Some(FAMILY_BY_ID[id as usize])
        } else {
            None
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
    pub fn into_tokenizer(self) -> Result<Box<dyn crate::Tokenizer>> {
        let family = self.family;
        let ctor = FAMILY_CTOR[family as usize].ok_or_else(|| unsupported(family))?;
        ctor(self)
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

/// "词表 → 分词器" 构造器：`FAMILY_CTOR[family as usize]` 为 `None` 表示族尚无实现。
type TokenizerCtor = fn(Vocab) -> Result<Box<dyn crate::Tokenizer>>;

/// 族 → 构造器的闭集扇出表（下标即 [`VocabFamily::id`]）。
const FAMILY_CTOR: [Option<TokenizerCtor>; 5] = [
    Some(bpe_ctor),
    Some(unigram_ctor),
    None,
    None,
    Some(bytes_ctor),
];

fn bpe_ctor(vocab: Vocab) -> Result<Box<dyn crate::Tokenizer>> {
    Ok(Box::new(crate::Bpe::new(vocab)?))
}

fn unigram_ctor(vocab: Vocab) -> Result<Box<dyn crate::Tokenizer>> {
    Ok(Box::new(crate::Unigram::new(vocab)?))
}

fn bytes_ctor(_vocab: Vocab) -> Result<Box<dyn crate::Tokenizer>> {
    Ok(Box::new(crate::Bytes))
}

fn unsupported(family: VocabFamily) -> Error {
    Error::new(
        Code::Unsupported,
        format!("词表族 {family:?} 尚无分词器实现"),
    )
    .with("tokenizer::Vocab::into_tokenizer")
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

mod fingerprint {
    //! 词表身份指纹：实验记录与对拍时标识一份词表，不承担密码学摘要职责。

    /// FNV-1a 64 位偏移基数。
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    /// FNV-1a 64 位质数。
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    /// 一份词表的只读身份：族名、词表大小与描述摘要的非加密摘要。
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Fingerprint {
        family: &'static str,
        vocab_size: usize,
        digest: u64,
    }

    impl Fingerprint {
        /// 由族名、词表大小与描述摘要构造；摘要进入 FNV-1a 混合。
        pub fn new(family: &'static str, vocab_size: usize, descriptor: &str) -> Self {
            let digest = fnv1a(&format!("{family};{vocab_size};{descriptor}"));
            Self {
                family,
                vocab_size,
                digest,
            }
        }

        /// 由族名、词表大小与词片序列构造；每个词片的长度与字节进入 FNV-1a 混合，
        /// 不依赖外部文件名或路径。
        pub fn from_pieces(family: &'static str, vocab_size: usize, pieces: &[Vec<u8>]) -> Self {
            let mut digest = mix(FNV_OFFSET, family.as_bytes());
            digest = mix(digest, &vocab_size.to_le_bytes());
            for piece in pieces {
                digest = mix(digest, &piece.len().to_le_bytes());
                digest = mix(digest, piece);
            }
            Self {
                family,
                vocab_size,
                digest,
            }
        }

        /// 在现有摘要上追加一段有序字节段，返回新指纹；用于把合并秩序、分数等附属内容纳入身份。
        #[must_use]
        pub fn fold_extra(&self, bytes: &[u8]) -> Self {
            Self {
                digest: mix(self.digest, bytes),
                family: self.family,
                vocab_size: self.vocab_size,
            }
        }

        /// 词表族名（实现标识，如 `bytes`、`bpe`、`unigram`）。
        #[must_use]
        pub fn family(&self) -> &'static str {
            self.family
        }

        /// 词表大小。
        #[must_use]
        pub fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        /// 64 位非加密摘要。
        #[must_use]
        pub fn digest(&self) -> u64 {
            self.digest
        }

        /// 摘要的 16 进制小写写法，便于写进实验记录。
        #[must_use]
        pub fn hex(&self) -> String {
            format!("{:016x}", self.digest)
        }
    }

    /// FNV-1a 64 位：在已有哈希状态上逐字节异或再乘质数；串行依赖来自哈希链，无法整块并行。
    fn mix(mut hash: u64, bytes: &[u8]) -> u64 {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// FNV-1a 64 位入口：从偏移基数起混合一段文本。
    fn fnv1a(text: &str) -> u64 {
        mix(FNV_OFFSET, text.as_bytes())
    }
}
