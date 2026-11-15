//! 词表身份指纹：实验记录与对拍时标识一份词表，不承担密码学摘要职责。

/// FNV-1a 64 位偏移基数。
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64 位质数。
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// 一份词表的只读身份：族名、词表大小与描述摘要的非加密摘要。
///
/// 同族同描述产出同一指纹；词表内容或算法标识一变，摘要随之改变。
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
