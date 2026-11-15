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

/// FNV-1a 64 位：逐字节异或再乘质数；串行依赖来自哈希链，无法整块并行。
fn fnv1a(text: &str) -> u64 {
    let mut hash = FNV_OFFSET;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
