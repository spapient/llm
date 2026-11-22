//! Philox4x32-10 计数器型伪随机引擎（CBRNG）。

/// 第一槽乘法常数。
const M0: u32 = 0xD251_1F53;
/// 第一槽轮常数。
const C0: u32 = 0x9E37_79B9;
/// 第三槽乘法常数。
const M1: u32 = 0xCD9E_8D57;
/// 第三槽轮常数。
const C1: u32 = 0xBB67_AE85;
/// 轮数：高于通过已知统计检验所需的下限，是实践通行取值。
const ROUNDS: usize = 10;

/// 32×32→64 乘法，返回（低 32 位，高 32 位）。
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let product = u64::from(a) * u64::from(b);
    (product as u32, (product >> 32) as u32)
}

/// 一轮 S-box：两个独立槽位各做一次乘法拆位，再异或轮密钥与同位旧值。
fn round(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (lo0, hi0) = mulhilo(M0, counter[0]);
    let (lo1, hi1) = mulhilo(M1, counter[2]);
    [
        hi1 ^ key[1] ^ counter[1],
        lo1 ^ key[0],
        hi0 ^ key[1] ^ counter[3],
        lo0,
    ]
}

/// 轮密钥按常数递推：每轮两个 32 位密钥各自模 2^32 累加轮常数。
fn advance_key(key: &mut [u32; 2]) {
    key[0] = key[0].wrapping_add(C0);
    key[1] = key[1].wrapping_add(C1);
}

/// 纯计数器派生：按 `key + stream + idx` 一次性产出 4 个 u32，不推进任何状态。
#[must_use]
pub fn block(key: u64, stream: u32, idx: u64) -> [u32; 4] {
    let mut counter = [stream, idx as u32, (idx >> 32) as u32, 0];
    let mut key_words = [key as u32, (key >> 32) as u32];
    for _ in 0..ROUNDS {
        counter = round(counter, key_words);
        advance_key(&mut key_words);
    }
    counter
}

/// 顺序推进的确定性流：持有种子、流号与当前下标，缓冲一个 4 词块。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rng {
    key: u64,
    stream: u32,
    idx: u64,
    buf: [u32; 4],
    slot: usize,
}

impl Rng {
    /// 固定流号 0 的流。
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self::with_stream(seed, 0)
    }

    /// 指定独立流号：不同 `stream` 是互不相关的两条序列，可并行填充不同张量/序列。
    #[must_use]
    pub fn with_stream(seed: u64, stream: u32) -> Self {
        Self {
            key: seed,
            stream,
            idx: 0,
            buf: [0; 4],
            slot: 4,
        }
    }

    /// 下一个 u32；缓冲耗尽时下一块按当前下标派生并推进下标。
    pub fn next_u32(&mut self) -> u32 {
        if self.slot >= 4 {
            self.buf = block(self.key, self.stream, self.idx);
            self.idx = self.idx.wrapping_add(1);
            self.slot = 0;
        }
        let value = self.buf[self.slot];
        self.slot += 1;
        value
    }

    /// 下一个 `[0, 1)` 内的 f32；均匀量在 f32 表示，逆变换由分布层完成。
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1 << 24) as f32
    }
}

#[cfg(test)]
#[path = "../tests/philox_tests.rs"]
mod tests;
