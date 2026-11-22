//! 计数器型确定性随机：Philox4x32-10 引擎与三个构造期校验参数的分布闭集。
//!
//! 引擎是**无状态纯派生**：[`block`] 由 `(key, stream, idx)` 直接算出 4 个 u32，[`Rng`] 只是把
//! 它缓冲成顺序流；不同 `stream` 是互不相关的两条序列，可并行填充不同张量。分布层把
//! `u ∈ [0,1) → 元素` 收成纯映射，参数只在构造期校验一次，采样路径不再做任何分支检查。
//!
//! 这条模块与训练循环没有耦合：它只回答"确定性随机"一个词，`serve`（多项式采样）与
//! `fuzz`（语义 oracle）是它的消费者，`train` 只是合并后的宿主。

use diagnostic::{Code, Error, Result};

/// 本模块诊断站点。
const SITE: &str = "train::rng";

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
#[inline]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let product = u64::from(a) * u64::from(b);
    (product as u32, (product >> 32) as u32)
}

/// 一轮 S-box：两个独立槽位各做一次乘法拆位，再异或轮密钥与同位旧值。
#[inline]
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
#[inline]
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
    #[inline]
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

    /// 下一个 u64：高 32 位先行，与两个连续 u32 拼接逐位一致。
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        (u64::from(self.next_u32()) << 32) | u64::from(self.next_u32())
    }

    /// 下一个 `[0, 1)` 内的 f32；取高 24 位映射到 2⁻²⁴ 网格（f32 尾数只有 24 位，多取一位也存不下）。
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// 下一个 `[0, 1)` 内的 f64；取 53 位有效位映射到 2⁻⁵³ 网格。
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// 把 `out` 当作顺序流的连续词批量填满：整块走 [`block`] 直填，绕开逐词缓冲分支。
    pub fn fill(&mut self, out: &mut [u32]) {
        let head = (4 - self.slot).min(out.len());
        for slot in &mut out[..head] {
            *slot = self.next_u32();
        }
        let mut rest = &mut out[head..];
        while rest.len() >= 4 {
            rest[..4].copy_from_slice(&block(self.key, self.stream, self.idx));
            self.idx = self.idx.wrapping_add(1);
            rest = &mut rest[4..];
        }
        for slot in rest {
            *slot = self.next_u32();
        }
    }
}

/// 校验谓词不成立时返回错误；成立为 `Ok`。分布参数校验共用这一条出口。
fn ensure(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::new(Code::Invalid, message).with(SITE))
    }
}

/// 均匀分布 `[lo, hi)`。
pub struct Uniform {
    lo: f32,
    hi: f32,
}

impl Uniform {
    /// 校验端点有限且 `lo < hi`；相等、空区间、NaN 或无穷都拒绝。
    pub fn new(lo: f32, hi: f32) -> Result<Self> {
        ensure(lo.is_finite() && hi.is_finite(), "均匀区间端点必须有限")?;
        ensure(lo < hi, "均匀区间必须满足 lo < hi")?;
        Ok(Self { lo, hi })
    }

    /// 抽一个 `[lo, hi)` 内的 f32。
    #[inline]
    pub fn sample(&self, rng: &mut Rng) -> f32 {
        self.lo + (self.hi - self.lo) * rng.next_f32()
    }
}

/// 伯努利分布：以概率 `p` 返回 `true`。
pub struct Bernoulli {
    p: f32,
}

impl Bernoulli {
    /// 校验概率落在 `[0, 1]` 内且有限。
    pub fn new(p: f32) -> Result<Self> {
        ensure(
            p.is_finite() && (0.0..=1.0).contains(&p),
            "伯努利概率必须是 [0,1] 内的有限值",
        )?;
        Ok(Self { p })
    }

    /// 抽一次试验结果。
    #[inline]
    pub fn sample(&self, rng: &mut Rng) -> bool {
        rng.next_f32() < self.p
    }
}

/// 正态分布 `N(mean, std²)`。
pub struct Normal {
    mean: f32,
    std: f32,
}

impl Normal {
    /// 校验均值有限、标准差为正且有限。
    pub fn new(mean: f32, std: f32) -> Result<Self> {
        ensure(mean.is_finite() && std.is_finite(), "正态参数必须有限")?;
        ensure(std > 0.0, "标准差必须为正数")?;
        Ok(Self { mean, std })
    }

    /// Box-Muller 变换抽一个标准正态样本。
    #[inline]
    pub fn sample(&self, rng: &mut Rng) -> f32 {
        let u1 = rng.next_f32().max(f32::MIN_POSITIVE);
        let u2 = rng.next_f32();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        self.mean + self.std * radius * theta.cos()
    }
}

#[cfg(test)]
#[path = "../tests/rng_tests.rs"]
mod tests;
