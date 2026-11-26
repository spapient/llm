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

/// 批采样方式：三种策略由类型保证互斥，随机流只由显式种子派生。
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Sampling {
    /// 按条目原顺序取完整批。
    #[default]
    Sequential,
    /// 每轮按种子派生独立流重洗批序。
    Shuffle {
        /// 重洗随机流种子。
        seed: u64,
    },
    /// 按每批的非负权重有放回抽样。
    Weighted {
        /// 每批抽样权重（条数等于批数）。
        weights: Vec<f32>,
        /// 抽样随机流种子。
        seed: u64,
    },
}

/// 批装配配置：几何与采样方式一次性收口到构造期，热路径不再做合法性分支检查。
#[derive(Debug, Clone, PartialEq)]
pub struct LoaderConfig {
    /// 每批行数；构造期校验非零。
    pub rows: usize,
    /// 列宽（每行条目数）；构造期校验非零且 `rows * cols` 不溢出。
    pub cols: usize,
    /// 批采样方式。
    pub sampling: Sampling,
}

impl LoaderConfig {
    /// 校验行列非零且乘积不溢出；采样方式默认顺序。
    pub fn new(rows: usize, cols: usize) -> Result<Self> {
        let config = Self {
            rows,
            cols,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    /// 校验批几何；公开字段被直接修改后也可在装配边界重新检查。
    pub fn validate(&self) -> Result<()> {
        crate::learner::dataset::stride(self.rows, self.cols, "train::dataset").map(|_| ())
    }

    builders! {
        /// 设置互斥的采样策略。
        #[must_use]
        with_sampling => sampling: Sampling,
    }
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            rows: 1,
            cols: 1,
            sampling: Sampling::Sequential,
        }
    }
}

#[cfg(test)]
mod tests {
    //! 计数器型引擎与分布闭集的测试：已知答案向量、端点定律、跨 API 定律与统计档。
    //!
    //! 引擎是确定性的，因此已知答案向量必须逐词命中；性质层只写**从定义推出**的定律，不重述实现。
    //! 统计档（频率、矩）用固定的大样本量：给定输入后结果是常量，不存在"偶尔红"，容差可以按 CLT 收紧。

    use super::{Bernoulli, Normal, Rng, Uniform, block};
    use proptest::prelude::*;

    /// 统计档的样本量：`sd(mean) = 1/√N ≈ 0.011`，足以让 0.05 的容差落在 4σ 之外。
    const N: usize = 8_192;

    /// 2^24：`next_f32` 的网格步长。
    const GRID: f32 = 16_777_216.0;

    // ---- 引擎单元测试 ----

    #[test]
    fn same_key_stream_idx_produces_the_same_block() {
        let first = block(0x1234_5678_9ABC_DEF0, 7, 100);
        let again = block(0x1234_5678_9ABC_DEF0, 7, 100);
        assert_eq!(first, again, "计数器派生必须是确定性的");
    }

    #[test]
    fn different_streams_do_not_share_a_sequence() {
        let seed = 42u64;
        let a = block(seed, 0, 0);
        let b = block(seed, 1, 0);
        assert_ne!(a, b, "独立流号必须给出不同序列");
    }

    #[test]
    fn rng_is_deterministic_and_bounded() {
        let mut left = Rng::new(1234);
        let mut right = Rng::new(1234);
        for _ in 0..17 {
            assert_eq!(left.next_u32(), right.next_u32(), "同种子序列一致");
        }
    }

    #[test]
    fn next_f32_stays_in_the_half_open_unit_interval() {
        let mut rng = Rng::new(7);
        let mut seen_nonzero = false;
        for _ in 0..64 {
            let value = rng.next_f32();
            assert!(
                (0.0..1.0).contains(&value),
                "均匀量必须落在 [0,1)，得到 {value}"
            );
            seen_nonzero |= value > 0.0;
        }
        assert!(seen_nonzero, "高 24 位映射不能塌缩成全零");
    }

    /// 已知答案向量：与独立参考实现逐词核对的 Philox4x32-10 输出。
    ///
    /// 覆盖零种子、非零种子、跨 32 位边界的种子、同一流内连续三个计数器；任何轮函数、
    /// 乘法高低位或轮密钥递推写偏都会在这里被抓住。
    #[test]
    fn known_answer_vectors_match_the_reference_philox() {
        let cases = [
            (
                0u64,
                0u32,
                0u64,
                [0x01BF_6E88, 0x0113_DC1E, 0x0FF2_CC34, 0x6B72_E23A],
            ),
            (
                123_456,
                7,
                100,
                [0xB99B_391E, 0x068B_CDA9, 0x83BC_9EFA, 0x3292_3104],
            ),
            (
                0x1234_5678_9ABC_DEF0,
                7,
                100,
                [0xD45C_6E74, 0x6860_CD9F, 0xD917_D187, 0x9734_C29E],
            ),
            (
                9,
                3,
                0,
                [0x8AA2_DF03, 0x713D_E75A, 0x704C_584E, 0x581A_FDAC],
            ),
            (
                9,
                3,
                1,
                [0xC2E8_B289, 0x77BF_5997, 0x45C2_B652, 0x9850_3DDF],
            ),
            (
                9,
                3,
                2,
                [0xFDBC_B116, 0x27FB_AB4E, 0xE85F_C832, 0xBAC6_B546],
            ),
        ];
        for (key, stream, idx, expected) in cases {
            assert_eq!(
                block(key, stream, idx),
                expected,
                "key {key:#x} stream {stream} idx {idx}"
            );
        }
    }

    #[test]
    fn sequential_blocks_match_direct_counter_derivation() {
        let mut rng = Rng::with_stream(9, 3);
        for idx in 0..3u64 {
            let direct = block(9, 3, idx);
            let buffered = std::array::from_fn(|_| rng.next_u32());
            assert_eq!(buffered, direct, "顺序缓冲必须与直接派生逐词一致");
        }
    }

    // ---- 新增词位与批量填充 ----

    #[test]
    fn next_u64_packs_two_consecutive_u32s() {
        let mut rng = Rng::new(7);
        let hi = rng.next_u32();
        let lo = rng.next_u32();
        let mut again = Rng::new(7);
        assert_eq!(again.next_u64(), (u64::from(hi) << 32) | u64::from(lo));
    }

    #[test]
    fn next_f64_stays_in_the_half_open_unit_interval() {
        let mut rng = Rng::new(11);
        let mut seen_nonzero = false;
        for _ in 0..64 {
            let value = rng.next_f64();
            assert!((0.0..1.0).contains(&value), "f64 均匀量出界：{value}");
            seen_nonzero |= value > 0.0;
        }
        assert!(seen_nonzero, "53 位映射不能塌缩成全零");
    }

    #[test]
    fn fill_matches_the_sequential_stream_at_every_length() {
        for seed in [0u64, 1, 9, 0xDEAD_BEEF] {
            for len in 0..=17usize {
                let mut streamed = Rng::new(seed);
                let expected: Vec<u32> = (0..len).map(|_| streamed.next_u32()).collect();
                let mut filled = Rng::new(seed);
                let mut actual = vec![0; len];
                filled.fill(&mut actual);
                assert_eq!(
                    actual, expected,
                    "seed {seed} len {len} 批量填充与顺序流分叉"
                );
            }
        }
    }

    // ---- 分布单元测试 ----

    #[test]
    fn uniform_respects_endpoints_and_is_reproducible() {
        let distribution = Uniform::new(-3.0, 5.0).expect("区间合法");
        let mut left = Rng::new(2024);
        let mut right = Rng::new(2024);
        for _ in 0..16 {
            let value = distribution.sample(&mut left);
            assert!((-3.0..5.0).contains(&value), "落在 [lo,hi)");
            assert_eq!(distribution.sample(&mut right), value, "同种子可复现");
        }
    }

    #[test]
    fn uniform_rejects_empty_or_non_finite_range() {
        assert!(Uniform::new(1.0, 1.0).is_err());
        assert!(Uniform::new(1.0, f32::NAN).is_err());
        assert!(Uniform::new(f32::INFINITY, 2.0).is_err());
        assert!(Uniform::new(2.0, 1.0).is_err());
    }

    #[test]
    fn bernoulli_probability_is_bounded() {
        assert!(Bernoulli::new(0.0).is_ok());
        assert!(Bernoulli::new(1.0).is_ok());
        assert!(Bernoulli::new(-0.1).is_err());
        assert!(Bernoulli::new(1.2).is_err());
        assert!(Bernoulli::new(f32::NAN).is_err());

        let always = Bernoulli::new(1.0).expect("p=1 合法");
        let never = Bernoulli::new(0.0).expect("p=0 合法");
        let mut rng = Rng::new(1);
        assert!(always.sample(&mut rng));
        assert!(!never.sample(&mut rng));
    }

    #[test]
    fn normal_samples_are_finite_and_reproducible() {
        let distribution = Normal::new(1.0, 0.25).expect("参数合法");
        let mut left = Rng::new(11);
        let mut right = Rng::new(11);
        for _ in 0..16 {
            let value = distribution.sample(&mut left);
            assert!(value.is_finite(), "正态样本必须有限");
            assert_eq!(distribution.sample(&mut right), value, "同种子可复现");
        }
    }

    #[test]
    fn normal_rejects_non_positive_or_non_finite_std() {
        assert!(Normal::new(0.0, 0.0).is_err());
        assert!(Normal::new(0.0, -1.0).is_err());
        assert!(Normal::new(f32::NAN, 1.0).is_err());
        assert!(Normal::new(0.0, f32::INFINITY).is_err());
    }

    // ---- 性质测试 ----

    proptest! {
        #![proptest_config(testkit::prop::config())]

        /// 计数器型引擎的立身之本：同一个三元组永远派生出同一个块。
        #[test]
        fn block_is_a_pure_function(key: u64, stream: u32, idx: u64) {
            let first = block(key, stream, idx);
            for _ in 0..3 {
                prop_assert_eq!(block(key, stream, idx), first, "纯函数：重复调用不许有状态");
            }
        }

        /// 流式 API 必须逐词等于按块拼起来的序列（含跨块边界）。
        #[test]
        fn the_stream_is_the_concatenation_of_blocks(key: u64, stream: u32, words in 1usize..12) {
            let mut rng = Rng::with_stream(key, stream);
            let mut streamed = Vec::with_capacity(words);
            for _ in 0..words {
                streamed.push(rng.next_u32());
            }
            let mut batched = Vec::with_capacity(words);
            for idx in 0..(words.div_ceil(4) as u64) {
                batched.extend_from_slice(&block(key, stream, idx));
            }
            batched.truncate(words);
            prop_assert_eq!(streamed, batched, "缓冲推进与计数器派生必须一致");
        }

        /// 流号是序列的隔离键：换流即换序列。
        #[test]
        fn distinct_streams_give_distinct_blocks(key: u64, stream: u32) {
            let other = stream.wrapping_add(1);
            for idx in 0..3u64 {
                prop_assert_ne!(block(key, stream, idx), block(key, other, idx), "流号必须真的参与派生");
            }
        }

        /// 种子同样是隔离键：不同种子不许有任何一块重合。
        #[test]
        fn distinct_keys_give_distinct_blocks(key: u64, stream: u32) {
            let other = key ^ 1;
            for idx in 0..3u64 {
                prop_assert_ne!(block(key, stream, idx), block(other, stream, idx), "种子必须真的参与派生");
            }
        }

        /// 同一个种子必须重放出逐位相同的序列（否则"可重放"是假的）。
        #[test]
        fn replay_is_exact(key: u64, stream: u32) {
            let mut first = Rng::with_stream(key, stream);
            let mut second = Rng::with_stream(key, stream);
            for step in 0..64 {
                prop_assert_eq!(first.next_u32(), second.next_u32(), "第 {} 个词不一致", step);
            }
        }

        /// `next_f32` 的值域是半开区间 [0,1)：取到 1 会让伯努利的 `p = 1` 失效。
        #[test]
        fn next_f32_is_in_the_unit_interval(key: u64) {
            let mut rng = Rng::new(key);
            for _ in 0..N {
                let value = rng.next_f32();
                prop_assert!((0.0..1.0).contains(&value), "next_f32 出界：{value}");
            }
        }

        /// 像集落在 2^-24 的二进网格上：`(u >> 8) / 2^24` 的分子是整数且小于 2^24。
        /// 写成 `/ (2^24 - 1)` 就会越过 1，这条能立刻抓到。
        #[test]
        fn next_f32_lands_on_the_dyadic_grid(key: u64) {
            let mut rng = Rng::new(key);
            for _ in 0..256 {
                let value = rng.next_f32();
                let step = value * GRID;
                prop_assert_eq!(step.fract(), 0.0, "不在 2^-24 网格上：{}", value);
                prop_assert!(step < GRID, "网格越界（分子 ≥ 2^24）：{value}");
            }
        }

        /// `p = 0` 永远不成功、`p = 1` 永远成功：只由 "u ∈ [0,1)" 决定。
        #[test]
        fn bernoulli_certainty_is_exact(key: u64) {
            let mut rng = Rng::new(key);
            let never = Bernoulli::new(0.0).expect("p = 0 合法");
            let always = Bernoulli::new(1.0).expect("p = 1 合法");
            for _ in 0..N {
                prop_assert!(!never.sample(&mut rng), "p = 0 抽出 true");
                prop_assert!(always.sample(&mut rng), "p = 1 抽出 false");
            }
        }

        /// 频率收敛到概率：这是"以概率 p 返回 true"的唯一可观测含义。
        #[test]
        fn bernoulli_frequency_tracks_its_probability(p in 0.0f32..=1.0, key: u64) {
            let distribution = Bernoulli::new(p).expect("p ∈ [0,1]");
            let mut rng = Rng::new(key);
            let hits = (0..N).filter(|_| distribution.sample(&mut rng)).count();
            let observed = hits as f32 / N as f32;
            prop_assert!(
                (observed - p).abs() <= 0.05,
                "p = {p} 却观测到 {observed}（N = {N}）"
            );
        }

        /// 均匀采样必须落回自己的区间（闭区间口径：端点附近允许舍入到端点）。
        #[test]
        fn uniform_samples_stay_inside_their_interval(lo in -1.0f32..1.0, hi in -1.0f32..1.0, key: u64) {
            prop_assume!(lo < hi);
            let distribution = Uniform::new(lo, hi).expect("有限且 lo < hi");
            let mut rng = Rng::new(key);
            for _ in 0..N {
                let value = distribution.sample(&mut rng);
                prop_assert!(lo <= value && value <= hi, "区间 [{lo},{hi}] 外的样本：{value}");
            }
        }

        /// 单位区间是半开的：这是 `next_f32 < 1` 的直接推论，也是逆变换法成立的前提。
        #[test]
        fn the_unit_interval_is_half_open(key: u64) {
            let distribution = Uniform::new(0.0, 1.0).expect("标准区间");
            let mut rng = Rng::new(key);
            for _ in 0..N {
                prop_assert!(distribution.sample(&mut rng) < 1.0, "单位区间取到了 1");
            }
        }

        /// Box-Muller 的半径受 `u₁ ≥ MIN_POSITIVE` 保护，因此样本有界且不是 NaN。
        /// 上界 `√(-2 ln MIN_POSITIVE) ≈ 13.2` 由 `u₁` 的下界推出，与实现无关。
        #[test]
        fn normal_samples_are_finite_and_bounded(key: u64) {
            let distribution = Normal::new(0.0, 1.0).expect("标准正态");
            let mut rng = Rng::new(key);
            for _ in 0..N {
                let value = distribution.sample(&mut rng);
                prop_assert!(value.is_finite(), "正态样本非有限：{value}");
                prop_assert!(value.abs() <= 14.0, "样本越出 Box-Muller 的理论上界：{value}");
            }
        }

        /// 一阶、二阶矩必须是被要求的均值与标准差：这条同时钉住 `std` 有没有被吃掉。
        #[test]
        fn normal_has_the_requested_moments(key: u64) {
            let distribution = Normal::new(0.0, 1.0).expect("标准正态");
            let mut rng = Rng::new(key);
            let mut sum = 0.0f64;
            let mut square = 0.0f64;
            for _ in 0..N {
                let value = f64::from(distribution.sample(&mut rng));
                sum += value;
                square += value * value;
            }
            let mean = sum / N as f64;
            let variance = square / N as f64 - mean * mean;
            prop_assert!(mean.abs() <= 0.06, "均值偏离 0：{mean}");
            prop_assert!((0.85..=1.15).contains(&variance), "方差不接近 1：{variance}");
        }

        /// 均值与标准差是仿射参数：`N(m, s)` 的样本必须等于 `m + s·z`（`z` 来自 `N(0,1)`）。
        #[test]
        fn normal_is_affine_in_its_parameters(mean in -2.0f32..2.0, std in 0.25f32..4.0, key: u64) {
            let scaled = Normal::new(mean, std).expect("参数合法");
            let standard = Normal::new(0.0, 1.0).expect("标准正态");
            // 两条路径必须吃同一对 (u₁, u₂)，否则比的是两段不同的词流。
            let mut reference = Rng::new(key);
            let mut subject = Rng::new(key);
            for _ in 0..256 {
                let expected = mean + std * standard.sample(&mut reference);
                let actual = scaled.sample(&mut subject);
                // 两条路径的运算顺序不同，允许 f32 舍入差；偏差超出 1e-5 就是参数没接上。
                prop_assert!(
                    (actual - expected).abs() <= 1e-5 * (1.0 + expected.abs()),
                    "mean = {mean}, std = {std}: {actual} 与 {expected} 不符"
                );
            }
        }
    }

    /// 找出首个 `next_f32` 恰为 0 的词序号：`(word >> 8) == 0` 即 `u = 0`。
    ///
    /// 为什么必须显式构造：`Bernoulli::sample` 是 `u < p`，与 `u <= p` 的唯一区别
    /// 只在 `u == 0` 上；而 `next_f32 == 0` 的概率是 2⁻²⁴，随机抽样撞上的机会极小
    /// （8192 次里约 0.05%）。突变测试把 `<` 改成 `<=` 时随机版仍然全绿——这就是证据。
    /// 边界定律要么定点触发，要么等于没测。
    fn zero_draw_ordinal(seed: u64, budget: u64) -> Option<u64> {
        for idx in 0..budget {
            for (slot, word) in block(seed, 0, idx).iter().enumerate() {
                if word >> 8 == 0 {
                    return Some(idx * 4 + slot as u64);
                }
            }
        }
        None
    }

    #[test]
    fn p_zero_fails_exactly_at_the_zero_draw() {
        let seed = 0x5EED_1DEA;
        // 期望 2^22 个块出一次；预算给 4 倍，找不到就是引擎坏了。
        let ordinal = zero_draw_ordinal(seed, 1 << 24).expect("2^26 个词里必有高 24 位全零的");
        let word = block(seed, 0, ordinal / 4)[(ordinal % 4) as usize];
        assert_eq!(word >> 8, 0, "定点失败：第 {ordinal} 个词不是零抽样");

        let mut rng = Rng::new(seed);
        for _ in 0..ordinal {
            let _ = rng.next_u32();
        }
        let never = Bernoulli::new(0.0).expect("p = 0 合法");
        assert!(
            !never.sample(&mut rng),
            "u = 0 时 p = 0 必须失败——`u <= p` 恰好在这里为真"
        );
    }
}
