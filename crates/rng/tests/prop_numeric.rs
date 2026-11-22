//! `llm-rng` 的性质测试：计数器型引擎的代数定律与分布的端点定律。
//!
//! 这里只写**从定义推出**的性质，不重述实现：
//!
//! * 计数器型引擎的定义决定了 `block` 是 `(key, stream, idx)` 的**纯函数**，
//!   而 `Rng::next_u32` 只是按同一函数顺序取词——所以两者必须逐词一致。
//!   这条跨 API 定律是本文件的核心：只测其中一侧时，缓冲推进错位可以长期不被发现。
//! * `next_f32` 的定义是 `(u >> 8) / 2^24`，所以它的像集是 [0,1) 上的二进网格，
//!   且**取不到 1**。伯努利的 `p = 1` 必定成功、`p = 0` 必定失败就是这条的直接推论；
//!   写成理想化的 `u <= p` 会在 `p = 0` 处偶尔为真，这正是要钉住的缺陷。
//!
//! 统计档（频率、矩）用固定的大样本量：引擎是确定性的，因此给定输入后结果是常量，
//! 不存在"这条测试偶尔红"的情形，容差可以按 CLT 收紧而不必留陈旧的宽裕量。

use crate::dist::{Bernoulli, Normal, Uniform};
use crate::philox::{Rng, block};
use proptest::prelude::*;

/// 统计档的样本量：`sd(mean) = 1/√N ≈ 0.011`，足以让 0.05 的容差落在 4σ 之外。
const N: usize = 8_192;

/// 2^24：`next_f32` 的网格步长。
const GRID: f32 = 16_777_216.0;

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
