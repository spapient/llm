//! Philox4x32-10 计数器型伪随机引擎（CBRNG）。
//!
//! 随机流由三个显式输入派生，不持有全局可变状态、也不读线程熵源：
//!
//! - `key`：一次实验/部署的种子；
//! - `stream`：独立流号（每条生成序列、每组掩码、每层初始化各占一号）；
//! - `index`：流内元素偏移，张量内按全局下标寻址。
//!
//! CPU 顺序模式（[`Rng`]）与 GPU 按索引直接寻址（[`block`]）走同一个混合函数：
//! 同 `key + stream + index` 必然出同一个数，两条后端因此可以逐位对拍。
//!
//! 算法参数取自广泛采用的 Philox4x32-10 配置（cuRAND / MKL / rocRAND / cuPy 同族）：
//! 10 轮替换-置换网络，每轮一次 32×32→64 乘法拆高低位。

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

/// 纯计数器派生：按 `key + stream + index` 一次性产出 4 个 u32，不推进任何状态。
///
/// GPU 后端由线程/块全局下标直接映射到 `index` 取数；CPU 顺序模式也以同一函数填充
/// [`Rng`] 的缓冲。第四计数器槽固定为 0，留出用途域扩展位。
#[must_use]
pub fn block(key: u64, stream: u32, index: u64) -> [u32; 4] {
    let mut counter = [stream, index as u32, (index >> 32) as u32, 0];
    let mut key_words = [key as u32, (key >> 32) as u32];
    for _ in 0..ROUNDS {
        counter = round(counter, key_words);
        advance_key(&mut key_words);
    }
    counter
}

/// 顺序推进的确定性流：持有种子、流号与当前下标，缓冲一个 4 词块。
///
/// 序列随实例走；拷贝两个同种子实例会得到相同序列，但它们之间不共享可变状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rng {
    key: u64,
    stream: u32,
    index: u64,
    buffer: [u32; 4],
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
            index: 0,
            buffer: [0; 4],
            slot: 4,
        }
    }

    /// 下一个 u32；缓冲耗尽时下一块按当前下标派生并推进下标。
    pub fn next_u32(&mut self) -> u32 {
        if self.slot >= 4 {
            self.buffer = block(self.key, self.stream, self.index);
            self.index = self.index.wrapping_add(1);
            self.slot = 0;
        }
        let value = self.buffer[self.slot];
        self.slot += 1;
        value
    }

    /// 下一个 `[0, 1)` 内的 f32；均匀量在 f32 表示，逆变换由分布层完成。
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / 4_294_967_296.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_key_stream_index_produces_the_same_block() {
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
        for _ in 0..32 {
            let value = rng.next_f32();
            assert!((0.0..1.0).contains(&value), "均匀量必须落在 [0,1)");
        }
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
        for (key, stream, index, expected) in cases {
            assert_eq!(
                block(key, stream, index),
                expected,
                "key {key:#x} stream {stream} index {index}"
            );
        }
    }

    #[test]
    fn sequential_blocks_match_direct_counter_derivation() {
        let mut rng = Rng::with_stream(9, 3);
        for index in 0..3u64 {
            let direct = block(9, 3, index);
            let buffered = std::array::from_fn(|_| rng.next_u32());
            assert_eq!(buffered, direct, "顺序缓冲必须与直接派生逐词一致");
        }
    }
}
